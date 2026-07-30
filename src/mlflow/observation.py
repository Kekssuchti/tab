from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import pstdev
from typing import Literal, cast

from src.classes.data_registry import dataset_task_for_target
from src.mlflow.serialization import canonical_json
from src.mlflow.tracking_contract import (
    METRIC_CV_TOTAL_TIME,
    METRIC_MODEL_TOTAL_TIME,
    METRIC_PIPELINE_TOTAL_TIME,
    METRIC_TRAIN_FIT_TIME,
    PARAM_DATASET_TARGET,
    PARAM_MODEL_BEST_PARAMS,
    PARAM_MODEL_TUNED,
    RUN_TYPE_CV_CANDIDATE,
    RUN_TYPE_MODEL,
    RUN_TYPE_PIPELINE,
    STATUS_FAILED,
    STATUS_SUCCESS,
    TAG_TRACKING_SCHEMA_VERSION,
    TAG_MODEL_INSTANCE,
    TAG_MODEL_NAME,
    TAG_PIPELINE_ID,
    TAG_RUN_TYPE,
    TAG_STATUS,
    TAG_TARGET,
    TAG_TASK_TYPE,
    TAG_TRAIN_SOURCES,
    TAG_TRAINED_ON,
    TRACKING_SCHEMA_VERSION,
    dataset_row_count_param,
    test_delta_metric,
    test_n_classes_param,
    test_predict_time_metric,
    test_score_ci_metric,
    test_score_metric,
)
from src.schemas.base_schemas import TaskType
from src.schemas.dataset_schemas import ClassificationTargetSummary, RegressionTargetSummary
from src.schemas.metrics import (
    BootstrapClassificationMetrics,
    BootstrapFinalTestMetrics,
    BootstrapRegressionMetrics,
    ClassificationMetrics,
    RegressionMetrics,
)
from src.schemas.pipeline_schemas import PipelineConfig
from src.schemas.run_records import (
    FoldRecord,
    ModelEvaluationRecord,
    ModelRunRecord,
    ModelTrainingResult,
    PipelineRunRecord,
    TuningRecord,
)
from src.schemas.training_schemas import ModelConfig
from src.utils.evaluation_utils import (
    classification_score,
    mean_classification_metrics,
    mean_regression_metrics,
    regression_score,
)
from src.utils.model_identity import model_instance_ids
from src.mlflow.validation import validate_pipeline_projection


@dataclass(frozen=True)
class MetricLog:
    """
    One metric value prepared for MLflow logging.

    ---
    Attributes:
        name: str
            Metric name.

        value: float
            Metric value.

        step: int or None, default=None
            Optional MLflow metric step.
    """

    name: str
    value: float
    step: int | None = None


@dataclass(frozen=True)
class EvaluationLog:
    """
    One MLflow evaluation payload.

    ---
    Attributes:
        inputs: dict
            Evaluation input descriptors.

        outputs: dict
            Evaluation output descriptors.

        targets: dict
            Target descriptors.

        metrics: dict
            Evaluation metrics.

        tags: dict
            Tags attached to the evaluation.
    """

    inputs: dict[str, str]
    outputs: dict[str, str]
    targets: dict[str, str]
    metrics: dict[str, float | int]
    tags: dict[str, str]


@dataclass(frozen=True)
class EvaluationTableRow:
    tracking_schema_version: str
    pipeline_run_id: str
    target: str
    trained_on: str
    model_name: str
    model_instance: str
    dataset: str
    scope: Literal["test", "test_delta"]
    metric: str
    value: float


@dataclass(frozen=True)
class RunObservation:
    """
    Serializable description of one MLflow run and its children.

    ---
    Attributes:
        run_name: str
            MLflow run name.

        tags: dict
            Run tags.

        params: dict
            Run parameters.

        metrics: tuple of MetricLog, default=()
            Metrics logged on the run.

        evaluations: tuple of EvaluationLog, default=()
            Evaluation payloads logged on the run.

        table_rows: tuple of EvaluationTableRow, default=()
            Rows for summary metric tables.

        children: tuple of RunObservation, default=()
            Nested child runs.

    """

    run_name: str
    tags: dict[str, str]
    params: dict[str, str]
    metrics: tuple[MetricLog, ...] = ()
    evaluations: tuple[EvaluationLog, ...] = ()
    table_rows: tuple[EvaluationTableRow, ...] = ()
    children: tuple[RunObservation, ...] = ()


@dataclass(frozen=True)
class _EvaluationBundle:
    """Grouped evaluation payloads and metric table rows."""

    evaluations: tuple[EvaluationLog, ...]
    table_rows: tuple[EvaluationTableRow, ...]


@dataclass(frozen=True)
class _CandidateSummary:
    """Aggregated CV summary for one tuning candidate."""

    candidate_index: int
    model_params: dict[str, object]
    folds: tuple[FoldRecord, ...]
    mean_score: float
    std_score: float
    mean_metrics: ClassificationMetrics | RegressionMetrics


def assemble_pipeline_observation(
    pipeline_config: PipelineConfig,
    pipeline_result: PipelineRunRecord,
) -> RunObservation:
    validate_pipeline_projection(pipeline_config, pipeline_result)
    model_params_by_id = _model_params_by_instance_id(pipeline_config.training)
    children = tuple(
        _model_run_observation(
            pipeline_config,
            run_record,
            model_params_by_id[run_record.model_instance_id],
        )
        for run_record in pipeline_result.model_runs
    )

    return RunObservation(
        run_name=_run_name(pipeline_config),
        tags=_pipeline_tags(pipeline_config),
        params={
            **_pipeline_params(pipeline_config),
            **_dataset_summary_params(pipeline_result),
            **_pipeline_model_config_params(pipeline_config),
        },
        metrics=_metric_logs_from_values(((METRIC_PIPELINE_TOTAL_TIME, pipeline_result.total_time),)),
        evaluations=tuple(evaluation for child in children for evaluation in child.evaluations),
        table_rows=tuple(row for child in children for row in child.table_rows),
        children=children,
    )


def table_rows_to_columns(
    rows: tuple[EvaluationTableRow, ...],
) -> dict[str, list[str | float]]:
    columns: dict[str, list[str | float]] = {
        "tracking_schema_version": [],
        "pipeline_run_id": [],
        "target": [],
        "trained_on": [],
        "model_name": [],
        "model_instance": [],
        "dataset": [],
        "scope": [],
        "metric": [],
        "value": [],
    }
    for row in rows:
        for key in columns:
            columns[key].append(getattr(row, key))
    return columns


def _model_run_observation(
    pipeline_config: PipelineConfig,
    run_record: ModelRunRecord,
    model_config: ModelConfig,
) -> RunObservation:
    training_result = run_record.training_result
    model_result = run_record.evaluation
    evaluation_bundle = _evaluation_bundle(pipeline_config, (run_record,))

    run_params = {
        **_model_run_params(model_config, training_result),
        **_model_metric_params(model_result),
    }
    if model_result is None:
        metrics = list(_metric_logs_from_values(((METRIC_TRAIN_FIT_TIME, training_result.fit_time),)))
        if training_result.tuning_result is not None:
            metrics.extend(
                _metric_logs_from_values(((METRIC_CV_TOTAL_TIME, training_result.tuning_result.total_time),))
            )
            children = _cv_candidate_observations(pipeline_config, run_record, model_config)
        else:
            children = ()
    else:
        metrics = _model_metric_logs(training_result, model_result)
        children = _cv_candidate_observations(pipeline_config, run_record, model_config)

    return RunObservation(
        run_name=run_record.model_instance_id,
        tags=_model_tags(pipeline_config, run_record, model_config),
        params=run_params,
        metrics=tuple(metrics),
        evaluations=evaluation_bundle.evaluations,
        table_rows=evaluation_bundle.table_rows,
        children=children,
    )


def _cv_candidate_observations(
    pipeline_config: PipelineConfig,
    run_record: ModelRunRecord,
    model_config: ModelConfig,
) -> tuple[RunObservation, ...]:
    tuning_result = run_record.training_result.tuning_result
    if tuning_result is None:
        return ()

    task_type = run_record.training_result.task_type
    candidate_summaries = _candidate_summaries(tuning_result, task_type)
    ranks = _candidate_ranks(
        [candidate.mean_score for candidate in candidate_summaries],
        maximize=tuning_result.scoring not in {"mae", "mse", "rmse"},
    )
    observations = []
    for position, candidate in enumerate(candidate_summaries):
        candidate_label = f"cv{candidate.candidate_index:02d}"
        metrics = [
            *_metric_logs_from_values(
                (
                    ("cv.rank", ranks[position]),
                    ("cv.mean_score", candidate.mean_score),
                    ("cv.std_score", candidate.std_score),
                )
            ),
            *_metric_logs("cv.mean", candidate.mean_metrics),
        ]
        for name, value in _metric_stds(candidate.folds).items():
            metric = _metric_log(f"cv.std.{name}", value)
            if metric is not None:
                metrics.append(metric)
        for fold in candidate.folds:
            metrics.extend(_metric_logs("cv", fold.metrics, step=fold.fold_index))
            metric = _metric_log("cv.time", fold.time, step=fold.fold_index)
            if metric is not None:
                metrics.append(metric)

        observations.append(
            RunObservation(
                run_name=f"{run_record.model_instance_id}/{candidate_label}",
                tags=_cv_candidate_tags(
                    pipeline_config,
                    run_record.model_instance_id,
                    model_config,
                    run_record.training_result.task_type,
                    candidate_label,
                    candidate.candidate_index,
                    ranks[position],
                    tuning_result.method,
                ),
                params={
                    "cv.method": _param_value(tuning_result.method),
                    "cv.candidate": candidate_label,
                    "cv.candidate_index": _param_value(candidate.candidate_index),
                    **{f"cv.params.{key}": _param_value(value) for key, value in candidate.model_params.items()},
                    **_classification_metric_params("cv.mean", candidate.mean_metrics),
                },
                metrics=tuple(metrics),
            )
        )

    return tuple(observations)


def _pipeline_tags(pipeline_config: PipelineConfig) -> dict[str, str]:
    return {
        TAG_RUN_TYPE: RUN_TYPE_PIPELINE,
        TAG_TRACKING_SCHEMA_VERSION: TRACKING_SCHEMA_VERSION,
        TAG_PIPELINE_ID: pipeline_config.run_id,
        "run_id": pipeline_config.run_id,
        TAG_TARGET: pipeline_config.dataset.target,
        TAG_TASK_TYPE: dataset_task_for_target(pipeline_config.dataset.target).task_type,
        TAG_TRAINED_ON: _trained_on(pipeline_config),
        TAG_TRAIN_SOURCES: _train_sources(pipeline_config),
        "trained_models": ",".join(model_params.name for model_params in pipeline_config.training),
    }


def _model_tags(
    pipeline_config: PipelineConfig,
    run_record: ModelRunRecord,
    model_config: ModelConfig,
) -> dict[str, str]:
    training_result = run_record.training_result
    tags = {
        TAG_PIPELINE_ID: pipeline_config.run_id,
        TAG_RUN_TYPE: RUN_TYPE_MODEL,
        TAG_TRACKING_SCHEMA_VERSION: TRACKING_SCHEMA_VERSION,
        TAG_MODEL_INSTANCE: run_record.model_instance_id,
        TAG_MODEL_NAME: model_config.name,
        TAG_TASK_TYPE: training_result.task_type,
        TAG_STATUS: STATUS_SUCCESS if training_result.succeeded else STATUS_FAILED,
        TAG_TRAINED_ON: _trained_on(pipeline_config),
        TAG_TRAIN_SOURCES: _train_sources(pipeline_config),
    }
    if training_result.failure_stage is not None:
        tags["failure_stage"] = training_result.failure_stage
    if training_result.error is not None:
        tags["error"] = training_result.error
    if training_result.tuning_result is not None:
        tags["tuning_method"] = training_result.tuning_result.method
    return tags


def _cv_candidate_tags(
    pipeline_config: PipelineConfig,
    model_id: str,
    model_config: ModelConfig,
    task_type: TaskType,
    candidate_label: str,
    candidate_index: int,
    candidate_rank: int,
    tuning_method: str,
) -> dict[str, str]:
    return {
        TAG_PIPELINE_ID: pipeline_config.run_id,
        TAG_RUN_TYPE: RUN_TYPE_CV_CANDIDATE,
        TAG_TRACKING_SCHEMA_VERSION: TRACKING_SCHEMA_VERSION,
        TAG_MODEL_INSTANCE: model_id,
        TAG_MODEL_NAME: model_config.name,
        "candidate": candidate_label,
        "candidate_index": str(candidate_index),
        "candidate_rank": str(candidate_rank),
        "tuning_method": tuning_method,
        TAG_TASK_TYPE: task_type,
        TAG_TRAINED_ON: _trained_on(pipeline_config),
        TAG_TRAIN_SOURCES: _train_sources(pipeline_config),
    }


def _pipeline_params(pipeline_config: PipelineConfig) -> dict[str, str]:
    dataset_task = dataset_task_for_target(pipeline_config.dataset.target)
    run_params: dict[str, object] = {
        "run_id": pipeline_config.run_id,
        "mlflow.experiment_name": pipeline_config.mlflow.experiment_name,
        PARAM_DATASET_TARGET: pipeline_config.dataset.target,
        "dataset.random_state": pipeline_config.dataset.random_state,
        "dataset.train_size": pipeline_config.dataset.train_size,
        "dataset.task_type": dataset_task.task_type,
        "dataset.kind": dataset_task.dataset_kind,
        "dataset.trained_on": _trained_on(pipeline_config),
        "training.model_names": ",".join(model.name for model in pipeline_config.training),
    }

    for index, split in enumerate(pipeline_config.dataset.train_on):
        run_params[f"dataset.train_on.{index}.dataset"] = split.dataset
        run_params[f"dataset.train_on.{index}.fraction"] = split.fraction

    return _string_params(run_params)


def _dataset_summary_params(pipeline_result: PipelineRunRecord) -> dict[str, str]:
    dataset_summary = pipeline_result.dataset_summary
    run_params: dict[str, object] = {}
    parts = {
        "train": dataset_summary.train,
        "test.mimic": dataset_summary.test_mimic,
        "test.tudd": dataset_summary.test_tudd,
    }
    for name, summary in parts.items():
        run_params[dataset_row_count_param(name)] = summary.row_count
        target_summary = summary.target_summary
        if isinstance(target_summary, ClassificationTargetSummary):
            for label, count in target_summary.class_balance.items():
                run_params[f"dataset.{name}.class_balance.{label}"] = count
        elif isinstance(target_summary, RegressionTargetSummary):
            prefix = f"dataset.{name}.target"
            run_params[f"{prefix}.count"] = target_summary.count
            run_params[f"{prefix}.mean"] = target_summary.mean
            run_params[f"{prefix}.std"] = target_summary.std
            run_params[f"{prefix}.min"] = target_summary.min
            run_params[f"{prefix}.max"] = target_summary.max

    for data_file in dataset_summary.data_files:
        prefix = f"dataset.file.{data_file.data_origin}"
        run_params[f"{prefix}.name"] = data_file.file_name
        run_params[f"{prefix}.sha256"] = data_file.sha256 or "missing"

    return _string_params(run_params)


def _pipeline_model_config_params(pipeline_config: PipelineConfig) -> dict[str, str]:
    run_params = {}
    model_ids = model_instance_ids(pipeline_config.training)
    for model_id, model_params in zip(model_ids, pipeline_config.training, strict=True):
        run_params.update(_model_config_params(model_id, model_params))
    return run_params


def _model_run_params(
    model_config: ModelConfig,
    training_result: ModelTrainingResult,
) -> dict[str, str]:
    run_params = {
        PARAM_MODEL_TUNED: _param_value(training_result.tuned),
        "model.task_type": training_result.task_type,
        **_model_config_params("config", model_config),
    }

    tuning_result = training_result.tuning_result
    if tuning_result is None:
        return run_params

    run_params.update(
        {
            "model.tuning.method": _param_value(tuning_result.method),
            "model.tuning.scoring": _param_value(tuning_result.scoring),
            PARAM_MODEL_BEST_PARAMS: _param_value(tuning_result.best_params),
        }
    )
    for key, value in tuning_result.best_params.items():
        run_params[f"model.best_params.{key}"] = _param_value(value)

    final_metrics = tuning_result.final_test_metrics.mimic_test
    run_params["model.final_evaluation.method"] = "bootstrap"
    run_params["model.final_evaluation.n_bootstrap"] = _param_value(final_metrics.n_bootstrap)
    return run_params


def _model_config_params(model_id: str, model_config: ModelConfig) -> dict[str, str]:
    prefix = f"model.{model_id}"
    run_params: dict[str, object] = {
        f"{prefix}.name": model_config.name,
    }

    if model_config.preprocessing is None:
        run_params[f"{prefix}.preprocessing.override"] = False
    else:
        run_params[f"{prefix}.preprocessing.override"] = True
        if model_config.preprocessing.imputer is not None:
            run_params[f"{prefix}.preprocessing.imputer"] = model_config.preprocessing.imputer.model_dump(mode="json")
        if model_config.preprocessing.scaler_encoder is not None:
            run_params[f"{prefix}.preprocessing.scaler_encoder"] = model_config.preprocessing.scaler_encoder.model_dump(
                mode="json"
            )

    run_params[f"{prefix}.tuning.enabled"] = True
    run_params[f"{prefix}.tuning.method"] = model_config.tuning.method
    run_params[f"{prefix}.tuning.scoring"] = model_config.tuning.scoring
    run_params[f"{prefix}.tuning.search_space"] = model_config.tuning.search_space
    run_params[f"{prefix}.tuning.cv.n_splits"] = model_config.tuning.cv.n_splits
    run_params[f"{prefix}.tuning.cv.shuffle"] = model_config.tuning.cv.shuffle
    run_params[f"{prefix}.tuning.cv.random_state"] = model_config.tuning.cv.random_state
    if model_config.tuning.grid is not None:
        run_params[f"{prefix}.tuning.grid"] = model_config.tuning.grid
    if model_config.tuning.method == "optuna":
        run_params[f"{prefix}.tuning.optuna.n_trials"] = model_config.tuning.optuna.n_trials
        run_params[f"{prefix}.tuning.optuna.sampler"] = model_config.tuning.optuna.sampler
        run_params[f"{prefix}.tuning.optuna.n_startup_trials"] = model_config.tuning.optuna.n_startup_trials
        run_params[f"{prefix}.tuning.optuna.timeout"] = model_config.tuning.optuna.timeout

    return _string_params(run_params)


def _model_metric_logs(
    training_result: ModelTrainingResult,
    model_result: ModelEvaluationRecord,
) -> tuple[MetricLog, ...]:
    metrics = [
        *_metric_logs_from_values(
            (
                (METRIC_TRAIN_FIT_TIME, training_result.fit_time),
                (METRIC_MODEL_TOTAL_TIME, model_result.total_time),
            )
        )
    ]
    tuning_result = training_result.tuning_result
    if tuning_result is not None:
        metrics.extend(_metric_logs_from_values(((METRIC_CV_TOTAL_TIME, tuning_result.total_time),)))
        metrics.extend(_bootstrap_final_test_metric_logs(tuning_result.final_test_metrics))

    for test_result in model_result.test_results:
        dataset_name = test_result.dataset_name
        metrics.extend(_metric_logs_from_values(((test_predict_time_metric(dataset_name), test_result.predict_time),)))
        if tuning_result is None:
            metrics.extend(_test_metric_logs(dataset_name, test_result.metrics))

    metrics.extend(_metric_delta_logs(model_result.final_test_metrics.mimic_minus_tudd))
    return tuple(metrics)


def _model_metric_params(
    model_result: ModelEvaluationRecord | None,
) -> dict[str, str]:
    if model_result is None:
        return {}

    params = {}
    for test_result in model_result.test_results:
        if isinstance(test_result.metrics, ClassificationMetrics):
            params[test_n_classes_param(test_result.dataset_name)] = _param_value(test_result.metrics.n_classes)
    return params


def _metric_logs(
    prefix: str,
    metrics: ClassificationMetrics | RegressionMetrics,
    *,
    step: int | None = None,
) -> tuple[MetricLog, ...]:
    return _metric_logs_from_values(
        ((f"{prefix}.{name}", value) for name, value in metrics.scores.items()),
        step=step,
    )


def _test_metric_logs(
    dataset: str,
    metrics: ClassificationMetrics | RegressionMetrics,
) -> tuple[MetricLog, ...]:
    return _metric_logs_from_values(
        ((test_score_metric(dataset, name), value) for name, value in metrics.scores.items())
    )


def _bootstrap_final_test_metric_logs(
    metrics: BootstrapFinalTestMetrics,
) -> tuple[MetricLog, ...]:
    return (
        *_bootstrap_metric_logs("mimic", metrics.mimic_test),
        *_bootstrap_metric_logs("tudd", metrics.tudd_test),
    )


def _bootstrap_metric_logs(
    dataset: str,
    metrics: BootstrapClassificationMetrics | BootstrapRegressionMetrics,
) -> tuple[MetricLog, ...]:
    values = [(test_score_metric(dataset, name), value) for name, value in metrics.scores.items()]
    for name, (lower, upper) in metrics.confidence_intervals.items():
        values.extend(
            (
                (test_score_ci_metric(dataset, name, "lower"), lower),
                (test_score_ci_metric(dataset, name, "upper"), upper),
            )
        )
    return _metric_logs_from_values(values)


def _metric_delta_logs(
    deltas: ClassificationMetrics | RegressionMetrics,
) -> tuple[MetricLog, ...]:
    return _metric_logs_from_values(((test_delta_metric(name), value) for name, value in deltas.scores.items()))


def _metric_logs_from_values(
    values: Iterable[tuple[str, float | int | None]],
    *,
    step: int | None = None,
) -> tuple[MetricLog, ...]:
    logs = []
    for name, value in values:
        metric = _metric_log(name, value, step=step)
        if metric is not None:
            logs.append(metric)
    return tuple(logs)


def _metric_log(
    name: str,
    value: float | int | None,
    *,
    step: int | None = None,
) -> MetricLog | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return MetricLog(name=name, value=value, step=step)


def _classification_metric_params(
    prefix: str,
    metrics: ClassificationMetrics | RegressionMetrics,
) -> dict[str, str]:
    if not isinstance(metrics, ClassificationMetrics):
        return {}
    return {f"{prefix}.n_classes": _param_value(metrics.n_classes)}


def _evaluation_bundle(
    pipeline_config: PipelineConfig,
    model_rows: tuple[ModelRunRecord, ...],
) -> _EvaluationBundle:
    evaluations = []
    table_rows = []
    for model_run in model_rows:
        model_id = model_run.model_instance_id
        model_result = model_run.evaluation
        if model_result is None:
            continue
        for test_result in model_result.test_results:
            metrics = {
                **test_result.metrics.scores,
                "predict_time": test_result.predict_time,
            }
            evaluation = _make_evaluation(
                pipeline_config,
                model_id,
                model_result.model_name,
                test_result.dataset_name,
                "test",
                metrics,
            )
            evaluations.append(evaluation)
            table_rows.extend(_evaluation_metric_rows(evaluation))

        delta_metrics = model_result.final_test_metrics.mimic_minus_tudd.scores
        evaluation = _make_evaluation(
            pipeline_config,
            model_id,
            model_result.model_name,
            "mimic_minus_tudd",
            "test_delta",
            delta_metrics,
        )
        evaluations.append(evaluation)
        table_rows.extend(_evaluation_metric_rows(evaluation))

    return _EvaluationBundle(tuple(evaluations), tuple(table_rows))


def _make_evaluation(
    pipeline_config: PipelineConfig,
    model_id: str,
    model_name: str,
    dataset_name: str,
    scope: str,
    metrics: dict[str, float | int],
) -> EvaluationLog:
    return EvaluationLog(
        inputs={
            TAG_MODEL_NAME: model_name,
            TAG_MODEL_INSTANCE: model_id,
            "dataset": dataset_name,
        },
        outputs={"scope": scope},
        targets={TAG_TARGET: pipeline_config.dataset.target},
        metrics=metrics,
        tags={
            TAG_PIPELINE_ID: pipeline_config.run_id,
            TAG_MODEL_NAME: model_name,
            TAG_MODEL_INSTANCE: model_id,
            "dataset": dataset_name,
            "scope": scope,
            TAG_TRAINED_ON: _trained_on(pipeline_config),
        },
    )


def _evaluation_metric_rows(
    evaluation: EvaluationLog,
) -> list[EvaluationTableRow]:
    rows = []
    for metric_name, metric_value in evaluation.metrics.items():
        value = float(metric_value)
        if not math.isfinite(value):
            continue
        rows.append(
            EvaluationTableRow(
                tracking_schema_version=TRACKING_SCHEMA_VERSION,
                pipeline_run_id=evaluation.tags[TAG_PIPELINE_ID],
                target=evaluation.targets[TAG_TARGET],
                trained_on=evaluation.tags[TAG_TRAINED_ON],
                model_name=evaluation.tags[TAG_MODEL_NAME],
                model_instance=evaluation.tags[TAG_MODEL_INSTANCE],
                dataset=evaluation.tags["dataset"],
                scope=cast(Literal["test", "test_delta"], evaluation.tags["scope"]),
                metric=metric_name,
                value=value,
            )
        )
    return rows


def _model_params_by_instance_id(
    models: tuple[ModelConfig, ...],
) -> dict[str, ModelConfig]:
    return dict(zip(model_instance_ids(models), models, strict=True))


def _run_name(pipeline_config: PipelineConfig) -> str:
    return pipeline_config.mlflow.run_name or pipeline_config.run_id


def _train_sources(pipeline_config: PipelineConfig) -> str:
    return ",".join(split.dataset for split in pipeline_config.dataset.train_on)


def _trained_on(pipeline_config: PipelineConfig) -> str:
    origins = {split.dataset for split in pipeline_config.dataset.train_on}
    if len(origins) == 1:
        return next(iter(origins))

    return "combination"


def _candidate_ranks(scores: list[float], *, maximize: bool = True) -> list[int]:
    ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=maximize)
    ranks = [0] * len(scores)
    for rank, index in enumerate(ranked_indices, start=1):
        ranks[index] = rank
    return ranks


def _candidate_summaries(
    tuning_result: TuningRecord,
    task_type: TaskType,
) -> tuple[_CandidateSummary, ...]:
    folds_by_candidate: defaultdict[int, list[FoldRecord]] = defaultdict(list)
    for fold in tuning_result.fold_results:
        folds_by_candidate[fold.candidate_index].append(fold)

    summaries = []
    for candidate_index in sorted(folds_by_candidate):
        folds = tuple(
            sorted(
                folds_by_candidate[candidate_index],
                key=lambda fold: fold.fold_index,
            )
        )
        if task_type == "classification":
            classification_metrics = [fold.metrics for fold in folds if isinstance(fold.metrics, ClassificationMetrics)]
            if len(classification_metrics) != len(folds):
                raise ValueError("Classification tuning record contains regression fold metrics")
            scores = [classification_score(metric, tuning_result.scoring) for metric in classification_metrics]
            mean_metrics = mean_classification_metrics(classification_metrics)
        else:
            regression_metrics = [fold.metrics for fold in folds if isinstance(fold.metrics, RegressionMetrics)]
            if len(regression_metrics) != len(folds):
                raise ValueError("Regression tuning record contains classification fold metrics")
            scores = [regression_score(metric, tuning_result.scoring) for metric in regression_metrics]
            mean_metrics = mean_regression_metrics(regression_metrics)

        summaries.append(
            _CandidateSummary(
                candidate_index=candidate_index,
                model_params=folds[0].model_params,
                folds=folds,
                mean_score=float(sum(scores) / len(scores)),
                std_score=float(pstdev(scores)),
                mean_metrics=mean_metrics,
            )
        )

    return tuple(summaries)


def _metric_stds(folds: Iterable[FoldRecord]) -> dict[str, float]:
    values_by_metric: defaultdict[str, list[float]] = defaultdict(list)
    for fold in folds:
        for name, value in fold.metrics.scores.items():
            values_by_metric[name].append(float(value))

    return {name: float(pstdev(values)) for name, values in values_by_metric.items() if values}


def _string_params(values: dict[str, object]) -> dict[str, str]:
    return {key: _param_value(value) for key, value in values.items()}


def _param_value(value: object) -> str:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("MLflow parameter values must be finite")
    if isinstance(value, str | int | float | bool) or value is None:
        return str(value)
    return canonical_json(value, indent=None)
