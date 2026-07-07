from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import pstdev
from typing import Any

from src.schemas.pipeline_schemas import PipelineConfig
from src.schemas.run_records import (
    FoldRecord,
    ModelEvaluationRecord,
    ModelRunRecord,
    ModelTrainingResult,
    PipelineRunRecord,
)
from src.schemas.training_schemas import ModelConfig
from src.schemas.metrics import (
    AggregatedFinalTestMetrics,
    ClassificationMetricDeltas,
    ClassificationMetrics,
    ClassificationMetricsAggregate,
    RegressionMetrics,
)
from src.utils.evaluation_utils import (
    classification_score,
    mean_classification_metrics,
)
from src.utils.model_identity import model_instance_ids


@dataclass(frozen=True)
class MetricLog:
    name: str
    value: float
    step: int | None = None


@dataclass(frozen=True)
class EvaluationLog:
    inputs: dict[str, str]
    outputs: dict[str, str]
    targets: dict[str, str]
    metrics: dict[str, float | int]
    tags: dict[str, str]


@dataclass(frozen=True)
class RunObservation:
    run_name: str
    tags: dict[str, str]
    params: dict[str, str]
    metrics: tuple[MetricLog, ...] = ()
    evaluations: tuple[EvaluationLog, ...] = ()
    table_rows: tuple[dict[str, str | float], ...] = ()
    children: tuple[RunObservation, ...] = ()
    cv_artifact_model_id: str | None = None


@dataclass(frozen=True)
class _EvaluationBundle:
    evaluations: tuple[EvaluationLog, ...]
    table_rows: tuple[dict[str, str | float], ...]


@dataclass(frozen=True)
class _CandidateSummary:
    candidate_index: int
    model_params: dict[str, Any]
    folds: tuple[FoldRecord, ...]
    mean_score: float
    std_score: float
    mean_metrics: ClassificationMetrics | RegressionMetrics


def assemble_pipeline_observation(
    pipeline_config: PipelineConfig,
    pipeline_result: PipelineRunRecord,
) -> RunObservation:
    model_params_by_id = _model_params_by_instance_id(pipeline_config.training)
    evaluation_bundle = _evaluation_bundle(pipeline_config, pipeline_result.model_runs)

    return RunObservation(
        run_name=_run_name(pipeline_config),
        tags=_pipeline_tags(pipeline_config),
        params={
            **_pipeline_params(pipeline_config),
            **_dataset_summary_params(pipeline_result),
            **_pipeline_model_config_params(pipeline_config),
        },
        metrics=_metric_logs_from_values(
            (("pipeline.total_time", pipeline_result.total_time),)
        ),
        evaluations=evaluation_bundle.evaluations,
        table_rows=evaluation_bundle.table_rows,
        children=tuple(
            _model_run_observation(
                pipeline_config,
                run_record,
                model_params_by_id[run_record.model_instance_id],
            )
            for run_record in pipeline_result.model_runs
        ),
    )


def table_rows_to_columns(
    rows: tuple[dict[str, str | float], ...],
) -> dict[str, list[str | float]]:
    columns: dict[str, list[str | float]] = {
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
            columns[key].append(row[key])
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
        **_model_metric_params(training_result, model_result),
    }
    if model_result is None:
        metrics = list(
            _metric_logs_from_values((("train.fit_time", training_result.fit_time),))
        )
        if training_result.tuning_result is not None:
            metrics.extend(
                _metric_logs_from_values(
                    (("cv.total_time", training_result.tuning_result.total_time),)
                )
            )
            children = _cv_candidate_observations(
                pipeline_config, run_record, model_config
            )
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
        cv_artifact_model_id=run_record.model_instance_id
        if training_result.tuning_result is not None
        else None,
    )


def _cv_candidate_observations(
    pipeline_config: PipelineConfig,
    run_record: ModelRunRecord,
    model_config: ModelConfig,
) -> tuple[RunObservation, ...]:
    tuning_result = run_record.training_result.tuning_result
    if tuning_result is None:
        return ()

    candidate_summaries = _candidate_summaries(tuning_result)
    ranks = _candidate_ranks(
        [candidate.mean_score for candidate in candidate_summaries]
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
        for name, value in _metric_stds(list(candidate.folds)).items():
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
                    candidate_label,
                    candidate.candidate_index,
                    ranks[position],
                    tuning_result.method,
                ),
                params={
                    "cv.method": _param_value(tuning_result.method),
                    "cv.candidate": candidate_label,
                    "cv.candidate_index": _param_value(candidate.candidate_index),
                    **{
                        f"cv.params.{key}": _param_value(value)
                        for key, value in candidate.model_params.items()
                    },
                    **_classification_metric_params("cv.mean", candidate.mean_metrics),
                },
                metrics=tuple(metrics),
            )
        )

    return tuple(observations)


def _pipeline_tags(pipeline_config: PipelineConfig) -> dict[str, str]:
    return _drop_none(
        {
            "run_type": "pipeline",
            "pipeline_id": pipeline_config.run_id,
            "run_id": pipeline_config.run_id,
            "target": pipeline_config.dataset.target,
            "task_type": "classification"
            if pipeline_config.dataset.classification
            else "regression",
            "trained_on": _trained_on(pipeline_config),
            "train_sources": _train_sources(pipeline_config),
            "trained_models": ",".join(
                model_params.name for model_params in pipeline_config.training
            ),
        }
    )


def _model_tags(
    pipeline_config: PipelineConfig,
    run_record: ModelRunRecord,
    model_config: ModelConfig,
) -> dict[str, str]:
    training_result = run_record.training_result
    tags = {
        "pipeline_id": pipeline_config.run_id,
        "run_type": "model",
        "model_instance": run_record.model_instance_id,
        "model_name": model_config.name,
        "task_type": model_config.task_type,
        "status": "success" if training_result.succeeded else "failed",
        "trained_on": _trained_on(pipeline_config),
        "train_sources": _train_sources(pipeline_config),
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
    candidate_label: str,
    candidate_index: int,
    candidate_rank: int,
    tuning_method: str,
) -> dict[str, str]:
    return {
        "pipeline_id": pipeline_config.run_id,
        "run_type": "cv_candidate",
        "model_instance": model_id,
        "model_name": model_config.name,
        "candidate": candidate_label,
        "candidate_index": str(candidate_index),
        "candidate_rank": str(candidate_rank),
        "tuning_method": tuning_method,
        "task_type": model_config.task_type,
        "trained_on": _trained_on(pipeline_config),
        "train_sources": _train_sources(pipeline_config),
    }


def _pipeline_params(pipeline_config: PipelineConfig) -> dict[str, str]:
    run_params: dict[str, Any] = {
        "run_id": pipeline_config.run_id,
        "mlflow.experiment_name": pipeline_config.mlflow.experiment_name,
        "dataset.target": pipeline_config.dataset.target,
        "dataset.random_state": pipeline_config.dataset.random_state,
        "dataset.train_size": pipeline_config.dataset.train_size,
        "dataset.classification": pipeline_config.dataset.classification,
        "dataset.trained_on": _trained_on(pipeline_config),
        "training.model_names": ",".join(
            model.name for model in pipeline_config.training
        ),
        "plotting.enabled": pipeline_config.plotting.enabled,
        "plotting.formats": ",".join(pipeline_config.plotting.formats),
    }

    for index, split in enumerate(pipeline_config.dataset.train_on):
        run_params[f"dataset.train_on.{index}.dataset"] = split.dataset
        run_params[f"dataset.train_on.{index}.fraction"] = split.fraction

    return _string_params(run_params)


def _dataset_summary_params(pipeline_result: PipelineRunRecord) -> dict[str, str]:
    dataset_summary = pipeline_result.dataset_summary
    run_params: dict[str, Any] = {}
    parts = {
        "train": dataset_summary.train,
        "test.mimic": dataset_summary.test_mimic,
        "test.tudd": dataset_summary.test_tudd,
    }
    for name, summary in parts.items():
        run_params[f"dataset.{name}.row_count"] = summary.row_count
        for label, count in summary.class_balance.items():
            run_params[f"dataset.{name}.class_balance.{label}"] = count

    for data_file in dataset_summary.data_files:
        prefix = f"dataset.file.{data_file.data_origin}"
        run_params[f"{prefix}.name"] = data_file.file_name
        run_params[f"{prefix}.sha256"] = data_file.sha256 or "missing"

    return _string_params(run_params)


def _pipeline_model_config_params(pipeline_config: PipelineConfig) -> dict[str, str]:
    run_params = {}
    model_ids = model_instance_ids(pipeline_config.training)
    for model_id, model_params in zip(
        model_ids, pipeline_config.training, strict=False
    ):
        run_params.update(_model_config_params(model_id, model_params))
    return run_params


def _model_run_params(
    model_config: ModelConfig,
    training_result: ModelTrainingResult,
) -> dict[str, str]:
    run_params = {
        "model.tuned": _param_value(training_result.tuned),
        **_model_config_params("config", model_config),
    }

    tuning_result = training_result.tuning_result
    if tuning_result is None:
        return run_params

    run_params.update(
        {
            "model.tuning.method": _param_value(tuning_result.method),
            "model.tuning.scoring": _param_value(tuning_result.scoring),
            "model.tuning.best_params": _param_value(tuning_result.best_params),
        }
    )
    for key, value in tuning_result.best_params.items():
        run_params[f"model.best_params.{key}"] = _param_value(value)
    return run_params


def _model_config_params(model_id: str, model_config: ModelConfig) -> dict[str, str]:
    prefix = f"model.{model_id}"
    run_params: dict[str, Any] = {
        f"{prefix}.name": model_config.name,
        f"{prefix}.task_type": model_config.task_type,
    }
    for key, value in model_config.params.items():
        run_params[f"{prefix}.params.{key}"] = value

    if model_config.preprocessing is None:
        run_params[f"{prefix}.preprocessing.override"] = False
    else:
        run_params[f"{prefix}.preprocessing.override"] = True
        if model_config.preprocessing.imputer is not None:
            run_params[f"{prefix}.preprocessing.imputer"] = (
                model_config.preprocessing.imputer.model_dump(mode="json")
            )
        if model_config.preprocessing.scaler_encoder is not None:
            run_params[f"{prefix}.preprocessing.scaler_encoder"] = (
                model_config.preprocessing.scaler_encoder.model_dump(mode="json")
            )

    if model_config.tuning is None:
        run_params[f"{prefix}.tuning.enabled"] = False
        return _string_params(run_params)

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
        run_params[f"{prefix}.tuning.optuna.n_trials"] = (
            model_config.tuning.optuna.n_trials
        )
        run_params[f"{prefix}.tuning.optuna.sampler"] = (
            model_config.tuning.optuna.sampler
        )
        run_params[f"{prefix}.tuning.optuna.n_startup_trials"] = (
            model_config.tuning.optuna.n_startup_trials
        )
        run_params[f"{prefix}.tuning.optuna.timeout"] = (
            model_config.tuning.optuna.timeout
        )

    return _string_params(run_params)


def _model_metric_logs(
    training_result: ModelTrainingResult,
    model_result: ModelEvaluationRecord,
) -> tuple[MetricLog, ...]:
    metrics = [
        *_metric_logs_from_values(
            (
                ("train.fit_time", training_result.fit_time),
                ("model.total_time", model_result.total_time),
            )
        )
    ]
    tuning_result = training_result.tuning_result
    if tuning_result is not None:
        metrics.extend(
            _metric_logs_from_values((("cv.total_time", tuning_result.total_time),))
        )
        metrics.extend(
            _cv_final_test_metric_logs("test", tuning_result.final_test_metrics)
        )

    for test_result in model_result.test_results:
        dataset_name = test_result.dataset_name
        metrics.extend(
            _metric_logs_from_values(
                ((f"test.{dataset_name}.predict_time", test_result.predict_time),)
            )
        )
        if tuning_result is None:
            metrics.extend(_metric_logs(f"test.{dataset_name}", test_result.metrics))

    metrics.extend(
        _metric_delta_logs(
            "test.mimic_minus_tudd",
            model_result.final_test_metrics.mimic_minus_tudd,
        )
    )
    return tuple(metrics)


def _model_metric_params(
    training_result: ModelTrainingResult,
    model_result: ModelEvaluationRecord | None,
) -> dict[str, str]:
    if model_result is None:
        return {}

    params = {}
    for test_result in model_result.test_results:
        params.update(
            _classification_metric_params(
                f"test.{test_result.dataset_name}",
                test_result.metrics,
            )
        )
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


def _cv_final_test_metric_logs(
    prefix: str,
    metrics: AggregatedFinalTestMetrics,
) -> tuple[MetricLog, ...]:
    return (
        *_cv_classification_metric_logs(f"{prefix}.mimic", metrics.mimic_test),
        *_cv_classification_metric_logs(f"{prefix}.tudd", metrics.tudd_test),
    )


def _cv_classification_metric_logs(
    prefix: str,
    metrics: ClassificationMetricsAggregate,
) -> tuple[MetricLog, ...]:
    metric_names = (
        "mean_roc_auc",
        "mean_prc_auc",
        "mean_f1",
        "mean_accuracy",
        "mean_sensitivity",
        "mean_precision",
        "ci_95_roc_auc_lower",
        "ci_95_roc_auc_upper",
        "ci_95_prc_auc_lower",
        "ci_95_prc_auc_upper",
        "ci_95_f1_lower",
        "ci_95_f1_upper",
        "ci_95_accuracy_lower",
        "ci_95_accuracy_upper",
        "ci_95_sensitivity_lower",
        "ci_95_sensitivity_upper",
        "ci_95_precision_lower",
        "ci_95_precision_upper",
    )
    return _metric_logs_from_values(
        (f"{prefix}.{name}", getattr(metrics, name)) for name in metric_names
    )


def _metric_delta_logs(
    prefix: str,
    deltas: ClassificationMetricDeltas,
) -> tuple[MetricLog, ...]:
    return _metric_logs_from_values(
        ((f"{prefix}.{name}", value) for name, value in deltas.scores.items())
    )


def _metric_logs_from_values(
    values: Any,
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
            evaluations.append(
                _make_evaluation(
                    pipeline_config,
                    model_id,
                    model_result.model_name,
                    test_result.dataset_name,
                    "test",
                    metrics,
                )
            )
            table_rows.extend(
                _evaluation_metric_rows(
                    pipeline_config,
                    model_id,
                    model_result.model_name,
                    test_result.dataset_name,
                    "test",
                    metrics,
                )
            )

        delta_metrics = model_result.final_test_metrics.mimic_minus_tudd.scores
        evaluations.append(
            _make_evaluation(
                pipeline_config,
                model_id,
                model_result.model_name,
                "mimic_minus_tudd",
                "test_delta",
                delta_metrics,
            )
        )
        table_rows.extend(
            _evaluation_metric_rows(
                pipeline_config,
                model_id,
                model_result.model_name,
                "mimic_minus_tudd",
                "test_delta",
                delta_metrics,
            )
        )

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
            "model_name": model_name,
            "model_instance": model_id,
            "dataset": dataset_name,
        },
        outputs={"scope": scope},
        targets={"target": pipeline_config.dataset.target},
        metrics=metrics,
        tags={
            "pipeline_id": pipeline_config.run_id,
            "model_name": model_name,
            "model_instance": model_id,
            "dataset": dataset_name,
            "scope": scope,
            "trained_on": _trained_on(pipeline_config),
        },
    )


def _evaluation_metric_rows(
    pipeline_config: PipelineConfig,
    model_id: str,
    model_name: str,
    dataset_name: str,
    scope: str,
    metrics: dict[str, float | int],
) -> list[dict[str, str | float]]:
    rows = []
    for metric_name, metric_value in metrics.items():
        value = float(metric_value)
        if not math.isfinite(value):
            continue
        rows.append(
            {
                "pipeline_run_id": pipeline_config.run_id,
                "target": pipeline_config.dataset.target,
                "trained_on": _trained_on(pipeline_config),
                "model_name": model_name,
                "model_instance": model_id,
                "dataset": dataset_name,
                "scope": scope,
                "metric": metric_name,
                "value": value,
            }
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
    origins = {
        _dataset_origin(split.dataset) for split in pipeline_config.dataset.train_on
    }
    if len(origins) == 1:
        return next(iter(origins))

    return "combination"


def _dataset_origin(dataset_name: str) -> str:
    if dataset_name.startswith("mimic"):
        return "mimic"
    if dataset_name.startswith("tudd"):
        return "tudd"
    return dataset_name


def _candidate_ranks(scores: list[float]) -> list[int]:
    ranked_indices = sorted(
        range(len(scores)), key=lambda index: scores[index], reverse=True
    )
    ranks = [0] * len(scores)
    for rank, index in enumerate(ranked_indices, start=1):
        ranks[index] = rank
    return ranks


def _candidate_summaries(tuning_result: Any) -> tuple[_CandidateSummary, ...]:
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
        if not isinstance(folds[0].metrics, ClassificationMetrics):
            raise NotImplementedError(
                "Regression tuning metrics are not implemented yet"
            )

        scores = [
            classification_score(fold.metrics, tuning_result.scoring) for fold in folds
        ]
        summaries.append(
            _CandidateSummary(
                candidate_index=candidate_index,
                model_params=folds[0].model_params,
                folds=folds,
                mean_score=float(sum(scores) / len(scores)),
                std_score=float(pstdev(scores)),
                mean_metrics=mean_classification_metrics(
                    [fold.metrics for fold in folds]
                ),
            )
        )

    return tuple(summaries)


def _metric_stds(folds: list[FoldRecord]) -> dict[str, float]:
    values_by_metric: defaultdict[str, list[float]] = defaultdict(list)
    for fold in folds:
        for name, value in fold.metrics.scores.items():
            values_by_metric[name].append(float(value))

    return {
        name: float(pstdev(values))
        for name, values in values_by_metric.items()
        if values
    }


def _drop_none(values: dict[str, str | None]) -> dict[str, str]:
    return {key: value for key, value in values.items() if value is not None}


def _string_params(values: dict[str, Any]) -> dict[str, str]:
    return {key: _param_value(value) for key, value in values.items()}


def _param_value(value: Any) -> str:
    if isinstance(value, str | int | float | bool) or value is None:
        return str(value)
    return json.dumps(value, sort_keys=True, default=str)
