from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import pstdev
from typing import Any

from src.classes.pipeline import ModelRunRecord, ModelRunResult, PipelineResult
from src.schemas.pipeline_schemas import PipelineParams
from src.schemas.training_schemas import FoldResult, ModelParams, ModelTrainingResult
from src.utils.evaluation_utils import (
    ClassificationMetricDeltas,
    ClassificationMetrics,
    CVClassificationMetrics,
    CVFinalTestMetrics,
    RegressionMetrics,
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
    params: dict[str, Any]
    folds: tuple[FoldResult, ...]
    mean_score: float
    std_score: float
    mean_metrics: ClassificationMetrics | RegressionMetrics


def assemble_pipeline_observation(
    params: PipelineParams,
    result: PipelineResult,
) -> RunObservation:
    model_params_by_id = _model_params_by_instance_id(params.training)
    evaluation_bundle = _evaluation_bundle(params, result.model_runs)

    return RunObservation(
        run_name=_run_name(params),
        tags=_pipeline_tags(params),
        params={
            **_pipeline_params(params),
            **_dataset_summary_params(result),
            **_pipeline_model_config_params(params),
        },
        metrics=_metric_logs_from_values((("pipeline.total_time", result.total_time),)),
        evaluations=evaluation_bundle.evaluations,
        table_rows=evaluation_bundle.table_rows,
        children=tuple(
            _model_run_observation(
                params,
                run_record,
                model_params_by_id[run_record.model_instance_id],
            )
            for run_record in result.model_runs
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
    params: PipelineParams,
    run_record: ModelRunRecord,
    model_params: ModelParams,
) -> RunObservation:
    training_result = run_record.training_result
    model_result = run_record.model_result
    evaluation_bundle = _evaluation_bundle(params, (run_record,))

    run_params = {
        **_model_run_params(model_params, training_result),
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
            children = _cv_candidate_observations(params, run_record, model_params)
        else:
            children = ()
    else:
        metrics = _model_metric_logs(training_result, model_result)
        children = _cv_candidate_observations(params, run_record, model_params)

    return RunObservation(
        run_name=run_record.model_instance_id,
        tags=_model_tags(params, run_record, model_params),
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
    params: PipelineParams,
    run_record: ModelRunRecord,
    model_params: ModelParams,
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
                    params,
                    run_record.model_instance_id,
                    model_params,
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
                        for key, value in candidate.params.items()
                    },
                    **_classification_metric_params("cv.mean", candidate.mean_metrics),
                },
                metrics=tuple(metrics),
            )
        )

    return tuple(observations)


def _pipeline_tags(params: PipelineParams) -> dict[str, str]:
    return _drop_none(
        {
            "run_type": "pipeline",
            "pipeline_id": params.run_id,
            "run_id": params.run_id,
            "target": params.dataset.target,
            "task_type": "classification"
            if params.dataset.classification
            else "regression",
            "trained_on": _trained_on(params),
            "train_sources": _train_sources(params),
            "trained_models": ",".join(
                model_params.name for model_params in params.training
            ),
        }
    )


def _model_tags(
    params: PipelineParams,
    run_record: ModelRunRecord,
    model_params: ModelParams,
) -> dict[str, str]:
    training_result = run_record.training_result
    tags = {
        "pipeline_id": params.run_id,
        "run_type": "model",
        "model_instance": run_record.model_instance_id,
        "model_name": model_params.name,
        "task_type": model_params.task_type,
        "status": "success" if training_result.succeeded else "failed",
        "trained_on": _trained_on(params),
        "train_sources": _train_sources(params),
    }
    if training_result.failure_stage is not None:
        tags["failure_stage"] = training_result.failure_stage
    if training_result.error is not None:
        tags["error"] = training_result.error
    if training_result.tuning_result is not None:
        tags["tuning_method"] = training_result.tuning_result.method
    return tags


def _cv_candidate_tags(
    params: PipelineParams,
    model_id: str,
    model_params: ModelParams,
    candidate_label: str,
    candidate_index: int,
    candidate_rank: int,
    tuning_method: str,
) -> dict[str, str]:
    return {
        "pipeline_id": params.run_id,
        "run_type": "cv_candidate",
        "model_instance": model_id,
        "model_name": model_params.name,
        "candidate": candidate_label,
        "candidate_index": str(candidate_index),
        "candidate_rank": str(candidate_rank),
        "tuning_method": tuning_method,
        "task_type": model_params.task_type,
        "trained_on": _trained_on(params),
        "train_sources": _train_sources(params),
    }


def _pipeline_params(params: PipelineParams) -> dict[str, str]:
    run_params: dict[str, Any] = {
        "run_id": params.run_id,
        "mlflow.experiment_name": params.mlflow.experiment_name,
        "dataset.target": params.dataset.target,
        "dataset.random_state": params.dataset.random_state,
        "dataset.train_size": params.dataset.train_size,
        "dataset.classification": params.dataset.classification,
        "dataset.trained_on": _trained_on(params),
        "training.model_names": ",".join(model.name for model in params.training),
        "plotting.enabled": params.plotting.enabled,
        "plotting.formats": ",".join(params.plotting.formats),
    }

    for index, split in enumerate(params.dataset.train_on):
        run_params[f"dataset.train_on.{index}.dataset"] = split.dataset
        run_params[f"dataset.train_on.{index}.fraction"] = split.fraction

    return _string_params(run_params)


def _dataset_summary_params(result: PipelineResult) -> dict[str, str]:
    dataset_summary = result.dataset_summary
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


def _pipeline_model_config_params(params: PipelineParams) -> dict[str, str]:
    run_params = {}
    model_ids = model_instance_ids(params.training)
    for model_id, model_params in zip(model_ids, params.training, strict=False):
        run_params.update(_model_config_params(model_id, model_params))
    return run_params


def _model_run_params(
    model_params: ModelParams,
    training_result: ModelTrainingResult,
) -> dict[str, str]:
    run_params = {
        "model.tuned": _param_value(training_result.tuned),
        **_model_config_params("config", model_params),
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


def _model_config_params(model_id: str, model_params: ModelParams) -> dict[str, str]:
    prefix = f"model.{model_id}"
    run_params: dict[str, Any] = {
        f"{prefix}.name": model_params.name,
        f"{prefix}.task_type": model_params.task_type,
    }
    for key, value in model_params.params.items():
        run_params[f"{prefix}.params.{key}"] = value

    if model_params.preprocessing is None:
        run_params[f"{prefix}.preprocessing.override"] = False
    else:
        run_params[f"{prefix}.preprocessing.override"] = True
        if model_params.preprocessing.imputer is not None:
            run_params[f"{prefix}.preprocessing.imputer"] = (
                model_params.preprocessing.imputer.model_dump(mode="json")
            )
        if model_params.preprocessing.scaler_encoder is not None:
            run_params[f"{prefix}.preprocessing.scaler_encoder"] = (
                model_params.preprocessing.scaler_encoder.model_dump(mode="json")
            )

    if model_params.tuning is None:
        run_params[f"{prefix}.tuning.enabled"] = False
        return _string_params(run_params)

    run_params[f"{prefix}.tuning.enabled"] = True
    run_params[f"{prefix}.tuning.method"] = model_params.tuning.method
    run_params[f"{prefix}.tuning.scoring"] = model_params.tuning.scoring
    run_params[f"{prefix}.tuning.search_space"] = model_params.tuning.search_space
    run_params[f"{prefix}.tuning.cv.n_splits"] = model_params.tuning.cv.n_splits
    run_params[f"{prefix}.tuning.cv.shuffle"] = model_params.tuning.cv.shuffle
    run_params[f"{prefix}.tuning.cv.random_state"] = model_params.tuning.cv.random_state
    if model_params.tuning.grid is not None:
        run_params[f"{prefix}.tuning.grid"] = model_params.tuning.grid
    if model_params.tuning.method == "optuna":
        run_params[f"{prefix}.tuning.optuna.n_trials"] = (
            model_params.tuning.optuna.n_trials
        )
        run_params[f"{prefix}.tuning.optuna.sampler"] = (
            model_params.tuning.optuna.sampler
        )
        run_params[f"{prefix}.tuning.optuna.n_startup_trials"] = (
            model_params.tuning.optuna.n_startup_trials
        )
        run_params[f"{prefix}.tuning.optuna.timeout"] = (
            model_params.tuning.optuna.timeout
        )

    return _string_params(run_params)


def _model_metric_logs(
    training_result: ModelTrainingResult,
    model_result: ModelRunResult,
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
        metrics.extend(_cv_final_test_metric_logs("test", tuning_result.test_metrics))

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
    model_result: ModelRunResult | None,
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
    metrics: CVFinalTestMetrics,
) -> tuple[MetricLog, ...]:
    return (
        *_cv_classification_metric_logs(f"{prefix}.mimic", metrics.mimic_test),
        *_cv_classification_metric_logs(f"{prefix}.tudd", metrics.tudd_test),
    )


def _cv_classification_metric_logs(
    prefix: str,
    metrics: CVClassificationMetrics,
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
    params: PipelineParams,
    model_rows: tuple[ModelRunRecord, ...],
) -> _EvaluationBundle:
    evaluations = []
    table_rows = []
    for model_run in model_rows:
        model_id = model_run.model_instance_id
        model_result = model_run.model_result
        if model_result is None:
            continue
        for test_result in model_result.test_results:
            metrics = {
                **test_result.metrics.scores,
                "predict_time": test_result.predict_time,
            }
            evaluations.append(
                _make_evaluation(
                    params,
                    model_id,
                    model_result.model_name,
                    test_result.dataset_name,
                    "test",
                    metrics,
                )
            )
            table_rows.extend(
                _evaluation_metric_rows(
                    params,
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
                params,
                model_id,
                model_result.model_name,
                "mimic_minus_tudd",
                "test_delta",
                delta_metrics,
            )
        )
        table_rows.extend(
            _evaluation_metric_rows(
                params,
                model_id,
                model_result.model_name,
                "mimic_minus_tudd",
                "test_delta",
                delta_metrics,
            )
        )

    return _EvaluationBundle(tuple(evaluations), tuple(table_rows))


def _make_evaluation(
    params: PipelineParams,
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
        targets={"target": params.dataset.target},
        metrics=metrics,
        tags={
            "pipeline_id": params.run_id,
            "model_name": model_name,
            "model_instance": model_id,
            "dataset": dataset_name,
            "scope": scope,
            "trained_on": _trained_on(params),
        },
    )


def _evaluation_metric_rows(
    params: PipelineParams,
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
                "pipeline_run_id": params.run_id,
                "target": params.dataset.target,
                "trained_on": _trained_on(params),
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
    models: tuple[ModelParams, ...],
) -> dict[str, ModelParams]:
    return dict(zip(model_instance_ids(models), models, strict=True))


def _run_name(params: PipelineParams) -> str:
    return params.mlflow.run_name or params.run_id


def _train_sources(params: PipelineParams) -> str:
    return ",".join(split.dataset for split in params.dataset.train_on)


def _trained_on(params: PipelineParams) -> str:
    origins = {_dataset_origin(split.dataset) for split in params.dataset.train_on}
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
    folds_by_candidate: defaultdict[int, list[FoldResult]] = defaultdict(list)
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
                params=folds[0].params,
                folds=folds,
                mean_score=float(sum(scores) / len(scores)),
                std_score=float(pstdev(scores)),
                mean_metrics=mean_classification_metrics(
                    [fold.metrics for fold in folds]
                ),
            )
        )

    return tuple(summaries)


def _metric_stds(folds: list[FoldResult]) -> dict[str, float]:
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
