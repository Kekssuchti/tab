import math
from typing import get_args

from src.classes.data_registry import dataset_task_for_target
from src.schemas.base_schemas import TaskType
from src.schemas.dataset_schemas import (
    ClassificationTargetSummary,
    DatasetPartSummary,
    RegressionTargetSummary,
)
from src.schemas.metrics import (
    BootstrapClassificationMetrics,
    BootstrapRegressionMetrics,
    ClassificationMetrics,
    RegressionMetrics,
)
from src.schemas.pipeline_schemas import PipelineConfig
from src.schemas.run_records import PipelineRunRecord, TuningRecord
from src.schemas.training_schemas import ClassificationScoring, RegressionScoring, TuningMethod
from src.utils.model_identity import model_instance_ids


def validate_pipeline_projection(config: PipelineConfig, result: PipelineRunRecord) -> None:
    validate_pipeline_result(result)
    if config.run_id != result.run_id:
        raise ValueError(f"Pipeline config/result run_id mismatch: {config.run_id!r} != {result.run_id!r}")
    if config.dataset.target != result.dataset_summary.target:
        raise ValueError(
            f"Pipeline config/result target mismatch: {config.dataset.target!r} != {result.dataset_summary.target!r}"
        )

    expected = list(zip(model_instance_ids(config.training), (model.name for model in config.training), strict=True))
    actual = [(model_run.model_instance_id, model_run.model_name) for model_run in result.model_runs]
    if actual != expected[: len(actual)]:
        raise ValueError(f"Pipeline config/result model instance mapping mismatch: {expected!r} != {actual!r}")


def validate_pipeline_result(result: PipelineRunRecord) -> None:
    task_type = dataset_task_for_target(result.dataset_summary.target).task_type
    model_ids = [model_run.model_instance_id for model_run in result.model_runs]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("Pipeline result contains duplicate model instance IDs")

    for name, part in (
        ("train", result.dataset_summary.train),
        ("test_mimic", result.dataset_summary.test_mimic),
        ("test_tudd", result.dataset_summary.test_tudd),
    ):
        _validate_target_summary(part, task_type, path=f"dataset_summary.{name}")

    for model_run in result.model_runs:
        training = model_run.training_result
        path = f"model_runs[{model_run.model_instance_id!r}]"
        if training.task_type != task_type:
            raise ValueError(
                f"{path}.training_result.task_type {training.task_type!r} does not match target task {task_type!r}"
            )
        if training.tuning_result is not None:
            validate_tuning_record(training.tuning_result, task_type, path=f"{path}.training_result.tuning_result")
        if model_run.evaluation is None:
            continue
        if model_run.evaluation.model_name != training.model_name:
            raise ValueError(f"{path} has inconsistent training and evaluation model names")
        for index, test_result in enumerate(model_run.evaluation.test_results):
            _validate_point_metrics(
                test_result.metrics, task_type, path=f"{path}.evaluation.test_results[{index}].metrics"
            )
        final = model_run.evaluation.final_test_metrics
        _validate_point_metrics(final.mimic_test, task_type, path=f"{path}.evaluation.final_test_metrics.mimic_test")
        _validate_point_metrics(final.tudd_test, task_type, path=f"{path}.evaluation.final_test_metrics.tudd_test")


def validate_tuning_record(tuning: TuningRecord, task_type: TaskType, *, path: str = "tuning_result") -> None:
    validate_tuning_settings(tuning.scoring, tuning.method, task_type, path=path)
    for index, fold in enumerate(tuning.fold_results):
        _validate_point_metrics(fold.metrics, task_type, path=f"{path}.fold_results[{index}].metrics")
    final = tuning.final_test_metrics
    _validate_bootstrap_metrics(final.mimic_test, task_type, path=f"{path}.final_test_metrics.mimic_test")
    _validate_bootstrap_metrics(final.tudd_test, task_type, path=f"{path}.final_test_metrics.tudd_test")
    if final.mimic_test.n_bootstrap != final.tudd_test.n_bootstrap:
        raise ValueError(f"{path}.final_test_metrics bootstrap counts must match")


def validate_tuning_settings(scoring: str, method: str, task_type: TaskType, *, path: str) -> None:
    valid_scoring = get_args(ClassificationScoring) if task_type == "classification" else get_args(RegressionScoring)
    if scoring not in valid_scoring:
        raise ValueError(f"{path}.scoring {scoring!r} is invalid for {task_type}")
    if method not in get_args(TuningMethod):
        raise ValueError(f"{path}.method has unsupported value {method!r}")


def _validate_target_summary(part: DatasetPartSummary, task_type: TaskType, *, path: str) -> None:
    summary = part.target_summary
    if task_type == "classification":
        if not isinstance(summary, ClassificationTargetSummary):
            raise ValueError(f"{path}.target_summary must be ClassificationTargetSummary")
        if any(count < 0 for count in summary.class_balance.values()):
            raise ValueError(f"{path}.target_summary class counts must be non-negative")
        if sum(summary.class_balance.values()) != part.row_count:
            raise ValueError(f"{path}.target_summary class counts must sum to row_count")
        return

    if not isinstance(summary, RegressionTargetSummary):
        raise ValueError(f"{path}.target_summary must be RegressionTargetSummary")
    if not 0 <= summary.count <= part.row_count:
        raise ValueError(f"{path}.target_summary.count must be between zero and row_count")
    values = (summary.mean, summary.std, summary.min, summary.max)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{path}.target_summary statistics must be finite")
    if summary.std < 0:
        raise ValueError(f"{path}.target_summary.std must be non-negative")
    if summary.count and not summary.min <= summary.mean <= summary.max:
        raise ValueError(f"{path}.target_summary mean must be between min and max")


def _validate_point_metrics(metrics: object, task_type: TaskType, *, path: str) -> None:
    expected = ClassificationMetrics if task_type == "classification" else RegressionMetrics
    if not isinstance(metrics, expected):
        raise ValueError(f"{path} does not match task type {task_type!r}")


def _validate_bootstrap_metrics(metrics: object, task_type: TaskType, *, path: str) -> None:
    expected = BootstrapClassificationMetrics if task_type == "classification" else BootstrapRegressionMetrics
    if not isinstance(metrics, expected):
        raise ValueError(f"{path} does not match task type {task_type!r}")
    _validate_point_metrics(metrics.metrics, task_type, path=f"{path}.metrics")
    if metrics.n_bootstrap < 1:
        raise ValueError(f"{path}.n_bootstrap must be at least 1")
