from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast, get_args

import numpy as np

from src.classes.data_registry import dataset_task_for_target
from src.mlflow.tracking_contract import (
    ARTIFACT_CONFIG,
    ARTIFACT_CV_RESULTS,
    ARTIFACT_ENVIRONMENT,
    ARTIFACT_EVALUATION_TABLE,
    ARTIFACT_PIPELINE_RESULT,
    TRACKING_SCHEMA_VERSION,
)
from src.schemas.base_schemas import TaskType
from src.schemas.dataset_schemas import (
    ClassificationTargetSummary,
    DatasetFileSummary,
    DatasetPartSummary,
    DatasetSummary,
    RegressionTargetSummary,
    Target,
)
from src.schemas.metrics import (
    AggregatedFinalTestMetrics,
    ClassificationMetrics,
    ClassificationMetricsAggregate,
    FinalTestMetrics,
    RegressionMetrics,
    RegressionMetricsAggregate,
)
from src.schemas.pipeline_schemas import PipelineConfig
from src.schemas.run_records import (
    FoldRecord,
    ModelEvaluationRecord,
    ModelRunRecord,
    ModelTrainingResult,
    PipelineRunRecord,
    TestSetEvaluationRecord,
    TuningRecord,
)
from src.utils.evaluation_utils import ScoringMethodCLS, ScoringMethodREG
from src.mlflow.validation import validate_pipeline_result, validate_tuning_record

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

_CLASSIFICATION_METRIC_KEYS = {field.name for field in fields(ClassificationMetrics)}
_REGRESSION_METRIC_KEYS = {field.name for field in fields(RegressionMetrics)}
_CLASSIFICATION_AGGREGATE_KEYS = {field.name for field in fields(ClassificationMetricsAggregate)}
_REGRESSION_AGGREGATE_KEYS = {field.name for field in fields(RegressionMetricsAggregate)}


@dataclass(frozen=True)
class PipelineResultEnvelope:
    tracking_schema_version: str
    pipeline_result: PipelineRunRecord

    def to_dict(self) -> JsonObject:
        validate_pipeline_result(self.pipeline_result)
        return {
            "tracking_schema_version": self.tracking_schema_version,
            "pipeline_result": _dataclass_to_object(self.pipeline_result, path="pipeline_result"),
        }


@dataclass(frozen=True)
class CVResultEnvelope:
    tracking_schema_version: str
    model_instance_id: str
    task_type: TaskType
    tuning_result: TuningRecord

    def to_dict(self) -> JsonObject:
        validate_tuning_record(self.tuning_result, self.task_type)
        return {
            "tracking_schema_version": self.tracking_schema_version,
            "model_instance_id": self.model_instance_id,
            "task_type": self.task_type,
            "tuning_result": _dataclass_to_object(self.tuning_result, path="tuning_result"),
        }


@dataclass(frozen=True)
class ArtifactManifest:
    tracking_schema_version: str
    config: str
    pipeline_result: str
    environment: str
    evaluation_table: str | None
    cv_results: tuple[str, ...]

    def to_dict(self) -> JsonObject:
        payload: JsonObject = {
            "tracking_schema_version": self.tracking_schema_version,
            "config": self.config,
            "pipeline_result": self.pipeline_result,
            "environment": self.environment,
            "cv_results": list(self.cv_results),
        }
        if self.evaluation_table is not None:
            payload["evaluation_table"] = self.evaluation_table
        return payload


def artifact_manifest(
    cv_result_names: tuple[str, ...],
    *,
    include_evaluation_table: bool,
) -> ArtifactManifest:
    return ArtifactManifest(
        tracking_schema_version=TRACKING_SCHEMA_VERSION,
        config=ARTIFACT_CONFIG,
        pipeline_result=ARTIFACT_PIPELINE_RESULT,
        environment=ARTIFACT_ENVIRONMENT,
        evaluation_table=ARTIFACT_EVALUATION_TABLE if include_evaluation_table else None,
        cv_results=tuple(f"{ARTIFACT_CV_RESULTS}/{name}" for name in cv_result_names),
    )


def pipeline_config_to_dict(params: PipelineConfig) -> JsonObject:
    return _json_object(params.model_dump(mode="json"), path="config")


def pipeline_result_to_record(result: PipelineRunRecord) -> PipelineResultEnvelope:
    validate_pipeline_result(result)
    return PipelineResultEnvelope(TRACKING_SCHEMA_VERSION, result)


def pipeline_result_to_dict(result: PipelineRunRecord) -> JsonObject:
    return pipeline_result_to_record(result).to_dict()


def pipeline_result_from_dict(value: object) -> PipelineResultEnvelope:
    payload = _object(value, path="pipeline_result_envelope")
    _exact_keys(payload, {"tracking_schema_version", "pipeline_result"}, path="pipeline_result_envelope")
    version = _string(payload["tracking_schema_version"], path="pipeline_result_envelope.tracking_schema_version")
    _require_schema_version(version)
    envelope = PipelineResultEnvelope(
        tracking_schema_version=version,
        pipeline_result=_pipeline_run_from_dict(payload["pipeline_result"], path="pipeline_result"),
    )
    validate_pipeline_result(envelope.pipeline_result)
    return envelope


def pipeline_result_from_json(value: str) -> PipelineResultEnvelope:
    return pipeline_result_from_dict(json.loads(value, parse_constant=_reject_json_constant))


def cv_result_to_record(
    model_instance_id: str,
    task_type: TaskType,
    tuning_result: TuningRecord,
) -> CVResultEnvelope:
    validate_tuning_record(tuning_result, task_type)
    return CVResultEnvelope(TRACKING_SCHEMA_VERSION, model_instance_id, task_type, tuning_result)


def cv_result_to_dict(
    model_instance_id: str,
    task_type: TaskType,
    tuning_result: TuningRecord,
) -> JsonObject:
    return cv_result_to_record(model_instance_id, task_type, tuning_result).to_dict()


def cv_result_from_dict(value: object) -> CVResultEnvelope:
    payload = _object(value, path="cv_result_envelope")
    _exact_keys(
        payload,
        {"tracking_schema_version", "model_instance_id", "task_type", "tuning_result"},
        path="cv_result_envelope",
    )
    version = _string(payload["tracking_schema_version"], path="cv_result_envelope.tracking_schema_version")
    _require_schema_version(version)
    task_type = _task_type(payload["task_type"], path="cv_result_envelope.task_type")
    tuning_result = _tuning_result_from_dict(payload["tuning_result"], task_type, path="tuning_result")
    envelope = CVResultEnvelope(
        tracking_schema_version=version,
        model_instance_id=_string(payload["model_instance_id"], path="cv_result_envelope.model_instance_id"),
        task_type=task_type,
        tuning_result=tuning_result,
    )
    validate_tuning_record(tuning_result, task_type)
    return envelope


def cv_result_from_json(value: str) -> CVResultEnvelope:
    return cv_result_from_dict(json.loads(value, parse_constant=_reject_json_constant))


def artifact_manifest_from_dict(value: object) -> ArtifactManifest:
    payload = _object(value, path="artifact_manifest")
    _required_keys(
        payload,
        {"tracking_schema_version", "config", "pipeline_result", "environment", "cv_results"},
        {"evaluation_table"},
        path="artifact_manifest",
    )
    version = _string(payload["tracking_schema_version"], path="artifact_manifest.tracking_schema_version")
    _require_schema_version(version)
    cv_results = _list(payload["cv_results"], path="artifact_manifest.cv_results")
    return ArtifactManifest(
        tracking_schema_version=version,
        config=_string(payload["config"], path="artifact_manifest.config"),
        pipeline_result=_string(payload["pipeline_result"], path="artifact_manifest.pipeline_result"),
        environment=_string(payload["environment"], path="artifact_manifest.environment"),
        evaluation_table=(
            _string(payload["evaluation_table"], path="artifact_manifest.evaluation_table")
            if "evaluation_table" in payload
            else None
        ),
        cv_results=tuple(
            _string(item, path=f"artifact_manifest.cv_results[{index}]") for index, item in enumerate(cv_results)
        ),
    )


def artifact_manifest_from_json(value: str) -> ArtifactManifest:
    return artifact_manifest_from_dict(json.loads(value, parse_constant=_reject_json_constant))


def canonical_json(value: object, *, indent: int | None = 2) -> str:
    checked = _json_value(value, path="$")
    return json.dumps(checked, allow_nan=False, indent=indent, sort_keys=True)


def _pipeline_run_from_dict(value: object, *, path: str) -> PipelineRunRecord:
    payload = _object(value, path=path)
    _exact_keys(payload, {"run_id", "dataset_summary", "model_runs", "total_time"}, path=path)
    model_runs = _list(payload["model_runs"], path=f"{path}.model_runs")
    return PipelineRunRecord(
        run_id=_string(payload["run_id"], path=f"{path}.run_id"),
        dataset_summary=_dataset_summary_from_dict(payload["dataset_summary"], path=f"{path}.dataset_summary"),
        model_runs=tuple(
            _model_run_from_dict(item, path=f"{path}.model_runs[{index}]") for index, item in enumerate(model_runs)
        ),
        total_time=_finite_float(payload["total_time"], path=f"{path}.total_time"),
    )


def _dataset_summary_from_dict(value: object, *, path: str) -> DatasetSummary:
    payload = _object(value, path=path)
    _exact_keys(payload, {"target", "train", "test_mimic", "test_tudd", "data_files"}, path=path)
    target_value = _string(payload["target"], path=f"{path}.target")
    if target_value not in get_args(Target):
        raise ValueError(f"{path}.target has unsupported value {target_value!r}")
    target = cast(Target, target_value)
    task_type = dataset_task_for_target(target).task_type
    data_files = _list(payload["data_files"], path=f"{path}.data_files")
    return DatasetSummary(
        target=target,
        train=_dataset_part_from_dict(payload["train"], task_type, path=f"{path}.train"),
        test_mimic=_dataset_part_from_dict(payload["test_mimic"], task_type, path=f"{path}.test_mimic"),
        test_tudd=_dataset_part_from_dict(payload["test_tudd"], task_type, path=f"{path}.test_tudd"),
        data_files=tuple(
            _dataset_file_from_dict(item, path=f"{path}.data_files[{index}]") for index, item in enumerate(data_files)
        ),
    )


def _dataset_part_from_dict(value: object, task_type: TaskType, *, path: str) -> DatasetPartSummary:
    payload = _object(value, path=path)
    _exact_keys(payload, {"row_count", "target_summary"}, path=path)
    target_summary = _object(payload["target_summary"], path=f"{path}.target_summary")
    if task_type == "classification":
        _exact_keys(target_summary, {"class_balance"}, path=f"{path}.target_summary")
        class_balance = _object(target_summary["class_balance"], path=f"{path}.target_summary.class_balance")
        parsed_summary = ClassificationTargetSummary(
            class_balance={
                key: _integer(count, path=f"{path}.target_summary.class_balance.{key}")
                for key, count in class_balance.items()
            }
        )
    else:
        _exact_keys(target_summary, {"count", "mean", "std", "min", "max"}, path=f"{path}.target_summary")
        parsed_summary = RegressionTargetSummary(
            count=_integer(target_summary["count"], path=f"{path}.target_summary.count"),
            mean=_finite_float(target_summary["mean"], path=f"{path}.target_summary.mean"),
            std=_finite_float(target_summary["std"], path=f"{path}.target_summary.std"),
            min=_finite_float(target_summary["min"], path=f"{path}.target_summary.min"),
            max=_finite_float(target_summary["max"], path=f"{path}.target_summary.max"),
        )
    return DatasetPartSummary(
        row_count=_integer(payload["row_count"], path=f"{path}.row_count"),
        target_summary=parsed_summary,
    )


def _dataset_file_from_dict(value: object, *, path: str) -> DatasetFileSummary:
    payload = _object(value, path=path)
    _exact_keys(payload, {"dataset_name", "data_origin", "file_name", "path", "sha256"}, path=path)
    return DatasetFileSummary(
        dataset_name=_string(payload["dataset_name"], path=f"{path}.dataset_name"),
        data_origin=_string(payload["data_origin"], path=f"{path}.data_origin"),
        file_name=_string(payload["file_name"], path=f"{path}.file_name"),
        path=_string(payload["path"], path=f"{path}.path"),
        sha256=_optional_string(payload["sha256"], path=f"{path}.sha256"),
    )


def _model_run_from_dict(value: object, *, path: str) -> ModelRunRecord:
    payload = _object(value, path=path)
    _exact_keys(payload, {"model_instance_id", "training_result", "evaluation"}, path=path)
    training_result = _training_result_from_dict(payload["training_result"], path=f"{path}.training_result")
    evaluation_value = payload["evaluation"]
    evaluation = (
        _evaluation_from_dict(evaluation_value, training_result.task_type, path=f"{path}.evaluation")
        if evaluation_value is not None
        else None
    )
    if evaluation is not None and evaluation.model_name != training_result.model_name:
        raise ValueError(f"{path} has inconsistent training and evaluation model names")
    return ModelRunRecord(
        model_instance_id=_string(payload["model_instance_id"], path=f"{path}.model_instance_id"),
        training_result=training_result,
        evaluation=evaluation,
    )


def _training_result_from_dict(value: object, *, path: str) -> ModelTrainingResult:
    payload = _object(value, path=path)
    _exact_keys(
        payload,
        {"model_name", "task_type", "tuned", "fit_time", "tuning_result", "error", "failure_stage"},
        path=path,
    )
    task_type = _task_type(payload["task_type"], path=f"{path}.task_type")
    tuning_value = payload["tuning_result"]
    return ModelTrainingResult(
        model_name=_string(payload["model_name"], path=f"{path}.model_name"),
        task_type=task_type,
        tuned=_boolean(payload["tuned"], path=f"{path}.tuned"),
        fit_time=_finite_float(payload["fit_time"], path=f"{path}.fit_time"),
        tuning_result=(
            _tuning_result_from_dict(tuning_value, task_type, path=f"{path}.tuning_result")
            if tuning_value is not None
            else None
        ),
        error=_optional_string(payload["error"], path=f"{path}.error"),
        failure_stage=_optional_string(payload["failure_stage"], path=f"{path}.failure_stage"),
    )


def _tuning_result_from_dict(value: object, task_type: TaskType, *, path: str) -> TuningRecord:
    payload = _object(value, path=path)
    _exact_keys(payload, {"best_params", "scoring", "final_test_metrics", "fold_results", "method"}, path=path)
    scoring = _string(payload["scoring"], path=f"{path}.scoring")
    valid_scoring = {"roc_auc", "f1", "accuracy"} if task_type == "classification" else {"r2", "mae", "mse", "rmse"}
    if scoring not in valid_scoring:
        raise ValueError(f"{path}.scoring {scoring!r} is invalid for {task_type}")
    method = _string(payload["method"], path=f"{path}.method")
    if method not in {"grid", "optuna"}:
        raise ValueError(f"{path}.method has unsupported value {method!r}")
    folds = _list(payload["fold_results"], path=f"{path}.fold_results")
    return TuningRecord(
        best_params=_json_object(payload["best_params"], path=f"{path}.best_params"),
        scoring=cast(ScoringMethodCLS | ScoringMethodREG, scoring),
        final_test_metrics=_aggregated_final_metrics_from_dict(
            payload["final_test_metrics"], task_type, path=f"{path}.final_test_metrics"
        ),
        fold_results=[
            _fold_from_dict(item, task_type, path=f"{path}.fold_results[{index}]") for index, item in enumerate(folds)
        ],
        method=cast(Literal["grid", "optuna"], method),
    )


def _fold_from_dict(value: object, task_type: TaskType, *, path: str) -> FoldRecord:
    payload = _object(value, path=path)
    _exact_keys(payload, {"candidate_index", "fold_index", "metrics", "time", "model_params"}, path=path)
    return FoldRecord(
        candidate_index=_integer(payload["candidate_index"], path=f"{path}.candidate_index"),
        fold_index=_integer(payload["fold_index"], path=f"{path}.fold_index"),
        metrics=_point_metrics_from_dict(payload["metrics"], task_type, path=f"{path}.metrics"),
        time=_finite_float(payload["time"], path=f"{path}.time"),
        model_params=_json_object(payload["model_params"], path=f"{path}.model_params"),
    )


def _evaluation_from_dict(value: object, task_type: TaskType, *, path: str) -> ModelEvaluationRecord:
    payload = _object(value, path=path)
    _exact_keys(payload, {"model_name", "test_results", "final_test_metrics", "fit_time"}, path=path)
    test_results = _list(payload["test_results"], path=f"{path}.test_results")
    return ModelEvaluationRecord(
        model_name=_string(payload["model_name"], path=f"{path}.model_name"),
        test_results=tuple(
            _test_evaluation_from_dict(item, task_type, path=f"{path}.test_results[{index}]")
            for index, item in enumerate(test_results)
        ),
        final_test_metrics=_final_metrics_from_dict(
            payload["final_test_metrics"], task_type, path=f"{path}.final_test_metrics"
        ),
        fit_time=_finite_float(payload["fit_time"], path=f"{path}.fit_time"),
    )


def _test_evaluation_from_dict(value: object, task_type: TaskType, *, path: str) -> TestSetEvaluationRecord:
    payload = _object(value, path=path)
    _exact_keys(payload, {"dataset_name", "metrics", "predict_time"}, path=path)
    return TestSetEvaluationRecord(
        dataset_name=_string(payload["dataset_name"], path=f"{path}.dataset_name"),
        metrics=_point_metrics_from_dict(payload["metrics"], task_type, path=f"{path}.metrics"),
        predict_time=_finite_float(payload["predict_time"], path=f"{path}.predict_time"),
    )


def _final_metrics_from_dict(value: object, task_type: TaskType, *, path: str) -> FinalTestMetrics:
    payload = _object(value, path=path)
    _exact_keys(payload, {"mimic_test", "mimic_prediction_time", "tudd_test", "tudd_prediction_time"}, path=path)
    return FinalTestMetrics(
        mimic_test=_point_metrics_from_dict(payload["mimic_test"], task_type, path=f"{path}.mimic_test"),
        mimic_prediction_time=_finite_float(payload["mimic_prediction_time"], path=f"{path}.mimic_prediction_time"),
        tudd_test=_point_metrics_from_dict(payload["tudd_test"], task_type, path=f"{path}.tudd_test"),
        tudd_prediction_time=_finite_float(payload["tudd_prediction_time"], path=f"{path}.tudd_prediction_time"),
    )


def _aggregated_final_metrics_from_dict(
    value: object,
    task_type: TaskType,
    *,
    path: str,
) -> AggregatedFinalTestMetrics:
    payload = _object(value, path=path)
    _exact_keys(payload, {"mimic_test", "mimic_prediction_time", "tudd_test", "tudd_prediction_time"}, path=path)
    return AggregatedFinalTestMetrics(
        mimic_test=_aggregate_metrics_from_dict(payload["mimic_test"], task_type, path=f"{path}.mimic_test"),
        mimic_prediction_time=_finite_float(payload["mimic_prediction_time"], path=f"{path}.mimic_prediction_time"),
        tudd_test=_aggregate_metrics_from_dict(payload["tudd_test"], task_type, path=f"{path}.tudd_test"),
        tudd_prediction_time=_finite_float(payload["tudd_prediction_time"], path=f"{path}.tudd_prediction_time"),
    )


def _point_metrics_from_dict(
    value: object,
    task_type: TaskType,
    *,
    path: str,
) -> ClassificationMetrics | RegressionMetrics:
    payload = _object(value, path=path)
    if task_type == "regression":
        _exact_keys(payload, _REGRESSION_METRIC_KEYS, path=path)
        return RegressionMetrics(
            r2=_finite_float(payload["r2"], path=f"{path}.r2"),
            mae=_finite_float(payload["mae"], path=f"{path}.mae"),
            mse=_finite_float(payload["mse"], path=f"{path}.mse"),
            rmse=_finite_float(payload["rmse"], path=f"{path}.rmse"),
        )

    _exact_keys(payload, _CLASSIFICATION_METRIC_KEYS, path=path)
    confusion_matrix = payload["confusion_matrix"]
    return ClassificationMetrics(
        roc_auc=_optional_float(payload["roc_auc"], path=f"{path}.roc_auc"),
        prc_auc=_optional_float(payload["prc_auc"], path=f"{path}.prc_auc"),
        f1=_finite_float(payload["f1"], path=f"{path}.f1"),
        accuracy=_finite_float(payload["accuracy"], path=f"{path}.accuracy"),
        sensitivity=_finite_float(payload["sensitivity"], path=f"{path}.sensitivity"),
        precision=_finite_float(payload["precision"], path=f"{path}.precision"),
        n_classes=_integer(payload["n_classes"], path=f"{path}.n_classes"),
        confusion_matrix=(
            _numeric_array(confusion_matrix, path=f"{path}.confusion_matrix") if confusion_matrix is not None else None
        ),
    )


def _aggregate_metrics_from_dict(
    value: object,
    task_type: TaskType,
    *,
    path: str,
) -> ClassificationMetricsAggregate | RegressionMetricsAggregate:
    payload = _object(value, path=path)
    if task_type == "classification":
        _exact_keys(payload, _CLASSIFICATION_AGGREGATE_KEYS, path=path)
        aggregate = ClassificationMetricsAggregate.__new__(ClassificationMetricsAggregate)
        for field in fields(ClassificationMetricsAggregate):
            field_path = f"{path}.{field.name}"
            if field.name == "mean_confusion_matrix":
                parsed = _numeric_array(payload[field.name], path=field_path)
            elif field.name == "n_classes":
                parsed = _integer(payload[field.name], path=field_path)
            else:
                parsed = _finite_float(payload[field.name], path=field_path)
            setattr(aggregate, field.name, parsed)
        return aggregate

    _exact_keys(payload, _REGRESSION_AGGREGATE_KEYS, path=path)
    aggregate = RegressionMetricsAggregate.__new__(RegressionMetricsAggregate)
    for field in fields(RegressionMetricsAggregate):
        setattr(aggregate, field.name, _finite_float(payload[field.name], path=f"{path}.{field.name}"))
    return aggregate


def _dataclass_to_object(value: object, *, path: str) -> JsonObject:
    if not is_dataclass(value) or isinstance(value, type):
        raise TypeError(f"{path} must be a dataclass instance")
    return {field.name: _json_value(getattr(value, field.name), path=f"{path}.{field.name}") for field in fields(value)}


def _json_value(value: object, *, path: str) -> JsonValue:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return _finite_float(value, path=path)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_value(value.item(), path=path)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist(), path=path)
    if isinstance(value, dict):
        result: JsonObject = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains non-string key {key!r}")
            result[key] = _json_value(item, path=f"{path}.{key}")
        return result
    if isinstance(value, list | tuple):
        return [_json_value(item, path=f"{path}[]") for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _dataclass_to_object(value, path=path)
    raise TypeError(f"Unsupported value at {path}: {type(value).__name__}")


def _json_object(value: object, *, path: str) -> JsonObject:
    converted = _json_value(value, path=path)
    if not isinstance(converted, dict):
        raise TypeError(f"{path} must be an object")
    return converted


def _object(value: object, *, path: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{path} must be an object with string keys")
    return value


def _list(value: object, *, path: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{path} must be an array")
    return value


def _string(value: object, *, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string")
    return value


def _optional_string(value: object, *, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path=path)


def _task_type(value: object, *, path: str) -> TaskType:
    task_type = _string(value, path=path)
    if task_type not in {"classification", "regression"}:
        raise ValueError(f"{path} has unsupported value {task_type!r}")
    return cast(TaskType, task_type)


def _boolean(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be a boolean")
    return value


def _integer(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer")
    return value


def _optional_float(value: object, *, path: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, path=path)


def _finite_float(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{path} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{path} must be finite")
    return converted


def _numeric_array(value: object, *, path: str) -> np.ndarray:
    items = _list(value, path=path)
    try:
        array = np.asarray(items, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{path} must be a rectangular numeric array") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{path} must contain only finite values")
    return array


def _exact_keys(value: dict[str, object], expected: set[str], *, path: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{path} has invalid keys; missing={missing}, extra={extra}")


def _required_keys(
    value: dict[str, object],
    required: set[str],
    optional: set[str],
    *,
    path: str,
) -> None:
    actual = set(value)
    missing = sorted(required - actual)
    extra = sorted(actual - required - optional)
    if missing or extra:
        raise ValueError(f"{path} has invalid keys; missing={missing}, extra={extra}")


def _require_schema_version(version: str) -> None:
    if version != TRACKING_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported MLflow tracking schema version {version!r}; expected {TRACKING_SCHEMA_VERSION!r}"
        )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid non-finite JSON value {value}")
