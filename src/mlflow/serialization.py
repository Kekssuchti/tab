from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.schemas.pipeline_schemas import PipelineConfig
from src.schemas.run_records import (
    ModelRunRecord,
    ModelTrainingResult,
    PipelineRunRecord,
)


def pipeline_config_to_dict(params: PipelineConfig) -> dict[str, Any]:
    return _json_safe(params.model_dump(mode="json"))


def pipeline_result_to_dict(result: PipelineRunRecord) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "total_time": _json_safe(result.total_time),
        "dataset_summary": _dataclass_to_dict(result.dataset_summary),
        "model_runs": [
            _model_run_to_dict(model_run) for model_run in result.model_runs
        ],
        "model_results": [
            _dataclass_to_dict(model_result) for model_result in result.model_results
        ],
        "training_results": [
            training_result_to_dict(training_result)
            for training_result in result.training_results
        ],
    }


def _model_run_to_dict(model_run: ModelRunRecord) -> dict[str, Any]:
    return {
        "model_instance_id": model_run.model_instance_id,
        "model_name": model_run.model_name,
        "status": "success" if model_run.succeeded else "failed",
        "training_result": training_result_to_dict(model_run.training_result),
        "evaluation": _dataclass_to_dict(model_run.evaluation),
    }


def training_result_to_dict(result: ModelTrainingResult) -> dict[str, Any]:
    return {
        "model_name": result.model_name,
        "task_type": result.task_type,
        "tuned": result.tuned,
        "fit_time": _json_safe(result.fit_time),
        "tuning_result": _dataclass_to_dict(result.tuning_result),
        "error": result.error,
        "failure_stage": result.failure_stage,
    }


def _dataclass_to_dict(value: Any) -> Any:
    if value is None:
        return None
    if not is_dataclass(value):
        return _json_safe(value)

    return {
        field.name: _dataclass_to_dict(getattr(value, field.name))
        for field in fields(value)
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(_json_safe(key)): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return _dataclass_to_dict(value)
    return str(value)
