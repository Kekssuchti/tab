from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
from src.classes.pipeline import ModelRunResult, PipelineResult
from src.evaluation.evaluation_utils import ClassificationMetrics, RegressionMetrics
from src.mlflow.serialization import (
    pipeline_params_to_dict,
    pipeline_result_to_dict,
)
from src.schemas.pipeline_schemas import PipelineParams
from src.schemas.training_schemas import ModelParams, ModelTrainingResult


class MLflowPipelineLogger:
    def log_pipeline_run(
        self,
        params: PipelineParams,
        result: PipelineResult,
        *,
        config_path: Path | None = None,
    ) -> None:
        if params.mlflow.tracking_uri:
            _allow_local_file_store(params.mlflow.tracking_uri)
            mlflow.set_tracking_uri(params.mlflow.tracking_uri)
        _set_experiment(params)

        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            artifact_paths = self._write_artifacts(params, result, temp_dir)

            with mlflow.start_run(run_name=params.run_id):
                self._log_run_tags(params, config_path)
                self._log_run_params(params)
                self._log_dataset_summary(result)
                self._log_run_metrics(result)
                self._log_artifacts(artifact_paths, config_path)

                if params.mlflow.nested_model_runs:
                    self._log_model_runs(params, result, artifact_paths["cv_dir"])

    def _log_model_runs(
        self,
        params: PipelineParams,
        result: PipelineResult,
        cv_dir: Path,
    ) -> None:
        model_ids = _model_instance_ids(params.training)
        for model_id, model_params, training_result, model_result in zip(
            model_ids,
            params.training,
            result.training_results,
            result.model_results,
            strict=False,
        ):
            with mlflow.start_run(run_name=model_id, nested=True):
                mlflow.set_tag("parent_run_id", result.run_id)
                mlflow.set_tag("model_name", model_params.name)
                mlflow.set_tag("task_type", model_params.task_type)
                self._log_model_params(
                    model_id, model_params, training_result, nested=True
                )
                self._log_model_metrics(
                    model_id, training_result, model_result, nested=True
                )

                cv_path = cv_dir / f"{model_id}.json"
                if cv_path.exists():
                    mlflow.log_artifact(str(cv_path), artifact_path="cv_results")

    def _log_run_tags(self, params: PipelineParams, config_path: Path | None) -> None:
        tags = {
            "run_id": params.run_id,
            "target": params.dataset.target,
            "task_type": "classification"
            if params.dataset.classification
            else "regression",
            "train_sources": ",".join(
                split.dataset for split in params.dataset.train_on
            ),
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
        }
        if config_path is not None:
            tags["config_path"] = str(config_path)

        mlflow.set_tags(
            {key: value for key, value in tags.items() if value is not None}
        )

    def _log_run_params(self, params: PipelineParams) -> None:
        run_params = {
            "run_id": params.run_id,
            "dataset.target": params.dataset.target,
            "dataset.random_state": params.dataset.random_state,
            "dataset.train_size": params.dataset.train_size,
            "dataset.classification": params.dataset.classification,
            "training.model_names": ",".join(
                model.name for model in params.training
            ),
            "plotting.enabled": params.plotting.enabled,
            "plotting.formats": ",".join(params.plotting.formats),
        }

        for index, split in enumerate(params.dataset.train_on):
            run_params[f"dataset.train_on.{index}.dataset"] = split.dataset
            run_params[f"dataset.train_on.{index}.fraction"] = split.fraction

        for key, value in run_params.items():
            mlflow.log_param(key, _param_value(value))

        model_ids = _model_instance_ids(params.training)
        for model_id, model_params in zip(
            model_ids, params.training, strict=False
        ):
            self._log_model_config_params(model_id, model_params)

    def _log_dataset_summary(self, result: PipelineResult) -> None:
        dataset_summary = result.dataset_summary
        parts = {
            "train": dataset_summary.train,
            "test.mimic": dataset_summary.test_mimic,
            "test.tudd": dataset_summary.test_tudd,
        }
        for name, summary in parts.items():
            mlflow.log_param(f"dataset.{name}.row_count", summary.row_count)
            for label, count in summary.class_balance.items():
                mlflow.log_param(f"dataset.{name}.class_balance.{label}", count)

        for data_file in dataset_summary.data_files:
            prefix = f"dataset.file.{data_file.data_origin}"
            mlflow.log_param(f"{prefix}.name", data_file.file_name)
            mlflow.log_param(f"{prefix}.sha256", data_file.sha256 or "missing")

    def _log_run_metrics(self, result: PipelineResult) -> None:
        self._log_metric("pipeline.total_time", result.total_time)

        for model_id, training_result, model_result in zip(
            _result_model_instance_ids(result.training_results),
            result.training_results,
            result.model_results,
            strict=False,
        ):
            self._log_model_params(model_id, None, training_result, nested=False)
            self._log_model_metrics(
                model_id, training_result, model_result, nested=False
            )

    def _log_model_config_params(
        self, model_id: str, model_params: ModelParams
    ) -> None:
        prefix = f"model.{model_id}"
        mlflow.log_param(f"{prefix}.name", model_params.name)
        mlflow.log_param(f"{prefix}.task_type", model_params.task_type)
        for key, value in model_params.params.items():
            mlflow.log_param(f"{prefix}.params.{key}", _param_value(value))

        if model_params.preprocessing is None:
            mlflow.log_param(f"{prefix}.preprocessing.override", False)
        else:
            mlflow.log_param(f"{prefix}.preprocessing.override", True)
            if model_params.preprocessing.imputer is not None:
                mlflow.log_param(
                    f"{prefix}.preprocessing.imputer",
                    _param_value(
                        model_params.preprocessing.imputer.model_dump(mode="json")
                    ),
                )
            if model_params.preprocessing.scaler_encoder is not None:
                mlflow.log_param(
                    f"{prefix}.preprocessing.scaler_encoder",
                    _param_value(
                        model_params.preprocessing.scaler_encoder.model_dump(
                            mode="json"
                        )
                    ),
                )

        if model_params.tuning is None:
            mlflow.log_param(f"{prefix}.tuning.enabled", False)
            return

        mlflow.log_param(f"{prefix}.tuning.enabled", True)
        mlflow.log_param(f"{prefix}.tuning.scoring", model_params.tuning.scoring)
        mlflow.log_param(
            f"{prefix}.tuning.search_space", model_params.tuning.search_space
        )
        mlflow.log_param(
            f"{prefix}.tuning.cv.n_splits", model_params.tuning.cv.n_splits
        )
        mlflow.log_param(f"{prefix}.tuning.cv.shuffle", model_params.tuning.cv.shuffle)
        mlflow.log_param(
            f"{prefix}.tuning.cv.random_state", model_params.tuning.cv.random_state
        )
        if model_params.tuning.grid is not None:
            mlflow.log_param(
                f"{prefix}.tuning.grid", _param_value(model_params.tuning.grid)
            )

    def _log_model_params(
        self,
        model_id: str,
        model_params: ModelParams | None,
        training_result: ModelTrainingResult,
        *,
        nested: bool,
    ) -> None:
        prefix = "model" if nested else f"model.{model_id}"
        mlflow.log_param(f"{prefix}.tuned", training_result.tuned)
        if model_params is not None:
            self._log_model_config_params("config", model_params)

        if training_result.tuning_result is None:
            return

        tuning_result = training_result.tuning_result
        mlflow.log_param(f"{prefix}.tuning.scoring", tuning_result.scoring)
        mlflow.log_param(
            f"{prefix}.tuning.best_params", _param_value(tuning_result.best_params)
        )
        for key, value in tuning_result.best_params.items():
            mlflow.log_param(f"{prefix}.best_params.{key}", _param_value(value))

    def _log_model_metrics(
        self,
        model_id: str,
        training_result: ModelTrainingResult,
        model_result: ModelRunResult,
        *,
        nested: bool,
    ) -> None:
        prefix = "" if nested else f"{model_id}."
        self._log_metric(f"{prefix}train.fit_time", training_result.fit_time)
        self._log_metric(f"{prefix}model.total_time", model_result.total_time)

        if training_result.training_metrics is not None:
            self._log_metrics(
                f"{prefix}train", training_result.training_metrics, nested=nested
            )

        if training_result.tuning_result is not None:
            tuning_result = training_result.tuning_result
            self._log_metric(f"{prefix}cv.total_time", tuning_result.total_time)
            self._log_metric(f"{prefix}cv.best_score", tuning_result.best_score)
            self._log_metrics(
                f"{prefix}cv.best", tuning_result.best_metrics, nested=nested
            )
            for index, score in enumerate(tuning_result.cv_results.mean_scores):
                self._log_metric(f"{prefix}cv.candidate_{index}.mean.primary", score)

        for test_result in model_result.test_results:
            dataset_name = test_result.dataset_name
            self._log_metric(
                f"{prefix}test.{dataset_name}.predict_time",
                test_result.predict_time,
            )
            self._log_metrics(
                f"{prefix}test.{dataset_name}", test_result.metrics, nested=nested
            )

    def _log_metrics(
        self,
        prefix: str,
        metrics: ClassificationMetrics | RegressionMetrics,
        *,
        nested: bool,
    ) -> None:
        self._log_metric(f"{prefix}.primary_score", metrics.primary_score)
        mlflow.log_param(f"{prefix}.primary_metric", metrics.primary_metric)
        for name, value in metrics.scores.items():
            self._log_metric(f"{prefix}.{name}", value)
        if isinstance(metrics, ClassificationMetrics):
            mlflow.log_param(f"{prefix}.n_classes", metrics.n_classes)

    def _log_metric(self, name: str, value: float | int | None) -> None:
        if value is None:
            return
        value = float(value)
        if math.isfinite(value):
            mlflow.log_metric(name, value)

    def _write_artifacts(
        self,
        params: PipelineParams,
        result: PipelineResult,
        temp_dir: Path,
    ) -> dict[str, Path]:
        config_path = temp_dir / "config.json"
        result_path = temp_dir / "pipeline_result.json"
        environment_path = temp_dir / "environment.json"
        cv_dir = temp_dir / "cv_results"
        cv_dir.mkdir()

        config_path.write_text(
            json.dumps(pipeline_params_to_dict(params), indent=2), encoding="utf-8"
        )
        result_path.write_text(
            json.dumps(pipeline_result_to_dict(result), indent=2), encoding="utf-8"
        )
        environment_path.write_text(
            json.dumps(_environment_info(), indent=2), encoding="utf-8"
        )

        for model_id, training_result in zip(
            _result_model_instance_ids(result.training_results),
            result.training_results,
            strict=False,
        ):
            if training_result.tuning_result is None:
                continue
            cv_path = cv_dir / f"{model_id}.json"
            cv_path.write_text(
                json.dumps(
                    pipeline_result_to_dict(
                        PipelineResult(
                            run_id=result.run_id,
                            dataset_summary=result.dataset_summary,
                            model_results=(),
                            training_results=(training_result,),
                            total_time=result.total_time,
                        )
                    )["training_results"][0]["tuning_result"],
                    indent=2,
                ),
                encoding="utf-8",
            )

        return {
            "config": config_path,
            "result": result_path,
            "environment": environment_path,
            "cv_dir": cv_dir,
        }

    def _log_artifacts(
        self,
        artifact_paths: dict[str, Path],
        config_path: Path | None,
    ) -> None:
        mlflow.log_artifact(str(artifact_paths["config"]))
        mlflow.log_artifact(str(artifact_paths["result"]))
        mlflow.log_artifact(str(artifact_paths["environment"]))

        if any(artifact_paths["cv_dir"].iterdir()):
            mlflow.log_artifacts(
                str(artifact_paths["cv_dir"]), artifact_path="cv_results"
            )

        if config_path is not None and config_path.exists():
            mlflow.log_artifact(str(config_path), artifact_path="config_source")

        uv_lock = Path("uv.lock")
        if uv_lock.exists():
            mlflow.log_artifact(str(uv_lock), artifact_path="environment")


def _model_instance_ids(models: tuple[ModelParams, ...]) -> list[str]:
    counts = Counter(model.name for model in models)
    seen: defaultdict[str, int] = defaultdict(int)
    model_ids = []
    for model in models:
        index = seen[model.name]
        seen[model.name] += 1
        model_ids.append(
            model.name if counts[model.name] == 1 else f"{model.name}__{index}"
        )
    return model_ids


def _result_model_instance_ids(results: tuple[ModelTrainingResult, ...]) -> list[str]:
    counts = Counter(result.model_name for result in results)
    seen: defaultdict[str, int] = defaultdict(int)
    model_ids = []
    for result in results:
        index = seen[result.model_name]
        seen[result.model_name] += 1
        model_ids.append(
            result.model_name
            if counts[result.model_name] == 1
            else f"{result.model_name}__{index}"
        )
    return model_ids


def _param_value(value: Any) -> str:
    if isinstance(value, str | int | float | bool) or value is None:
        text = str(value)
    else:
        text = json.dumps(value, sort_keys=True, default=str)
    return text[:5000]


def _allow_local_file_store(tracking_uri: str) -> None:
    if "://" not in tracking_uri or tracking_uri.startswith("file://"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


def _set_experiment(params: PipelineParams) -> None:
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(params.mlflow.experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            params.mlflow.experiment_name,
            artifact_location=params.mlflow.artifact_location,
        )
        mlflow.set_experiment(experiment_id=experiment_id)
        return

    mlflow.set_experiment(experiment_name=params.mlflow.experiment_name)


def _git_commit() -> str | None:
    return _git_command("rev-parse", "HEAD")


def _git_dirty() -> str | None:
    status = _git_command("status", "--porcelain")
    if status is None:
        return None
    return str(bool(status)).lower()


def _git_command(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _environment_info() -> dict[str, Any]:
    packages = {}
    for package in ("mlflow", "numpy", "pandas", "scikit-learn", "xgboost"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None

    return {
        "python": sys.version,
        "packages": packages,
    }
