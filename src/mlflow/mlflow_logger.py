from __future__ import annotations

import json
import math
import sys
import warnings
from collections import Counter, defaultdict
from importlib import metadata
from pathlib import Path
from statistics import pstdev
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
from mlflow.evaluation import Evaluation, log_evaluations
from src.classes.pipeline import ModelRunResult, PipelineResult
from src.config import config
from src.mlflow.serialization import (
    pipeline_params_to_dict,
    pipeline_result_to_dict,
)
from src.schemas.pipeline_schemas import PipelineParams
from src.schemas.training_schemas import FoldResult, ModelParams, ModelTrainingResult
from src.utils.evaluation_utils import (
    ClassificationMetricDeltas,
    ClassificationMetrics,
    RegressionMetrics,
)


class MLflowPipelineLogger:
    def log_pipeline_run(
        self,
        params: PipelineParams,
        result: PipelineResult,
        *,
        config_path: Path | None = None,
    ) -> None:
        mlflow.set_tracking_uri(params.mlflow.tracking_uri)
        _set_experiment(params)

        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            artifact_paths = self._write_artifacts(params, result, temp_dir)

            with mlflow.start_run(run_name=_run_name(params)) as pipeline_run:
                self._log_run_tags(params, config_path)
                self._log_run_params(params)
                self._log_dataset_summary(result)
                self._log_run_metrics(result)
                self._log_evaluation_tables(params, result)
                self._log_artifacts(artifact_paths, config_path)

                self._log_model_runs(
                    params,
                    result,
                    artifact_paths["cv_dir"],
                    pipeline_mlflow_run_id=pipeline_run.info.run_id,
                )

    def _log_model_runs(
        self,
        params: PipelineParams,
        result: PipelineResult,
        cv_dir: Path,
        *,
        pipeline_mlflow_run_id: str,
    ) -> None:
        model_ids = _model_instance_ids(params.training)
        model_results_by_training = _model_results_by_training_result(result)
        for model_id, model_params, training_result, model_result in zip(
            model_ids,
            params.training,
            result.training_results,
            model_results_by_training,
            strict=False,
        ):
            with mlflow.start_run(run_name=model_id, nested=True) as model_run:
                mlflow.set_tag("pipeline_id", params.run_id)
                mlflow.set_tag("pipeline_mlflow_run_id", pipeline_mlflow_run_id)
                mlflow.set_tag("model_mlflow_run_id", model_run.info.run_id)
                mlflow.set_tag("run_type", "model")
                mlflow.set_tag("model_instance", model_id)
                mlflow.set_tag("model_name", model_params.name)
                mlflow.set_tag("task_type", model_params.task_type)
                mlflow.set_tag(
                    "status", "success" if training_result.succeeded else "failed"
                )
                mlflow.set_tag("trained_on", _trained_on(params))
                mlflow.set_tag("train_sources", _train_sources(params))
                self._log_model_params(model_params, training_result)
                if model_result is None:
                    self._log_model_failure(training_result)
                else:
                    self._log_model_metrics(training_result, model_result)
                    self._log_evaluation_tables(
                        params,
                        result,
                        model_filter=(model_id, training_result, model_result),
                    )
                    self._log_cv_candidate_runs(
                        params,
                        model_id,
                        model_params,
                        training_result,
                        pipeline_mlflow_run_id=pipeline_mlflow_run_id,
                        model_mlflow_run_id=model_run.info.run_id,
                    )

                cv_path = cv_dir / f"{model_id}.json"
                if cv_path.exists():
                    mlflow.log_artifact(str(cv_path), artifact_path="cv_results")

    def _log_cv_candidate_runs(
        self,
        params: PipelineParams,
        model_id: str,
        model_params: ModelParams,
        training_result: ModelTrainingResult,
        *,
        pipeline_mlflow_run_id: str,
        model_mlflow_run_id: str,
    ) -> None:
        tuning_result = training_result.tuning_result
        if tuning_result is None:
            return

        ranks = _candidate_ranks(tuning_result.cv_results.mean_scores)
        for candidate_index, candidate_params in enumerate(
            tuning_result.cv_results.params
        ):
            candidate_label = f"cv{candidate_index:02d}"
            candidate_name = f"{model_id}/{candidate_label}"
            candidate_folds = [
                fold
                for fold in tuning_result.fold_results
                if fold.candidate_index == candidate_index
            ]
            with mlflow.start_run(run_name=candidate_name, nested=True):
                mlflow.set_tag("pipeline_id", params.run_id)
                mlflow.set_tag("pipeline_mlflow_run_id", pipeline_mlflow_run_id)
                mlflow.set_tag("model_mlflow_run_id", model_mlflow_run_id)
                mlflow.set_tag("run_type", "cv_candidate")
                mlflow.set_tag("model_instance", model_id)
                mlflow.set_tag("model_name", model_params.name)
                mlflow.set_tag("candidate", candidate_label)
                mlflow.set_tag("candidate_index", str(candidate_index))
                mlflow.set_tag("candidate_rank", str(ranks[candidate_index]))
                mlflow.set_tag("task_type", model_params.task_type)
                mlflow.set_tag("trained_on", _trained_on(params))
                mlflow.set_tag("train_sources", _train_sources(params))

                mlflow.log_param("cv.candidate", candidate_label)
                mlflow.log_param("cv.candidate_index", candidate_index)
                for key, value in candidate_params.items():
                    mlflow.log_param(f"cv.params.{key}", _param_value(value))

                self._log_metric("cv.rank", ranks[candidate_index])
                self._log_metric(
                    "cv.mean_score",
                    tuning_result.cv_results.mean_scores[candidate_index],
                )
                self._log_metric(
                    "cv.std_score",
                    tuning_result.cv_results.std_scores[candidate_index],
                )
                self._log_metrics(
                    "cv.mean",
                    tuning_result.cv_results.mean_metrics[candidate_index],
                )
                for name, value in _metric_stds(candidate_folds).items():
                    self._log_metric(f"cv.std.{name}", value)

                for fold in candidate_folds:
                    self._log_metrics("cv", fold.metrics, step=fold.fold_index)
                    self._log_metric("cv.time", fold.time, step=fold.fold_index)

    def _log_run_tags(self, params: PipelineParams, config_path: Path | None) -> None:
        tags = {
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

        mlflow.set_tags(
            {key: value for key, value in tags.items() if value is not None}
        )

    def _log_run_params(self, params: PipelineParams) -> None:
        run_params = {
            "run_id": params.run_id,
            "mlflow.experiment_name": params.mlflow.experiment_name,
            "mlflow.run_name": params.mlflow.run_name,
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

        for key, value in run_params.items():
            mlflow.log_param(key, _param_value(value))

        model_ids = _model_instance_ids(params.training)
        for model_id, model_params in zip(model_ids, params.training, strict=False):
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
        model_params: ModelParams | None,
        training_result: ModelTrainingResult,
    ) -> None:
        prefix = "model"
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
        training_result: ModelTrainingResult,
        model_result: ModelRunResult,
    ) -> None:
        prefix = ""
        self._log_training_metrics(prefix, training_result, model_result)
        self._log_tuning_metrics(prefix, training_result)
        self._log_test_metrics(prefix, model_result)

    def _log_model_failure(self, training_result: ModelTrainingResult) -> None:
        self._log_metric("train.fit_time", training_result.fit_time)
        if training_result.failure_stage is not None:
            mlflow.log_param("model.failure_stage", training_result.failure_stage)
        if training_result.error is not None:
            mlflow.log_param("model.error", training_result.error)

    def _log_training_metrics(
        self,
        prefix: str,
        training_result: ModelTrainingResult,
        model_result: ModelRunResult,
    ) -> None:
        self._log_metric(f"{prefix}train.fit_time", training_result.fit_time)
        self._log_metric(f"{prefix}model.total_time", model_result.total_time)
        if training_result.training_metrics is not None:
            self._log_metrics(f"{prefix}train", training_result.training_metrics)

    def _log_tuning_metrics(
        self,
        prefix: str,
        training_result: ModelTrainingResult,
    ) -> None:
        tuning_result = training_result.tuning_result
        if tuning_result is None:
            return

        self._log_metric(f"{prefix}cv.total_time", tuning_result.total_time)

    def _log_test_metrics(
        self,
        prefix: str,
        model_result: ModelRunResult,
    ) -> None:
        for test_result in model_result.test_results:
            dataset_name = test_result.dataset_name
            self._log_metric(
                f"{prefix}test.{dataset_name}.predict_time",
                test_result.predict_time,
            )
            self._log_metrics(f"{prefix}test.{dataset_name}", test_result.metrics)

        self._log_metric_deltas(
            f"{prefix}test.mimic_minus_tudd",
            model_result.final_test_metrics.mimic_minus_tudd,
        )

    def _log_metrics(
        self,
        prefix: str,
        metrics: ClassificationMetrics | RegressionMetrics,
        *,
        step: int | None = None,
    ) -> None:
        for name, value in metrics.scores.items():
            self._log_metric(f"{prefix}.{name}", value, step=step)
        if isinstance(metrics, ClassificationMetrics) and step is None:
            mlflow.log_param(f"{prefix}.n_classes", metrics.n_classes)

    def _log_metric_deltas(
        self,
        prefix: str,
        deltas: ClassificationMetricDeltas,
    ) -> None:
        for name, value in deltas.scores.items():
            self._log_metric(f"{prefix}.{name}", value)

    def _log_metric(
        self,
        name: str,
        value: float | int | None,
        *,
        step: int | None = None,
    ) -> None:
        if value is None:
            return
        value = float(value)
        if math.isfinite(value):
            mlflow.log_metric(name, value, step=step)

    def _log_evaluation_tables(
        self,
        params: PipelineParams,
        result: PipelineResult,
        *,
        model_filter: tuple[str, ModelTrainingResult, ModelRunResult] | None = None,
    ) -> None:
        if model_filter is None:
            model_rows = tuple(
                (model_id, training_result, model_result)
                for model_id, training_result, model_result in zip(
                    _result_model_instance_ids(result.training_results),
                    result.training_results,
                    _model_results_by_training_result(result),
                    strict=False,
                )
                if model_result is not None
            )
        else:
            model_rows = (model_filter,)

        evaluations = []
        table_rows = []
        for model_id, _training_result, model_result in model_rows:
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

        if not evaluations:
            return

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            log_evaluations(evaluations=evaluations)
        mlflow.log_table(
            data=_table_columns(table_rows), artifact_file="evaluation_metrics.json"
        )

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

        log_path = Path(config.dir_log / "active.log")
        if log_path.exists():
            mlflow.log_artifact(str(log_path), artifact_path="environment")


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


def _model_results_by_training_result(
    result: PipelineResult,
) -> list[ModelRunResult | None]:
    model_results = iter(result.model_results)
    paired_results: list[ModelRunResult | None] = []
    for training_result in result.training_results:
        if training_result.succeeded:
            paired_results.append(next(model_results, None))
        else:
            paired_results.append(None)
    return paired_results


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


def _make_evaluation(
    params: PipelineParams,
    model_id: str,
    model_name: str,
    dataset_name: str,
    scope: str,
    metrics: dict[str, float | int],
) -> Evaluation:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        return Evaluation(
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


def _candidate_ranks(scores: list[float]) -> list[int]:
    ranked_indices = sorted(
        range(len(scores)), key=lambda index: scores[index], reverse=True
    )
    ranks = [0] * len(scores)
    for rank, index in enumerate(ranked_indices, start=1):
        ranks[index] = rank
    return ranks


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


def _table_columns(rows: list[dict[str, str | float]]) -> dict[str, list[str | float]]:
    columns = {
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


def _param_value(value: Any) -> str:
    if isinstance(value, str | int | float | bool) or value is None:
        text = str(value)
    else:
        text = json.dumps(value, sort_keys=True, default=str)
    return text


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
