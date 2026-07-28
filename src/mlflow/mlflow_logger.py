from __future__ import annotations

import json
import sys
import warnings
from contextlib import contextmanager
from dataclasses import replace
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
from mlflow.entities import Run
from mlflow.evaluation import Evaluation, log_evaluations

from src.schemas.run_records import ModelRunRecord, PipelineRunRecord
from src.config import config
from src.mlflow.observation import (
    EvaluationLog,
    RunObservation,
    assemble_pipeline_observation,
    table_rows_to_columns,
)
from src.mlflow.serialization import (
    pipeline_config_to_dict,
    pipeline_result_to_dict,
    training_result_to_dict,
)
from src.mlflow.tracking_contract import (
    RUN_TYPE_PIPELINE,
    TAG_MODEL_MLFLOW_RUN_ID,
    TAG_PIPELINE_ID,
    TAG_PIPELINE_MLFLOW_RUN_ID,
    TAG_RUN_TYPE,
)
from src.schemas.pipeline_schemas import PipelineConfig


class MLflowPipelineLogger:
    def log_pipeline_run(
        self,
        params: PipelineConfig,
        result: PipelineRunRecord,
        *,
        config_path: Path | None = None,
    ) -> None:
        mlflow.set_tracking_uri(params.mlflow.tracking_uri)
        _set_experiment(params)

        observation = assemble_pipeline_observation(params, result)
        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            artifact_paths = self._write_artifacts(params, result, temp_dir)

            with mlflow.start_run(run_name=observation.run_name) as pipeline_run:
                self._log_observation(observation)
                self._log_artifacts(artifact_paths, config_path)
                self._log_model_runs(
                    observation.children,
                    artifact_paths["cv_dir"],
                    pipeline_mlflow_run_id=pipeline_run.info.run_id,
                )

    def log_model_run(
        self,
        params: PipelineConfig,
        result: PipelineRunRecord,
        model_run: ModelRunRecord,
        *,
        config_path: Path | None = None,
    ) -> None:
        mlflow.set_tracking_uri(params.mlflow.tracking_uri)
        _set_experiment(params)

        observation = assemble_pipeline_observation(params, result)
        model_observation = _find_child_observation(
            observation,
            model_run.model_instance_id,
        )

        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            artifact_paths = self._write_artifacts(params, result, temp_dir)

            with self._start_or_resume_pipeline_run(
                params,
                observation,
                artifact_paths,
                config_path,
            ) as pipeline_run:
                self._log_model_runs(
                    (model_observation,),
                    artifact_paths["cv_dir"],
                    pipeline_mlflow_run_id=pipeline_run.info.run_id,
                )

    def log_pipeline_summary(
        self,
        params: PipelineConfig,
        result: PipelineRunRecord,
        *,
        config_path: Path | None = None,
    ) -> None:
        mlflow.set_tracking_uri(params.mlflow.tracking_uri)
        _set_experiment(params)

        observation = assemble_pipeline_observation(params, result)
        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            artifact_paths = self._write_artifacts(params, result, temp_dir)
            with self._start_or_resume_pipeline_run(
                params,
                observation,
                artifact_paths,
                config_path,
                include_evaluations=True,
            ):
                pass

    @contextmanager
    def _start_or_resume_pipeline_run(
        self,
        params: PipelineConfig,
        observation: RunObservation,
        artifact_paths: dict[str, Path],
        config_path: Path | None,
        *,
        include_evaluations: bool = False,
    ):
        existing_run = _find_pipeline_run(params)
        if existing_run is None:
            run_context = mlflow.start_run(run_name=observation.run_name)
        else:
            run_context = mlflow.start_run(run_id=existing_run.info.run_id)

        with run_context as pipeline_run:
            parent_observation = replace(
                observation,
                children=(),
                evaluations=observation.evaluations if include_evaluations else (),
                table_rows=observation.table_rows if include_evaluations else (),
            )
            self._log_observation(parent_observation)
            self._log_artifacts(artifact_paths, config_path)
            yield pipeline_run

    def _log_model_runs(
        self,
        model_runs: tuple[RunObservation, ...],
        cv_dir: Path,
        *,
        pipeline_mlflow_run_id: str,
    ) -> None:
        for model_run in model_runs:
            with mlflow.start_run(
                run_name=model_run.run_name,
                nested=True,
            ) as active_run:
                model_mlflow_run_id = active_run.info.run_id
                self._log_observation(
                    model_run,
                    extra_tags={
                        TAG_PIPELINE_MLFLOW_RUN_ID: pipeline_mlflow_run_id,
                        TAG_MODEL_MLFLOW_RUN_ID: model_mlflow_run_id,
                    },
                )
                self._log_cv_artifact(model_run, cv_dir)
                self._log_cv_candidate_runs(
                    model_run.children,
                    pipeline_mlflow_run_id=pipeline_mlflow_run_id,
                    model_mlflow_run_id=model_mlflow_run_id,
                )

    def _log_cv_candidate_runs(
        self,
        cv_runs: tuple[RunObservation, ...],
        *,
        pipeline_mlflow_run_id: str,
        model_mlflow_run_id: str,
    ) -> None:
        for cv_run in cv_runs:
            with mlflow.start_run(run_name=cv_run.run_name, nested=True):
                self._log_observation(
                    cv_run,
                    extra_tags={
                        TAG_PIPELINE_MLFLOW_RUN_ID: pipeline_mlflow_run_id,
                        TAG_MODEL_MLFLOW_RUN_ID: model_mlflow_run_id,
                    },
                )

    def _log_observation(
        self,
        observation: RunObservation,
        *,
        extra_tags: dict[str, str] | None = None,
    ) -> None:
        tags = observation.tags if extra_tags is None else observation.tags | extra_tags
        mlflow.set_tags(tags)

        for key, value in observation.params.items():
            mlflow.log_param(key, value)
        for metric in observation.metrics:
            mlflow.log_metric(metric.name, metric.value, step=metric.step)

        self._log_evaluations(observation)

    def _log_evaluations(self, observation: RunObservation) -> None:
        if not observation.evaluations:
            return

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            log_evaluations(evaluations=[_make_mlflow_evaluation(evaluation) for evaluation in observation.evaluations])
        mlflow.log_table(
            data=table_rows_to_columns(observation.table_rows),
            artifact_file="evaluation_metrics.json",
        )

    def _write_artifacts(
        self,
        params: PipelineConfig,
        result: PipelineRunRecord,
        temp_dir: Path,
    ) -> dict[str, Path]:
        config_path = temp_dir / "config.json"
        result_path = temp_dir / "pipeline_result.json"
        environment_path = temp_dir / "environment.json"
        cv_dir = temp_dir / "cv_results"
        cv_dir.mkdir()

        config_path.write_text(json.dumps(pipeline_config_to_dict(params), indent=2), encoding="utf-8")
        result_path.write_text(json.dumps(pipeline_result_to_dict(result), indent=2), encoding="utf-8")
        environment_path.write_text(json.dumps(_environment_info(), indent=2), encoding="utf-8")

        for model_run in result.model_runs:
            training_result = model_run.training_result
            if training_result.tuning_result is None:
                continue
            cv_path = cv_dir / f"{model_run.model_instance_id}.json"
            cv_path.write_text(
                json.dumps(
                    training_result_to_dict(training_result)["tuning_result"],
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
            mlflow.log_artifacts(str(artifact_paths["cv_dir"]), artifact_path="cv_results")

        if config_path is not None and config_path.exists():
            mlflow.log_artifact(str(config_path), artifact_path="config_source")

        uv_lock = config.dir_root / "uv.lock"
        if uv_lock.exists():
            mlflow.log_artifact(str(uv_lock), artifact_path="environment")

        log_path = config.dir_log / "active.log"
        if log_path.exists():
            mlflow.log_artifact(str(log_path), artifact_path="environment")

    def _log_cv_artifact(self, observation: RunObservation, cv_dir: Path) -> None:
        if observation.cv_artifact_model_id is None:
            return

        cv_path = cv_dir / f"{observation.cv_artifact_model_id}.json"
        if cv_path.exists():
            mlflow.log_artifact(str(cv_path), artifact_path="cv_results")


def _make_mlflow_evaluation(evaluation: EvaluationLog) -> Evaluation:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        return Evaluation(
            inputs=evaluation.inputs,
            outputs=evaluation.outputs,
            targets=evaluation.targets,
            metrics=evaluation.metrics,
            tags=evaluation.tags,
        )


def _find_child_observation(
    observation: RunObservation,
    run_name: str,
) -> RunObservation:
    for child in observation.children:
        if child.run_name == run_name:
            return child
    raise ValueError(f"No MLflow observation found for model run {run_name!r}")


def _find_pipeline_run(params: PipelineConfig) -> Run | None:
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(params.mlflow.experiment_name)
    if experiment is None:
        return None

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=(
            f"tags.{TAG_PIPELINE_ID} = '{_mlflow_filter_value(params.run_id)}' "
            f"and tags.{TAG_RUN_TYPE} = '{RUN_TYPE_PIPELINE}'"
        ),
        max_results=1,
        order_by=["attributes.start_time ASC"],
    )
    return runs[0] if runs else None


def _mlflow_filter_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _set_experiment(params: PipelineConfig) -> None:
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
