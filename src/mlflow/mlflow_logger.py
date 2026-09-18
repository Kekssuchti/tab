from __future__ import annotations

import json
import logging
import platform
import sys
from contextlib import contextmanager
from dataclasses import asdict
from functools import cache
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

import mlflow
from mlflow.entities import Run
from src.classes.data_registry import dataset_task_for_target
from src.config import config
from src.mlflow.tracking_contract import (
    ARTIFACT_ACTIVE_LOG,
    ARTIFACT_BOOTSTRAP_METRICS,
    ARTIFACT_CLASSIFICATION_METRICS,
    ARTIFACT_CONFIG,
    ARTIFACT_CONFIG_YAML,
    ARTIFACT_CV_RESULTS,
    ARTIFACT_DATASET_SUMMARY,
    ARTIFACT_ENVIRONMENT,
    ARTIFACT_PAIRWISE_WINS,
    ARTIFACT_TEST_PREDICTIONS,
    METRIC_TRAIN_FIT_TIME,
    RUN_TYPE_MODEL,
    RUN_TYPE_PIPELINE,
    STATUS_SUCCESS,
    TAG_MODEL_INSTANCE,
    TAG_MODEL_INSTANCES,
    TAG_MODEL_NAME,
    TAG_PIPELINE_ID,
    TAG_PIPELINE_MLFLOW_RUN_ID,
    TAG_RUN_TYPE,
    TAG_STATUS,
    TAG_TARGET,
    TAG_TASK_TYPE,
    TAG_TRACKING_SCHEMA_VERSION,
    TAG_TRAIN_SOURCES,
    TAG_TRAINED_ON,
    TAG_TRAINING_SIZE,
    TRACKING_SCHEMA_VERSION,
    test_predict_time_metric,
    test_score_metric,
)
from src.schemas.pipeline_schemas import PipelineConfig
from src.schemas.run_records import ModelRunRecord, PipelineRunRecord
from src.utils.prediction_metrics import ClassificationModelEvaluation
from src.utils.prediction_tables import PREDICTION_MANIFEST_FILENAME, PredictionTableAccumulator


class MLflowPipelineLogger:
    """Log compact metrics, predictions, and reconstruction metadata."""

    def __init__(self, source_config_path: str | Path | None = None) -> None:
        self.source_config_path = Path(source_config_path) if source_config_path is not None else None

    def log_model_run(
        self,
        params: PipelineConfig,
        result: PipelineRunRecord,
        model_run: ModelRunRecord,
        prediction_tables: PredictionTableAccumulator,
    ) -> None:
        self._configure(params)
        with self._parent_run(params) as parent:
            self._log_parent_metadata(params, result)
            self._log_prediction_tables(prediction_tables)
            if model_run.evaluation is None:
                return

            with mlflow.start_run(run_name=model_run.model_instance_id, nested=True):
                mlflow.set_tags(
                    {
                        TAG_RUN_TYPE: RUN_TYPE_MODEL,
                        TAG_TRACKING_SCHEMA_VERSION: TRACKING_SCHEMA_VERSION,
                        TAG_PIPELINE_MLFLOW_RUN_ID: parent.info.run_id,
                        TAG_MODEL_NAME: model_run.model_name,
                        TAG_MODEL_INSTANCE: model_run.model_instance_id,
                        TAG_TARGET: params.dataset.target,
                        TAG_TASK_TYPE: model_run.training_result.task_type,
                        TAG_STATUS: STATUS_SUCCESS,
                        **_training_tags(params),
                    }
                )
                metrics = {METRIC_TRAIN_FIT_TIME: model_run.evaluation.fit_time}
                for test_result in model_run.evaluation.test_results:
                    metrics[test_predict_time_metric(test_result.dataset_name)] = test_result.predict_time
                    metrics.update(
                        {
                            test_score_metric(test_result.dataset_name, metric): value
                            for metric, value in test_result.metrics.scores.items()
                        }
                    )
                mlflow.log_metrics(metrics)

    def log_pipeline_summary(
        self,
        params: PipelineConfig,
        result: PipelineRunRecord,
        prediction_tables: PredictionTableAccumulator,
        classification_evaluation: ClassificationModelEvaluation | None,
    ) -> None:
        self._configure(params)
        with self._parent_run(params):
            self._log_parent_metadata(params, result)
            self._log_prediction_tables(prediction_tables)
            if classification_evaluation is not None:
                self._log_classification_evaluation(classification_evaluation)

    def _configure(self, params: PipelineConfig) -> None:
        mlflow.set_tracking_uri(params.mlflow.tracking_uri)
        experiment = mlflow.MlflowClient().get_experiment_by_name(params.mlflow.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                params.mlflow.experiment_name,
                artifact_location=params.mlflow.artifact_location,
            )
            mlflow.set_experiment(experiment_id=experiment_id)
        else:
            mlflow.set_experiment(experiment_name=params.mlflow.experiment_name)

    @contextmanager
    def _parent_run(self, params: PipelineConfig):
        existing = _find_pipeline_run(params)
        context = (
            mlflow.start_run(run_name=params.mlflow.run_name or params.run_id)
            if existing is None
            else mlflow.start_run(run_id=existing.info.run_id)
        )
        with context as run:
            yield run

    def _log_parent_metadata(self, params: PipelineConfig, result: PipelineRunRecord) -> None:
        successful_models = [run.model_instance_id for run in result.model_runs if run.succeeded]
        mlflow.set_tags(
            {
                TAG_RUN_TYPE: RUN_TYPE_PIPELINE,
                TAG_TRACKING_SCHEMA_VERSION: TRACKING_SCHEMA_VERSION,
                TAG_PIPELINE_ID: params.run_id,
                TAG_TARGET: params.dataset.target,
                TAG_TASK_TYPE: dataset_task_for_target(params.dataset.target).task_type,
                TAG_MODEL_INSTANCES: ",".join(successful_models),
                TAG_TRAINING_SIZE: str(result.dataset_summary.train.row_count),
                **_training_tags(params),
            }
        )
        mlflow.log_params(
            {
                "dataset.train.row_count": result.dataset_summary.train.row_count,
                "dataset.test.mimic.row_count": result.dataset_summary.test_mimic.row_count,
                "dataset.test.tudd.row_count": result.dataset_summary.test_tudd.row_count,
            }
        )
        self._log_reconstruction_artifacts(params, result)

    def _log_reconstruction_artifacts(self, params: PipelineConfig, result: PipelineRunRecord) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / ARTIFACT_CONFIG
            config_yaml_path = root / ARTIFACT_CONFIG_YAML
            dataset_summary_path = root / ARTIFACT_DATASET_SUMMARY
            environment_path = root / ARTIFACT_ENVIRONMENT
            config_path.write_text(json.dumps(params.model_dump(mode="json"), indent=2), encoding="utf-8")
            if self.source_config_path is None:
                config_yaml_path.write_text(
                    yaml.safe_dump(params.model_dump(mode="json"), sort_keys=False),
                    encoding="utf-8",
                )
            else:
                config_yaml_path.write_bytes(self.source_config_path.read_bytes())
            dataset_summary_path.write_text(json.dumps(asdict(result.dataset_summary), indent=2), encoding="utf-8")
            environment_path.write_text(json.dumps(_environment(), indent=2), encoding="utf-8")
            mlflow.log_artifact(str(config_path))
            mlflow.log_artifact(str(config_yaml_path))
            mlflow.log_artifact(str(dataset_summary_path))
            mlflow.log_artifact(str(environment_path))

            cv_dir = root / ARTIFACT_CV_RESULTS
            for model_run in result.model_runs:
                tuning = model_run.training_result.tuning_result
                if tuning is None:
                    continue
                cv_dir.mkdir(exist_ok=True)
                payload = {
                    "model_instance": model_run.model_instance_id,
                    "model_name": model_run.model_name,
                    "method": tuning.method,
                    "scoring": tuning.scoring,
                    "best_params": tuning.best_params,
                    "folds": [
                        {
                            "candidate_index": fold.candidate_index,
                            "fold_index": fold.fold_index,
                            "model_params": fold.model_params,
                            "metrics": fold.metrics.scores,
                            "time": fold.time,
                        }
                        for fold in tuning.fold_results
                    ],
                }
                (cv_dir / f"{model_run.model_instance_id}.json").write_text(
                    json.dumps(payload, indent=2),
                    encoding="utf-8",
                )
            if cv_dir.exists():
                mlflow.log_artifacts(str(cv_dir), artifact_path=ARTIFACT_CV_RESULTS)

        for handler in logging.getLogger().handlers:
            handler.flush()
        active_log = config.dir_log / ARTIFACT_ACTIVE_LOG
        if active_log.exists():
            mlflow.log_artifact(str(active_log))

    def _log_prediction_tables(self, prediction_tables: PredictionTableAccumulator) -> None:
        if not prediction_tables:
            return
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir) / ARTIFACT_TEST_PREDICTIONS
            prediction_tables.write(directory)
            for dataset in ("mimic", "tudd"):
                mlflow.log_artifact(str(directory / f"{dataset}.csv"), artifact_path=ARTIFACT_TEST_PREDICTIONS)
            mlflow.log_artifact(
                str(directory / PREDICTION_MANIFEST_FILENAME),
                artifact_path=ARTIFACT_TEST_PREDICTIONS,
            )

    def _log_classification_evaluation(self, evaluation: ClassificationModelEvaluation) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metrics_path = root / ARTIFACT_CLASSIFICATION_METRICS
            bootstrap_path = root / ARTIFACT_BOOTSTRAP_METRICS
            evaluation.metrics.to_csv(metrics_path, index=False)
            evaluation.bootstrap_scores.to_csv(bootstrap_path, index=False)
            mlflow.log_artifact(str(metrics_path))
            mlflow.log_artifact(str(bootstrap_path))

            pairwise_dir = root / ARTIFACT_PAIRWISE_WINS
            pairwise_dir.mkdir()
            for name, matrix in evaluation.pairwise_wins.items():
                matrix.to_csv(pairwise_dir / f"{name}.csv")
            mlflow.log_artifacts(str(pairwise_dir), artifact_path=ARTIFACT_PAIRWISE_WINS)


@cache
def _environment() -> dict[str, object]:
    packages = {
        name: distribution.version
        for distribution in metadata.distributions()
        if (name := distribution.metadata.get("Name"))
    }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
    }


def _training_tags(params: PipelineConfig) -> dict[str, str]:
    sources = [split.dataset for split in params.dataset.train_on]
    return {
        TAG_TRAINED_ON: sources[0] if len(sources) == 1 else "mixed",
        TAG_TRAIN_SOURCES: ",".join(sources),
    }


def _find_pipeline_run(params: PipelineConfig) -> Run | None:
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(params.mlflow.experiment_name)
    if experiment is None:
        return None
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=(
            f"tags.{TAG_PIPELINE_ID} = '{params.run_id}' "
            f"and tags.{TAG_RUN_TYPE} = '{RUN_TYPE_PIPELINE}' "
            f"and tags.{TAG_TRACKING_SCHEMA_VERSION} = '{TRACKING_SCHEMA_VERSION}'"
        ),
        max_results=1,
        order_by=["attributes.start_time ASC"],
    )
    return runs[0] if runs else None
