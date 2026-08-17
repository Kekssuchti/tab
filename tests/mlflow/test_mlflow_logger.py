import json
from pathlib import Path

import numpy as np

import mlflow
from src.mlflow.evaluation_data import load_evaluation_data
from src.mlflow.mlflow_logger import MLflowPipelineLogger
from src.mlflow.serialization import (
    artifact_manifest_from_json,
    cv_result_from_json,
    pipeline_result_from_json,
)
from src.mlflow.tracking_contract import TRACKING_SCHEMA_VERSION
from src.schemas.metrics import ClassificationMetrics
from tests.factories import failed_result, pipeline_config, pipeline_result


def test_mlflow_logger_writes_nested_runs_and_artifacts(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_location = str(tmp_path / "mlartifacts")
    params = pipeline_config(tracking_uri, artifact_location, run_name="friendly-run")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run_number: 7\n", encoding="utf-8")

    result = pipeline_result(tuned=True)
    logger = MLflowPipelineLogger()
    logger.log_model_run(params, result, result.model_runs[0], config_path=config_path)
    logger.log_pipeline_summary(params, result, config_path=config_path)

    mlflow.set_tracking_uri(tracking_uri)
    runs = mlflow.search_runs(
        experiment_names=["test-tab"],
        output_format="list",
    )
    runs_by_name = {run.data.tags["mlflow.runName"]: run for run in runs}

    assert set(runs_by_name) == {
        "friendly-run",
        "logistic-regression",
        "logistic-regression/cv00",
        "logistic-regression/cv01",
    }
    parent = runs_by_name["friendly-run"]
    model = runs_by_name["logistic-regression"]
    cv0 = runs_by_name["logistic-regression/cv00"]

    assert parent.data.params["dataset.target"] == "mortality"
    assert parent.data.tags["tracking_schema_version"] == TRACKING_SCHEMA_VERSION
    assert parent.data.metrics["pipeline.total_time"] == 0.5
    assert model.data.tags["mlflow.parentRunId"] == parent.info.run_id
    assert model.data.tags["pipeline_mlflow_run_id"] == parent.info.run_id
    assert model.data.tags["model_mlflow_run_id"] == model.info.run_id
    assert model.data.metrics["test.mimic_minus_tudd.accuracy"] == 0.0
    assert model.data.metrics["test.mimic.accuracy"] == 0.95
    assert model.data.metrics["test.mimic.roc_auc"] == 0.95
    assert cv0.data.tags["mlflow.parentRunId"] == model.info.run_id
    assert cv0.data.tags["candidate_rank"] == "2"
    assert cv0.data.metrics["cv.mean.accuracy"] == 0.75

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    cv0_history = client.get_metric_history(cv0.info.run_id, "cv.accuracy")
    assert [(metric.step, metric.value) for metric in cv0_history] == [
        (0, 0.7),
        (1, 0.8),
    ]
    artifact_names = {artifact.path for artifact in client.list_artifacts(parent.info.run_id)}
    assert {
        "config.json",
        "pipeline_result.json",
        "environment.json",
        "tracking_manifest.json",
        "_evaluations.json",
        "_metrics.json",
        "evaluation_metrics.json",
        "cv_results",
    } <= artifact_names
    result_artifact = Path(client.download_artifacts(parent.info.run_id, "pipeline_result.json"))
    loaded_result = pipeline_result_from_json(result_artifact.read_text(encoding="utf-8"))
    assert loaded_result.pipeline_result.model_runs[0].model_instance_id == "logistic-regression"
    manifest_artifact = Path(client.download_artifacts(parent.info.run_id, "tracking_manifest.json"))
    manifest = artifact_manifest_from_json(manifest_artifact.read_text(encoding="utf-8"))
    assert manifest.pipeline_result == "pipeline_result.json"
    assert manifest.evaluation_table == "evaluation_metrics.json"
    assert manifest.cv_results == ("cv_results/logistic-regression.json",)
    cv_artifact = Path(client.download_artifacts(parent.info.run_id, "cv_results/logistic-regression.json"))
    cv_result = cv_result_from_json(cv_artifact.read_text(encoding="utf-8"))
    assert cv_result.model_instance_id == "logistic-regression"
    assert cv_result.task_type == "classification"
    assert isinstance(cv_result.tuning_result.fold_results[0].metrics, ClassificationMetrics)
    config_artifact = Path(client.download_artifacts(parent.info.run_id, "config.json"))
    logged_config = json.loads(config_artifact.read_text(encoding="utf-8"))
    assert "plotting" not in logged_config
    assert "params" not in logged_config["training"][0]
    evaluation_metrics = mlflow.load_table("evaluation_metrics.json", run_ids=[parent.info.run_id])
    assert {"mimic", "tudd", "mimic_minus_tudd"} <= set(evaluation_metrics["dataset"])

    loaded = load_evaluation_data("test-tab", tracking_uri=tracking_uri)
    accuracy = loaded.loc[(loaded["dataset"] == "mimic") & (loaded["scope"] == "test")].iloc[0]
    assert accuracy["accuracy"] == 0.95
    assert accuracy["accuracy_ci_lower"] is not None
    assert {"kind", "unit"}.isdisjoint(loaded.columns)
    assert accuracy["cv_time"] == 0.1
    assert accuracy["fit_time"] == 0.2
    assert accuracy["predict_time_mimic"] == 0.03
    assert accuracy["predict_time_tudd"] == 0.04
    assert np.isclose(accuracy["total_time"], 0.37)


def test_mlflow_logger_appends_model_runs_incrementally(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_location = str(tmp_path / "mlartifacts")
    params = pipeline_config(tracking_uri, artifact_location, run_name="incremental-run")
    result = pipeline_result(tuned=True)
    logger = MLflowPipelineLogger()

    logger.log_model_run(params, result, result.model_runs[0])
    logger.log_pipeline_summary(params, result)

    mlflow.set_tracking_uri(tracking_uri)
    runs = mlflow.search_runs(
        experiment_names=["test-tab"],
        output_format="list",
    )
    runs_by_name = {run.data.tags["mlflow.runName"]: run for run in runs}

    assert set(runs_by_name) == {
        "incremental-run",
        "logistic-regression",
        "logistic-regression/cv00",
        "logistic-regression/cv01",
    }
    parent = runs_by_name["incremental-run"]
    model = runs_by_name["logistic-regression"]

    assert parent.data.tags["pipeline_id"] == "test-pipeline-id"
    assert parent.data.metrics["pipeline.total_time"] == 0.5
    assert model.data.tags["mlflow.parentRunId"] == parent.info.run_id
    assert model.data.metrics["test.mimic.accuracy"] == 0.95

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    artifact_names = {artifact.path for artifact in client.list_artifacts(parent.info.run_id)}
    assert "pipeline_result.json" in artifact_names
    evaluation_metrics = mlflow.load_table("evaluation_metrics.json", run_ids=[parent.info.run_id])
    assert {"mimic", "tudd", "mimic_minus_tudd"} <= set(evaluation_metrics["dataset"])


def test_mlflow_logger_writes_failed_nested_model_run(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_location = str(tmp_path / "mlartifacts")
    params = pipeline_config(tracking_uri, artifact_location, run_name="failed-run")

    result = failed_result()
    logger = MLflowPipelineLogger()
    logger.log_model_run(params, result, result.model_runs[0])
    logger.log_pipeline_summary(params, result)

    mlflow.set_tracking_uri(tracking_uri)
    runs = mlflow.search_runs(
        experiment_names=["test-tab"],
        output_format="list",
    )
    child = next(run for run in runs if run.data.tags["mlflow.runName"] == "logistic-regression")

    assert child.data.tags["status"] == "failed"
    assert child.data.tags["failure_stage"] == "training"
    assert child.data.tags["error"] == "ValueError: bad params"
    assert child.data.metrics["train.fit_time"] == 0.1
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    artifact_names = {artifact.path for artifact in client.list_artifacts(child.data.tags["mlflow.parentRunId"])}
    assert "evaluation_metrics.json" not in artifact_names
    manifest_path = Path(client.download_artifacts(child.data.tags["mlflow.parentRunId"], "tracking_manifest.json"))
    manifest_json = manifest_path.read_text(encoding="utf-8")
    assert "evaluation_table" not in json.loads(manifest_json)
