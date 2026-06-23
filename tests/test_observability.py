from pathlib import Path

import mlflow
from src.classes.pipeline import (
    ModelRunResult,
    PipelineResult,
)
from src.classes.pipeline import (
    TestSetEvaluationResult as EvaluationResult,
)
from src.evaluation.evaluation_utils import ClassificationMetrics
from src.mlflow.mlflow_logger import MLflowPipelineLogger
from src.mlflow.serialization import pipeline_result_to_dict
from src.schemas.dataset_schemas import (
    DatasetFileSummary,
    DatasetPartSummary,
    DatasetSummary,
    DataSplitParams,
)
from src.schemas.pipeline_schemas import MLflowParams, PipelineParams
from src.schemas.plotting_schemas import PlottingParams
from src.schemas.training_schemas import (
    ModelParams,
    ModelTrainingResult,
)


class _FakeModel:
    pass


def _metrics() -> ClassificationMetrics:
    return ClassificationMetrics(
        primary_metric="accuracy",
        primary_score=1.0,
        roc_auc=1.0,
        prc_auc=1.0,
        f1=1.0,
        accuracy=1.0,
        sensitivity=1.0,
        precision=1.0,
        n_classes=2,
    )


def _params(tracking_uri: str, artifact_location: str | None = None) -> PipelineParams:
    return PipelineParams(
        run_number=7,
        run_date="2026-06-23",
        dataset={
            "target": "mortality",
            "random_state": 42,
            "train_size": 0.75,
            "train_on": (DataSplitParams(dataset="mimic", fraction=1.0),),
        },
        training=(
            ModelParams(
                name="logistic-regression",
                task_type="classification",
                params={"max_iter": 100},
                preprocessing={
                    "imputer": {"imputation_method": "mean"},
                    "scaler_encoder": {"type": "standardization"},
                },
            ),
        ),
        plotting=PlottingParams(enabled=False),
        mlflow=MLflowParams(
            enabled=True,
            tracking_uri=tracking_uri,
            artifact_location=artifact_location,
            experiment_name="test-tab",
            nested_model_runs=True,
        ),
    )


def _result() -> PipelineResult:
    metrics = _metrics()
    training_result = ModelTrainingResult(
        model_name="logistic-regression",
        task_type="classification",
        trained_model=_FakeModel(),
        tuned=False,
        fit_time=0.2,
        training_metrics=metrics,
    )
    model_result = ModelRunResult(
        model_name="logistic-regression",
        fit_time=0.2,
        test_results=(
            EvaluationResult("mimic", metrics, 0.03),
            EvaluationResult("tudd", metrics, 0.04),
        ),
    )
    dataset_summary = DatasetSummary(
        target="mortality",
        train=DatasetPartSummary(row_count=8, class_balance={"0": 4, "1": 4}),
        test_mimic=DatasetPartSummary(row_count=4, class_balance={"0": 2, "1": 2}),
        test_tudd=DatasetPartSummary(row_count=4, class_balance={"0": 2, "1": 2}),
        data_files=(
            DatasetFileSummary(
                dataset_name="mimic",
                data_origin="mimic",
                file_name="mimic.csv",
                path="/tmp/mimic.csv",
                sha256="a" * 64,
            ),
            DatasetFileSummary(
                dataset_name="tudd",
                data_origin="tudd",
                file_name="tudd.csv",
                path="/tmp/tudd.csv",
                sha256="b" * 64,
            ),
        ),
    )
    return PipelineResult(
        run_id="0007_2026-06-23",
        dataset_summary=dataset_summary,
        model_results=(model_result,),
        training_results=(training_result,),
        total_time=0.5,
    )


def test_pipeline_result_serialization_omits_trained_model():
    serialized = pipeline_result_to_dict(_result())

    assert serialized["run_id"] == "0007_2026-06-23"
    assert serialized["dataset_summary"]["train"]["row_count"] == 8
    assert "trained_model" not in serialized["training_results"][0]
    assert serialized["training_results"][0]["training_metrics"]["accuracy"] == 1.0


def test_mlflow_logger_writes_parent_and_nested_model_runs(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_location = str(tmp_path / "mlartifacts")
    params = _params(tracking_uri, artifact_location)
    result = _result()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run_number: 7\n", encoding="utf-8")

    MLflowPipelineLogger().log_pipeline_run(
        params,
        result,
        config_path=config_path,
    )

    mlflow.set_tracking_uri(tracking_uri)
    runs = mlflow.search_runs(
        experiment_names=["test-tab"],
        output_format="list",
    )

    run_names = {run.data.tags["mlflow.runName"] for run in runs}
    assert run_names == {"0007_2026-06-23", "logistic-regression"}

    parent = next(
        run for run in runs if run.data.tags["mlflow.runName"] == "0007_2026-06-23"
    )
    child = next(
        run for run in runs if run.data.tags["mlflow.runName"] == "logistic-regression"
    )

    assert parent.data.params["dataset.target"] == "mortality"
    assert parent.data.params["dataset.train.row_count"] == "8"
    assert parent.data.params["model.logistic-regression.preprocessing.override"] == "True"
    assert parent.data.metrics["pipeline.total_time"] == 0.5
    assert child.data.tags["mlflow.parentRunId"] == parent.info.run_id
    assert child.data.metrics["train.accuracy"] == 1.0

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    artifact_names = {
        artifact.path for artifact in client.list_artifacts(parent.info.run_id)
    }
    assert {"config.json", "pipeline_result.json", "environment.json"} <= artifact_names
    assert (tmp_path / "mlflow.db").exists()
    assert (tmp_path / "mlartifacts").exists()
