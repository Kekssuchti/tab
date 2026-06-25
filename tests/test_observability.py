from pathlib import Path

import mlflow
import pytest
from src.classes.pipeline import (
    ModelRunResult,
    PipelineResult,
)
from src.classes.pipeline import (
    TestSetEvaluationResult as EvaluationResult,
)
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
    FoldResult,
    ModelParams,
    ModelTrainingResult,
    TuningCVResults,
    TuningResult,
)
from src.utils.evaluation_utils import ClassificationMetrics
from src.utils.evaluation_utils import final_test_metrics


class _FakeModel:
    pass


def _metrics(value: float = 1.0) -> ClassificationMetrics:
    return ClassificationMetrics(
        roc_auc=value,
        prc_auc=value,
        f1=value,
        accuracy=value,
        sensitivity=value,
        precision=value,
        n_classes=2,
    )


def _tuning_result() -> TuningResult:
    cv0_fold0 = _metrics(0.7)
    cv0_fold1 = _metrics(0.8)
    cv1_fold0 = _metrics(0.9)
    cv1_fold1 = _metrics(1.0)
    return TuningResult(
        best_params={"C": 1.0},
        scoring="accuracy",
        best_metrics=_metrics(0.95),
        cv_results=TuningCVResults(
            params=[{"C": 0.1}, {"C": 1.0}],
            mean_scores=[0.75, 0.95],
            std_scores=[0.05, 0.05],
            fold_scores=[[0.7, 0.8], [0.9, 1.0]],
            fold_times=[[0.01, 0.02], [0.03, 0.04]],
            mean_metrics=[_metrics(0.75), _metrics(0.95)],
        ),
        fold_results=[
            FoldResult(0, 0, cv0_fold0, 0.01, {"C": 0.1}),
            FoldResult(0, 1, cv0_fold1, 0.02, {"C": 0.1}),
            FoldResult(1, 0, cv1_fold0, 0.03, {"C": 1.0}),
            FoldResult(1, 1, cv1_fold1, 0.04, {"C": 1.0}),
        ],
    )


def _params(
    tracking_uri: str,
    artifact_location: str | None = None,
    run_name: str | None = None,
) -> PipelineParams:
    return PipelineParams(
        run_id="test-pipeline-id",
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
            run_name=run_name,
            nested_model_runs=True,
        ),
    )


def _result(*, tuned: bool = False) -> PipelineResult:
    metrics = _metrics()
    tuning_result = _tuning_result() if tuned else None
    training_result = ModelTrainingResult(
        model_name="logistic-regression",
        task_type="classification",
        trained_model=_FakeModel(),
        tuned=tuned,
        fit_time=0.2,
        training_metrics=metrics,
        tuning_result=tuning_result,
    )
    model_result = ModelRunResult(
        model_name="logistic-regression",
        fit_time=0.2,
        test_results=(
            EvaluationResult("mimic", metrics, 0.03),
            EvaluationResult("tudd", metrics, 0.04),
        ),
        final_test_metrics=final_test_metrics(metrics, metrics),
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
        run_id="test-pipeline-id",
        dataset_summary=dataset_summary,
        model_results=(model_result,),
        training_results=(training_result,),
        total_time=0.5,
    )


def test_pipeline_result_serialization_omits_trained_model():
    serialized = pipeline_result_to_dict(_result())

    assert serialized["run_id"] == "test-pipeline-id"
    assert serialized["dataset_summary"]["train"]["row_count"] == 8
    assert "trained_model" not in serialized["training_results"][0]
    assert serialized["training_results"][0]["training_metrics"]["accuracy"] == 1.0
    assert "primary_metric" not in serialized["training_results"][0]["training_metrics"]
    assert (
        serialized["model_results"][0]["final_test_metrics"]["mimic_minus_tudd"][
            "accuracy"
        ]
        == 0.0
    )


def test_mlflow_logger_writes_parent_and_nested_model_runs(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_location = str(tmp_path / "mlartifacts")
    params = _params(tracking_uri, artifact_location, run_name="friendly-run")
    result = _result(tuned=True)
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
    assert run_names == {
        "friendly-run",
        "logistic-regression",
        "logistic-regression/cv00",
        "logistic-regression/cv01",
    }

    parent = next(
        run for run in runs if run.data.tags["mlflow.runName"] == "friendly-run"
    )
    child = next(
        run for run in runs if run.data.tags["mlflow.runName"] == "logistic-regression"
    )
    cv0 = next(
        run
        for run in runs
        if run.data.tags["mlflow.runName"] == "logistic-regression/cv00"
    )
    cv1 = next(
        run
        for run in runs
        if run.data.tags["mlflow.runName"] == "logistic-regression/cv01"
    )

    assert parent.data.params["dataset.target"] == "mortality"
    assert parent.data.params["run_id"] == "test-pipeline-id"
    assert parent.data.params["mlflow.run_name"] == "friendly-run"
    assert parent.data.params["dataset.train.row_count"] == "8"
    assert (
        parent.data.params["model.logistic-regression.preprocessing.override"] == "True"
    )
    assert parent.data.tags["run_type"] == "pipeline"
    assert parent.data.tags["pipeline_id"] == "test-pipeline-id"
    assert parent.data.tags["run_id"] == "test-pipeline-id"
    assert parent.data.tags["trained_on"] == "mimic"
    assert parent.data.tags["trained_models"] == "logistic-regression"
    assert parent.data.metrics["pipeline.total_time"] == 0.5
    assert not any(
        metric_name.startswith("logistic-regression.")
        for metric_name in parent.data.metrics
    )

    assert child.data.tags["mlflow.parentRunId"] == parent.info.run_id
    assert child.data.tags["run_type"] == "model"
    assert child.data.tags["pipeline_id"] == "test-pipeline-id"
    assert child.data.tags["pipeline_mlflow_run_id"] == parent.info.run_id
    assert child.data.tags["model_mlflow_run_id"] == child.info.run_id
    assert child.data.tags["model_name"] == "logistic-regression"
    assert child.data.tags["model_instance"] == "logistic-regression"
    assert child.data.tags["trained_on"] == "mimic"
    assert child.data.metrics["model.total_time"] == 0.27
    assert child.data.metrics["train.accuracy"] == 1.0
    assert "cv.best.accuracy" not in child.data.metrics
    assert "cv.best_score" not in child.data.metrics
    assert child.data.metrics["test.mimic_minus_tudd.accuracy"] == 0.0

    assert cv0.data.tags["mlflow.parentRunId"] == child.info.run_id
    assert cv0.data.tags["run_type"] == "cv_candidate"
    assert cv0.data.tags["pipeline_id"] == "test-pipeline-id"
    assert cv0.data.tags["pipeline_mlflow_run_id"] == parent.info.run_id
    assert cv0.data.tags["model_mlflow_run_id"] == child.info.run_id
    assert cv0.data.tags["model_name"] == "logistic-regression"
    assert cv0.data.tags["candidate"] == "cv00"
    assert cv0.data.tags["candidate_rank"] == "2"
    assert cv1.data.tags["candidate_rank"] == "1"
    assert cv0.data.params["cv.candidate"] == "cv00"
    assert cv0.data.params["cv.candidate_index"] == "0"
    assert cv0.data.metrics["cv.rank"] == 2.0
    assert cv0.data.metrics["cv.mean.accuracy"] == 0.75
    assert cv0.data.metrics["cv.std.accuracy"] == pytest.approx(0.05)
    assert cv1.data.metrics["cv.mean.accuracy"] == 0.95
    assert cv1.data.metrics["cv.accuracy"] == 1.0

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    cv0_history = client.get_metric_history(cv0.info.run_id, "cv.accuracy")
    assert [(metric.step, metric.value) for metric in cv0_history] == [
        (0, 0.7),
        (1, 0.8),
    ]
    cv1_history = client.get_metric_history(cv1.info.run_id, "cv.accuracy")
    assert [(metric.step, metric.value) for metric in cv1_history] == [
        (0, 0.9),
        (1, 1.0),
    ]
    artifact_names = {
        artifact.path for artifact in client.list_artifacts(parent.info.run_id)
    }
    assert {
        "config.json",
        "pipeline_result.json",
        "environment.json",
        "_evaluations.json",
        "_metrics.json",
        "evaluation_metrics.json",
        "cv_results",
    } <= artifact_names
    child_artifact_names = {
        artifact.path for artifact in client.list_artifacts(child.info.run_id)
    }
    assert {"_evaluations.json", "_metrics.json", "evaluation_metrics.json"} <= (
        child_artifact_names
    )
    evaluation_metrics = mlflow.load_table(
        "evaluation_metrics.json", run_ids=[parent.info.run_id]
    )
    assert {"mimic", "tudd", "mimic_minus_tudd"} <= set(
        evaluation_metrics["dataset"]
    )
    assert "accuracy" in set(evaluation_metrics["metric"])
    assert (tmp_path / "mlflow.db").exists()
    assert (tmp_path / "mlartifacts").exists()
