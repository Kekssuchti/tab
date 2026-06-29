from pathlib import Path

import mlflow

from src.classes.pipeline import (
    ModelRunRecord,
    ModelRunResult,
    PipelineResult,
)
from src.classes.pipeline import (
    TestSetEvaluationResult as EvaluationResult,
)
from src.mlflow.mlflow_logger import MLflowPipelineLogger
from src.mlflow.observation import MetricLog, assemble_pipeline_observation
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
        model_runs=(
            ModelRunRecord(
                model_instance_id="logistic-regression",
                training_result=training_result,
                model_result=model_result,
            ),
        ),
        total_time=0.5,
    )


def _failed_result() -> PipelineResult:
    result = _result()
    failed_training_result = ModelTrainingResult(
        model_name="logistic-regression",
        task_type="classification",
        trained_model=None,
        tuned=False,
        fit_time=0.1,
        error="ValueError: bad params",
        failure_stage="training",
    )
    return PipelineResult(
        run_id=result.run_id,
        dataset_summary=result.dataset_summary,
        model_runs=(
            ModelRunRecord(
                model_instance_id="logistic-regression",
                training_result=failed_training_result,
                model_result=None,
            ),
        ),
        total_time=0.2,
    )


def _metric_value(metrics: tuple[MetricLog, ...], name: str) -> float:
    return next(metric.value for metric in metrics if metric.name == name)


def test_pipeline_result_serialization_omits_trained_model():
    serialized = pipeline_result_to_dict(_result())

    assert serialized["run_id"] == "test-pipeline-id"
    assert serialized["dataset_summary"]["train"]["row_count"] == 8
    assert serialized["model_runs"][0]["model_instance_id"] == "logistic-regression"
    assert serialized["model_runs"][0]["status"] == "success"
    assert "trained_model" not in serialized["model_runs"][0]["training_result"]
    assert "trained_model" not in serialized["training_results"][0]
    assert serialized["training_results"][0]["training_metrics"]["accuracy"] == 1.0
    assert serialized["training_results"][0]["error"] is None
    assert serialized["training_results"][0]["failure_stage"] is None


def test_observation_assembly_describes_parent_model_and_cv_runs():
    observation = assemble_pipeline_observation(
        _params("sqlite:///unused", run_name="friendly-run"),
        _result(tuned=True),
    )

    assert observation.run_name == "friendly-run"
    assert observation.tags["run_type"] == "pipeline"
    assert observation.tags["trained_on"] == "mimic"
    assert observation.params["dataset.train.row_count"] == "8"
    assert (
        observation.params["model.logistic-regression.preprocessing.override"]
        == "True"
    )
    assert _metric_value(observation.metrics, "pipeline.total_time") == 0.5
    assert {row["dataset"] for row in observation.table_rows} == {
        "mimic",
        "tudd",
        "mimic_minus_tudd",
    }

    model_run = observation.children[0]
    assert model_run.run_name == "logistic-regression"
    assert model_run.tags["status"] == "success"
    assert model_run.params["model.tuning.best_params"] == '{"C": 1.0}'
    assert _metric_value(model_run.metrics, "train.accuracy") == 1.0
    assert _metric_value(model_run.metrics, "test.mimic_minus_tudd.accuracy") == 0.0

    cv0, cv1 = model_run.children
    assert cv0.run_name == "logistic-regression/cv00"
    assert cv1.tags["candidate_rank"] == "1"
    assert cv0.params["cv.params.C"] == "0.1"
    assert _metric_value(cv0.metrics, "cv.rank") == 2.0
    assert [
        (metric.step, metric.value)
        for metric in cv0.metrics
        if metric.name == "cv.accuracy"
    ] == [(0, 0.7), (1, 0.8)]


def test_observation_assembly_marks_failed_model_without_evaluations():
    observation = assemble_pipeline_observation(
        _params("sqlite:///unused", run_name="failed-run"),
        _failed_result(),
    )
    model_run = observation.children[0]

    assert model_run.tags["status"] == "failed"
    assert model_run.tags["failure_stage"] == "training"
    assert model_run.tags["error"] == "ValueError: bad params"
    assert _metric_value(model_run.metrics, "train.fit_time") == 0.1
    assert model_run.evaluations == ()
    assert model_run.children == ()


def test_observation_assembly_keeps_cv_runs_for_failed_tuned_model():
    result = _result(tuned=True)
    training_result = result.model_runs[0].training_result
    training_result.error = "RuntimeError: evaluation failed"
    training_result.failure_stage = "evaluation"
    failed_after_tuning = PipelineResult(
        run_id=result.run_id,
        dataset_summary=result.dataset_summary,
        model_runs=(
            ModelRunRecord(
                model_instance_id="logistic-regression",
                training_result=training_result,
                model_result=None,
            ),
        ),
        total_time=0.4,
    )

    observation = assemble_pipeline_observation(
        _params("sqlite:///unused", run_name="failed-tuned-run"),
        failed_after_tuning,
    )
    model_run = observation.children[0]

    assert model_run.tags["status"] == "failed"
    assert _metric_value(model_run.metrics, "cv.total_time") == 0.1
    assert [child.run_name for child in model_run.children] == [
        "logistic-regression/cv00",
        "logistic-regression/cv01",
    ]


def test_mlflow_logger_writes_nested_runs_and_artifacts(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_location = str(tmp_path / "mlartifacts")
    params = _params(tracking_uri, artifact_location, run_name="friendly-run")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run_number: 7\n", encoding="utf-8")

    MLflowPipelineLogger().log_pipeline_run(
        params,
        _result(tuned=True),
        config_path=config_path,
    )

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
    assert parent.data.metrics["pipeline.total_time"] == 0.5
    assert model.data.tags["mlflow.parentRunId"] == parent.info.run_id
    assert model.data.tags["pipeline_mlflow_run_id"] == parent.info.run_id
    assert model.data.tags["model_mlflow_run_id"] == model.info.run_id
    assert model.data.metrics["train.accuracy"] == 1.0
    assert model.data.metrics["test.mimic_minus_tudd.accuracy"] == 0.0
    assert cv0.data.tags["mlflow.parentRunId"] == model.info.run_id
    assert cv0.data.tags["candidate_rank"] == "2"
    assert cv0.data.metrics["cv.mean.accuracy"] == 0.75

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    cv0_history = client.get_metric_history(cv0.info.run_id, "cv.accuracy")
    assert [(metric.step, metric.value) for metric in cv0_history] == [
        (0, 0.7),
        (1, 0.8),
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
    evaluation_metrics = mlflow.load_table(
        "evaluation_metrics.json", run_ids=[parent.info.run_id]
    )
    assert {"mimic", "tudd", "mimic_minus_tudd"} <= set(
        evaluation_metrics["dataset"]
    )


def test_mlflow_logger_appends_model_runs_incrementally(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_location = str(tmp_path / "mlartifacts")
    params = _params(tracking_uri, artifact_location, run_name="incremental-run")
    result = _result(tuned=True)
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
    assert model.data.metrics["test.mimic.accuracy"] == 1.0

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    artifact_names = {
        artifact.path for artifact in client.list_artifacts(parent.info.run_id)
    }
    assert "pipeline_result.json" in artifact_names
    evaluation_metrics = mlflow.load_table(
        "evaluation_metrics.json", run_ids=[parent.info.run_id]
    )
    assert {"mimic", "tudd", "mimic_minus_tudd"} <= set(
        evaluation_metrics["dataset"]
    )


def test_mlflow_logger_writes_failed_nested_model_run(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_location = str(tmp_path / "mlartifacts")
    params = _params(tracking_uri, artifact_location, run_name="failed-run")

    MLflowPipelineLogger().log_pipeline_run(params, _failed_result())

    mlflow.set_tracking_uri(tracking_uri)
    runs = mlflow.search_runs(
        experiment_names=["test-tab"],
        output_format="list",
    )
    child = next(
        run for run in runs if run.data.tags["mlflow.runName"] == "logistic-regression"
    )

    assert child.data.tags["status"] == "failed"
    assert child.data.tags["failure_stage"] == "training"
    assert child.data.tags["error"] == "ValueError: bad params"
    assert child.data.metrics["train.fit_time"] == 0.1
