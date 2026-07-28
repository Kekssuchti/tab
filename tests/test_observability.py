import json
from dataclasses import replace
from pathlib import Path

import mlflow
import numpy as np
import pytest
from src.mlflow.mlflow_logger import MLflowPipelineLogger
from src.mlflow.evaluation_data import load_evaluation_data
from src.mlflow.observation import MetricLog, assemble_pipeline_observation
from src.mlflow.serialization import (
    artifact_manifest_from_json,
    canonical_json,
    cv_result_from_dict,
    cv_result_from_json,
    cv_result_to_dict,
    pipeline_result_from_dict,
    pipeline_result_from_json,
    pipeline_result_to_dict,
)
from src.mlflow.tracking_contract import TRACKING_SCHEMA_VERSION
from src.schemas.dataset_schemas import (
    ClassificationTargetSummary,
    DatasetFileSummary,
    DatasetPartSummary,
    DatasetSummary,
    DataSplitConfig,
    RegressionTargetSummary,
    Target,
)
from src.schemas.pipeline_schemas import MLflowConfig, PipelineConfig
from src.schemas.metrics import (
    AggregatedFinalTestMetrics,
    ClassificationMetrics,
    ClassificationMetricsAggregate,
    FinalTestMetrics,
    RegressionMetrics,
    RegressionMetricsAggregate,
)
from src.schemas.run_records import (
    FoldRecord,
    ModelEvaluationRecord,
    ModelRunRecord,
    ModelTrainingResult,
    PipelineRunRecord,
    TestSetEvaluationRecord as EvaluationRecord,
    TuningRecord,
)
from src.schemas.training_schemas import ModelConfig
from src.utils.evaluation_utils import final_test_metrics


def _metrics(value: float = 1.0) -> ClassificationMetrics:
    return ClassificationMetrics(
        roc_auc=value,
        prc_auc=value,
        f1=value,
        accuracy=value,
        sensitivity=value,
        precision=value,
        confusion_matrix=np.array([[value, 0.0], [0.0, value]]),
        n_classes=2,
    )


def _tuning_result() -> TuningRecord:
    cv0_fold0 = _metrics(0.7)
    cv0_fold1 = _metrics(0.8)
    cv1_fold0 = _metrics(0.9)
    cv1_fold1 = _metrics(1.0)
    return TuningRecord(
        best_params={"C": 1.0},
        scoring="accuracy",
        final_test_metrics=AggregatedFinalTestMetrics(
            mimic_test=ClassificationMetricsAggregate([_metrics(0.9), _metrics(1.0)]),
            mimic_prediction_time=0.03,
            tudd_test=ClassificationMetricsAggregate([_metrics(0.9), _metrics(1.0)]),
            tudd_prediction_time=0.04,
        ),
        fold_results=[
            FoldRecord(0, 0, cv0_fold0, 0.01, {"C": 0.1}),
            FoldRecord(0, 1, cv0_fold1, 0.02, {"C": 0.1}),
            FoldRecord(1, 0, cv1_fold0, 0.03, {"C": 1.0}),
            FoldRecord(1, 1, cv1_fold1, 0.04, {"C": 1.0}),
        ],
    )


def _params(
    tracking_uri: str,
    artifact_location: str | None = None,
    run_name: str | None = None,
    target: Target = "mortality",
    model_name: str = "logistic-regression",
    run_id: str = "test-pipeline-id",
) -> PipelineConfig:
    return PipelineConfig(
        run_id=run_id,
        dataset={
            "target": target,
            "random_state": 42,
            "train_size": 0.75,
            "train_on": (DataSplitConfig(dataset="mimic", fraction=1.0),),
        },
        training=(
            ModelConfig(
                name=model_name,
                preprocessing={
                    "imputer": {"imputation_method": "mean"},
                    "scaler_encoder": {"type": "standardization"},
                },
            ),
        ),
        mlflow=MLflowConfig(
            enabled=True,
            tracking_uri=tracking_uri,
            artifact_location=artifact_location,
            experiment_name="test-tab",
            run_name=run_name,
        ),
    )


def _result(*, tuned: bool = False) -> PipelineRunRecord:
    metrics = _metrics()
    tuning_result = _tuning_result() if tuned else None
    training_result = ModelTrainingResult(
        model_name="logistic-regression",
        task_type="classification",
        tuned=tuned,
        fit_time=0.2,
        tuning_result=tuning_result,
    )
    model_result = ModelEvaluationRecord(
        model_name="logistic-regression",
        fit_time=0.2,
        test_results=(
            EvaluationRecord("mimic", metrics, 0.03),
            EvaluationRecord("tudd", metrics, 0.04),
        ),
        final_test_metrics=final_test_metrics(metrics, metrics),
    )
    dataset_summary = DatasetSummary(
        target="mortality",
        train=DatasetPartSummary(
            row_count=8,
            target_summary=ClassificationTargetSummary(class_balance={"0": 4, "1": 4}),
        ),
        test_mimic=DatasetPartSummary(
            row_count=4,
            target_summary=ClassificationTargetSummary(class_balance={"0": 2, "1": 2}),
        ),
        test_tudd=DatasetPartSummary(
            row_count=4,
            target_summary=ClassificationTargetSummary(class_balance={"0": 2, "1": 2}),
        ),
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
    return PipelineRunRecord(
        run_id="test-pipeline-id",
        dataset_summary=dataset_summary,
        model_runs=(
            ModelRunRecord(
                model_instance_id="logistic-regression",
                training_result=training_result,
                evaluation=model_result,
            ),
        ),
        total_time=0.5,
    )


def _failed_result() -> PipelineRunRecord:
    result = _result()
    failed_training_result = ModelTrainingResult(
        model_name="logistic-regression",
        task_type="classification",
        tuned=False,
        fit_time=0.1,
        error="ValueError: bad params",
        failure_stage="training",
    )
    return PipelineRunRecord(
        run_id=result.run_id,
        dataset_summary=result.dataset_summary,
        model_runs=(
            ModelRunRecord(
                model_instance_id="logistic-regression",
                training_result=failed_training_result,
                evaluation=None,
            ),
        ),
        total_time=0.2,
    )


def _regression_result(*, tuned: bool = True) -> PipelineRunRecord:
    mimic_metrics = RegressionMetrics(r2=0.8, mae=0.2, mse=0.1, rmse=0.3)
    tudd_metrics = RegressionMetrics(r2=0.6, mae=0.4, mse=0.3, rmse=0.5)
    tuning_result = None
    if tuned:
        tuning_result = TuningRecord(
            best_params={"alpha": 0.5},
            scoring="rmse",
            final_test_metrics=AggregatedFinalTestMetrics(
                mimic_test=RegressionMetricsAggregate(
                    [mimic_metrics, RegressionMetrics(r2=0.7, mae=0.3, mse=0.2, rmse=0.4)]
                ),
                mimic_prediction_time=0.03,
                tudd_test=RegressionMetricsAggregate(
                    [tudd_metrics, RegressionMetrics(r2=0.5, mae=0.5, mse=0.4, rmse=0.6)]
                ),
                tudd_prediction_time=0.04,
            ),
            fold_results=[FoldRecord(0, 0, mimic_metrics, 0.01, {"alpha": 0.5})],
            method="grid",
        )
    training_result = ModelTrainingResult(
        model_name="linear-regression",
        task_type="regression",
        tuned=tuned,
        fit_time=0.2,
        tuning_result=tuning_result,
    )
    evaluation = ModelEvaluationRecord(
        model_name="linear-regression",
        fit_time=0.2,
        test_results=(
            EvaluationRecord("mimic", mimic_metrics, 0.03),
            EvaluationRecord("tudd", tudd_metrics, 0.04),
        ),
        final_test_metrics=FinalTestMetrics(
            mimic_test=mimic_metrics,
            mimic_prediction_time=0.03,
            tudd_test=tudd_metrics,
            tudd_prediction_time=0.04,
        ),
    )
    base = _result()
    return PipelineRunRecord(
        run_id="regression-pipeline-id",
        dataset_summary=DatasetSummary(
            target="LOS",
            train=DatasetPartSummary(
                row_count=8,
                target_summary=RegressionTargetSummary(count=8, mean=4.0, std=2.0, min=1.0, max=7.0),
            ),
            test_mimic=DatasetPartSummary(
                row_count=4,
                target_summary=RegressionTargetSummary(count=4, mean=3.0, std=1.0, min=2.0, max=4.0),
            ),
            test_tudd=DatasetPartSummary(
                row_count=4,
                target_summary=RegressionTargetSummary(count=4, mean=5.0, std=1.0, min=4.0, max=6.0),
            ),
            data_files=base.dataset_summary.data_files,
        ),
        model_runs=(ModelRunRecord("linear-regression", training_result, evaluation),),
        total_time=0.5,
    )


def _metric_value(metrics: tuple[MetricLog, ...], name: str) -> float:
    return next(metric.value for metric in metrics if metric.name == name)


def test_pipeline_result_serialization_omits_trained_model():
    serialized = pipeline_result_to_dict(_result())
    result = serialized["pipeline_result"]

    assert serialized["tracking_schema_version"] == TRACKING_SCHEMA_VERSION
    assert result["run_id"] == "test-pipeline-id"
    assert result["dataset_summary"]["train"]["row_count"] == 8
    assert result["model_runs"][0]["model_instance_id"] == "logistic-regression"
    assert set(result["model_runs"][0]) == {"model_instance_id", "training_result", "evaluation"}
    assert "model_results" not in result
    assert "training_results" not in result
    assert result["model_runs"][0]["training_result"]["error"] is None
    assert result["model_runs"][0]["training_result"]["failure_stage"] is None

    tuned_serialized = pipeline_result_to_dict(_result(tuned=True))
    tuning_result = tuned_serialized["pipeline_result"]["model_runs"][0]["training_result"]["tuning_result"]
    assert "cv_results" not in tuning_result


def test_pipeline_result_serialization_round_trips_to_typed_record():
    serialized = pipeline_result_to_dict(_result(tuned=True))

    restored = pipeline_result_from_json(canonical_json(serialized))
    result = restored.pipeline_result
    source = _result(tuned=True)

    assert restored.tracking_schema_version == TRACKING_SCHEMA_VERSION
    assert isinstance(result, PipelineRunRecord)
    assert isinstance(result.dataset_summary, DatasetSummary)
    assert isinstance(result.dataset_summary.train.target_summary, ClassificationTargetSummary)
    assert isinstance(result.model_runs[0], ModelRunRecord)
    assert isinstance(result.model_runs[0].training_result, ModelTrainingResult)
    assert isinstance(result.model_runs[0].training_result.tuning_result, TuningRecord)
    assert isinstance(result.model_runs[0].training_result.tuning_result.fold_results[0], FoldRecord)
    assert isinstance(result.model_runs[0].training_result.tuning_result.fold_results[0].metrics, ClassificationMetrics)
    aggregate = result.model_runs[0].training_result.tuning_result.final_test_metrics.mimic_test
    assert isinstance(aggregate, ClassificationMetricsAggregate)
    assert isinstance(result.model_runs[0].evaluation, ModelEvaluationRecord)
    assert isinstance(result.model_runs[0].evaluation.test_results[0], EvaluationRecord)
    np.testing.assert_array_equal(
        result.model_runs[0].evaluation.test_results[0].metrics.confusion_matrix,
        source.model_runs[0].evaluation.test_results[0].metrics.confusion_matrix,
    )
    np.testing.assert_array_equal(
        aggregate.mean_confusion_matrix,
        source.model_runs[0].training_result.tuning_result.final_test_metrics.mimic_test.mean_confusion_matrix,
    )
    assert restored.to_dict() == serialized


def test_regression_pipeline_result_round_trips_to_typed_records():
    serialized = pipeline_result_to_dict(_regression_result())

    restored = pipeline_result_from_json(canonical_json(serialized)).pipeline_result

    assert restored.dataset_summary.target == "LOS"
    assert isinstance(restored.dataset_summary.train.target_summary, RegressionTargetSummary)
    assert restored.dataset_summary.train.target_summary.mean == 4.0
    training = restored.model_runs[0].training_result
    assert training.task_type == "regression"
    assert isinstance(training.tuning_result, TuningRecord)
    assert isinstance(training.tuning_result.fold_results[0].metrics, RegressionMetrics)
    assert isinstance(training.tuning_result.final_test_metrics.mimic_test, RegressionMetricsAggregate)
    assert isinstance(restored.model_runs[0].evaluation.test_results[0].metrics, RegressionMetrics)
    assert pipeline_result_to_dict(restored) == serialized


def test_pipeline_result_serialization_rejects_unknown_and_non_finite_values():
    unsupported = _result(tuned=True)
    unsupported.model_runs[0].training_result.tuning_result.best_params["invalid"] = object()
    with pytest.raises(TypeError, match="Unsupported value"):
        pipeline_result_to_dict(unsupported)

    non_finite = _result()
    non_finite.model_runs[0].training_result.fit_time = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        pipeline_result_to_dict(non_finite)

    unsupported_version = pipeline_result_to_dict(_result())
    unsupported_version["tracking_schema_version"] = "0"
    with pytest.raises(ValueError, match="Unsupported MLflow tracking schema version"):
        pipeline_result_from_json(canonical_json(unsupported_version))


def test_pipeline_result_reader_rejects_unknown_nested_keys_and_non_finite_values():
    unknown_metric = pipeline_result_to_dict(_result())
    metrics = unknown_metric["pipeline_result"]["model_runs"][0]["evaluation"]["test_results"][0]["metrics"]
    metrics["unexpected"] = 1.0
    with pytest.raises(ValueError, match="metrics has invalid keys"):
        pipeline_result_from_dict(unknown_metric)

    non_finite = pipeline_result_to_dict(_result())
    non_finite["pipeline_result"]["model_runs"][0]["training_result"]["fit_time"] = float("inf")
    with pytest.raises(ValueError, match="must be finite"):
        pipeline_result_from_dict(non_finite)


def test_pipeline_projection_rejects_config_result_identity_mismatches():
    result = _result()
    with pytest.raises(ValueError, match="run_id mismatch"):
        assemble_pipeline_observation(_params("sqlite:///unused", run_id="other"), result)
    with pytest.raises(ValueError, match="target mismatch"):
        assemble_pipeline_observation(_params("sqlite:///unused", target="LOS"), result)
    with pytest.raises(ValueError, match="model instance mapping mismatch"):
        assemble_pipeline_observation(_params("sqlite:///unused", model_name="xgboost"), result)


def test_pipeline_projection_accepts_ordered_partial_model_prefix():
    params = _params("sqlite:///unused")
    params = params.model_copy(update={"training": (*params.training, ModelConfig(name="xgboost"))})

    observation = assemble_pipeline_observation(params, _result())

    assert [child.run_name for child in observation.children] == ["logistic-regression"]


def test_pipeline_serialization_rejects_task_summary_and_metric_family_mismatches():
    wrong_task = _result()
    wrong_task.model_runs[0].training_result.task_type = "regression"
    with pytest.raises(ValueError, match="does not match target task"):
        pipeline_result_to_dict(wrong_task)

    wrong_metrics = _result(tuned=True)
    wrong_metrics.model_runs[0].training_result.tuning_result.fold_results[0].metrics = RegressionMetrics(
        r2=0.8,
        mae=0.2,
        mse=0.1,
        rmse=0.3,
    )
    with pytest.raises(ValueError, match="does not match task type"):
        pipeline_result_to_dict(wrong_metrics)

    malformed_reader_payload = pipeline_result_to_dict(_result())
    malformed_reader_payload["pipeline_result"]["model_runs"][0]["training_result"]["task_type"] = "regression"
    with pytest.raises(ValueError, match="invalid keys"):
        pipeline_result_from_dict(malformed_reader_payload)

    invalid_summary = _regression_result(tuned=False)
    invalid_summary = replace(
        invalid_summary,
        dataset_summary=replace(
            invalid_summary.dataset_summary,
            train=replace(
                invalid_summary.dataset_summary.train,
                target_summary=replace(invalid_summary.dataset_summary.train.target_summary, count=9),
            ),
        ),
    )
    with pytest.raises(ValueError, match="between zero and row_count"):
        pipeline_result_to_dict(invalid_summary)


def test_cv_result_envelope_round_trips_and_rejects_task_mismatch():
    serialized = cv_result_to_dict("logistic-regression", "classification", _tuning_result())

    restored = cv_result_from_json(canonical_json(serialized))

    assert restored.tracking_schema_version == TRACKING_SCHEMA_VERSION
    assert restored.model_instance_id == "logistic-regression"
    assert restored.task_type == "classification"
    assert isinstance(restored.tuning_result.fold_results[0].metrics, ClassificationMetrics)

    serialized["task_type"] = "regression"
    with pytest.raises(ValueError, match="invalid for regression"):
        cv_result_from_dict(serialized)

    regression_tuning = _regression_result().model_runs[0].training_result.tuning_result
    regression = cv_result_from_dict(cv_result_to_dict("linear-regression", "regression", regression_tuning))
    assert regression.task_type == "regression"
    assert isinstance(regression.tuning_result.fold_results[0].metrics, RegressionMetrics)


def test_observation_assembly_describes_parent_model_and_cv_runs():
    observation = assemble_pipeline_observation(
        _params("sqlite:///unused", run_name="friendly-run"),
        _result(tuned=True),
    )

    assert observation.run_name == "friendly-run"
    assert observation.tags["run_type"] == "pipeline"
    assert observation.tags["trained_on"] == "mimic"
    assert observation.tags["task_type"] == "classification"
    assert observation.params["dataset.task_type"] == "classification"
    assert observation.params["dataset.kind"] == "normal"
    assert "dataset.classification" not in observation.params
    assert observation.params["dataset.train.row_count"] == "8"
    assert observation.params["dataset.train.class_balance.0"] == "4"
    assert all(".target.mean" not in key for key in observation.params)
    assert observation.params["model.logistic-regression.preprocessing.override"] == "True"
    assert all(not key.startswith("plotting.") for key in observation.params)
    assert all(".params." not in key for key in observation.params)
    assert _metric_value(observation.metrics, "pipeline.total_time") == 0.5
    assert {row.dataset for row in observation.table_rows} == {
        "mimic",
        "tudd",
        "mimic_minus_tudd",
    }

    model_run = observation.children[0]
    assert model_run.run_name == "logistic-regression"
    assert model_run.tags["status"] == "success"
    assert model_run.params["model.tuning.best_params"] == '{"C": 1.0}'
    assert _metric_value(model_run.metrics, "test.mimic_minus_tudd.accuracy") == 0.0
    assert _metric_value(model_run.metrics, "test.mimic.mean_accuracy") == 0.95
    assert all(metric.name != "test.mimic.accuracy" for metric in model_run.metrics)

    cv0, cv1 = model_run.children
    assert cv0.run_name == "logistic-regression/cv00"
    assert cv1.tags["candidate_rank"] == "1"
    assert cv0.params["cv.params.C"] == "0.1"
    assert _metric_value(cv0.metrics, "cv.rank") == 2.0
    assert [(metric.step, metric.value) for metric in cv0.metrics if metric.name == "cv.accuracy"] == [
        (0, 0.7),
        (1, 0.8),
    ]


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


def test_observation_only_logs_class_count_for_classification_metrics():
    regression_result = _regression_result(tuned=False)

    observation = assemble_pipeline_observation(
        _params(
            "sqlite:///unused",
            target="LOS",
            model_name="linear-regression",
            run_id="regression-pipeline-id",
        ),
        regression_result,
    )

    assert all(not key.endswith(".n_classes") for key in observation.children[0].params)
    assert observation.tags["task_type"] == "regression"
    assert observation.children[0].tags["task_type"] == "regression"
    assert observation.params["dataset.train.target.count"] == "8"
    assert observation.params["dataset.train.target.mean"] == "4.0"
    assert all("class_balance" not in key for key in observation.params)


def test_observation_logs_regression_aggregate_metrics_and_candidate():
    observation = assemble_pipeline_observation(
        _params(
            "sqlite:///unused",
            target="LOS",
            model_name="linear-regression",
            run_id="regression-pipeline-id",
        ),
        _regression_result(),
    )
    model_run = observation.children[0]

    assert _metric_value(model_run.metrics, "test.mimic.mean_r2") == pytest.approx(0.75)
    assert _metric_value(model_run.metrics, "test.mimic.mean_rmse") == pytest.approx(0.35)
    assert _metric_value(model_run.metrics, "test.mimic.ci_95_rmse_lower") <= 0.35
    assert _metric_value(model_run.metrics, "test.mimic.ci_95_rmse_upper") >= 0.35
    assert model_run.children[0].tags["task_type"] == "regression"
    assert _metric_value(model_run.children[0].metrics, "cv.mean.rmse") == pytest.approx(0.3)


def test_observation_assembly_keeps_cv_runs_for_failed_tuned_model():
    result = _result(tuned=True)
    training_result = result.model_runs[0].training_result
    training_result.error = "RuntimeError: evaluation failed"
    training_result.failure_stage = "evaluation"
    failed_after_tuning = PipelineRunRecord(
        run_id=result.run_id,
        dataset_summary=result.dataset_summary,
        model_runs=(
            ModelRunRecord(
                model_instance_id="logistic-regression",
                training_result=training_result,
                evaluation=None,
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
    assert parent.data.tags["tracking_schema_version"] == TRACKING_SCHEMA_VERSION
    assert parent.data.metrics["pipeline.total_time"] == 0.5
    assert model.data.tags["mlflow.parentRunId"] == parent.info.run_id
    assert model.data.tags["pipeline_mlflow_run_id"] == parent.info.run_id
    assert model.data.tags["model_mlflow_run_id"] == model.info.run_id
    assert model.data.metrics["test.mimic_minus_tudd.accuracy"] == 0.0
    assert model.data.metrics["test.mimic.mean_accuracy"] == 0.95
    assert "test.mimic.accuracy" not in model.data.metrics
    assert "test.mimic.roc_auc" not in model.data.metrics
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
    assert model.data.metrics["test.mimic.mean_accuracy"] == 0.95
    assert "test.mimic.accuracy" not in model.data.metrics

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    artifact_names = {artifact.path for artifact in client.list_artifacts(parent.info.run_id)}
    assert "pipeline_result.json" in artifact_names
    evaluation_metrics = mlflow.load_table("evaluation_metrics.json", run_ids=[parent.info.run_id])
    assert {"mimic", "tudd", "mimic_minus_tudd"} <= set(evaluation_metrics["dataset"])


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
    assert artifact_manifest_from_json(manifest_json).evaluation_table is None
