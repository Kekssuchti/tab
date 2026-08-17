from dataclasses import replace

import numpy as np
import pytest

from src.mlflow.serialization import (
    canonical_json,
    cv_result_from_dict,
    cv_result_from_json,
    cv_result_to_dict,
    pipeline_result_from_dict,
    pipeline_result_from_json,
    pipeline_result_to_dict,
)
from src.mlflow.tracking_contract import TRACKING_SCHEMA_VERSION
from src.schemas.dataset_schemas import ClassificationTargetSummary, DatasetSummary, RegressionTargetSummary
from src.schemas.metrics import (
    BootstrapClassificationMetrics,
    BootstrapRegressionMetrics,
    ClassificationMetrics,
    RegressionMetrics,
)
from src.schemas.run_records import (
    FoldRecord,
    ModelEvaluationRecord,
    ModelRunRecord,
    ModelTrainingResult,
    PipelineRunRecord,
    TuningRecord,
)
from src.schemas.run_records import TestSetEvaluationRecord as EvaluationRecord
from tests.factories import bootstrap_result, pipeline_result, regression_result, tuning_result


def test_pipeline_result_serialization_omits_trained_model():
    serialized = pipeline_result_to_dict(pipeline_result())
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

    tuned_serialized = pipeline_result_to_dict(pipeline_result(tuned=True))
    tuning_result = tuned_serialized["pipeline_result"]["model_runs"][0]["training_result"]["tuning_result"]
    assert "cv_results" not in tuning_result


def test_pipeline_result_serialization_round_trips_to_typed_record():
    serialized = pipeline_result_to_dict(pipeline_result(tuned=True))

    restored = pipeline_result_from_json(canonical_json(serialized))
    result = restored.pipeline_result
    source = pipeline_result(tuned=True)

    assert restored.tracking_schema_version == TRACKING_SCHEMA_VERSION
    assert isinstance(result, PipelineRunRecord)
    assert isinstance(result.dataset_summary, DatasetSummary)
    assert isinstance(result.dataset_summary.train.target_summary, ClassificationTargetSummary)
    assert isinstance(result.model_runs[0], ModelRunRecord)
    assert isinstance(result.model_runs[0].training_result, ModelTrainingResult)
    assert isinstance(result.model_runs[0].training_result.tuning_result, TuningRecord)
    assert isinstance(result.model_runs[0].training_result.tuning_result.fold_results[0], FoldRecord)
    assert isinstance(result.model_runs[0].training_result.tuning_result.fold_results[0].metrics, ClassificationMetrics)
    bootstrap = result.model_runs[0].training_result.tuning_result.final_test_metrics.mimic_test
    assert isinstance(bootstrap, BootstrapClassificationMetrics)
    assert isinstance(result.model_runs[0].evaluation, ModelEvaluationRecord)
    assert isinstance(result.model_runs[0].evaluation.test_results[0], EvaluationRecord)
    np.testing.assert_array_equal(
        result.model_runs[0].evaluation.test_results[0].metrics.confusion_matrix,
        source.model_runs[0].evaluation.test_results[0].metrics.confusion_matrix,
    )
    np.testing.assert_array_equal(
        bootstrap.metrics.confusion_matrix,
        source.model_runs[0].training_result.tuning_result.final_test_metrics.mimic_test.metrics.confusion_matrix,
    )
    assert restored.to_dict() == serialized


def test_regression_pipeline_result_round_trips_to_typed_records():
    serialized = pipeline_result_to_dict(regression_result())

    restored = pipeline_result_from_json(canonical_json(serialized)).pipeline_result

    assert restored.dataset_summary.target == "LOS"
    assert isinstance(restored.dataset_summary.train.target_summary, RegressionTargetSummary)
    assert restored.dataset_summary.train.target_summary.mean == 4.0
    training = restored.model_runs[0].training_result
    assert training.task_type == "regression"
    assert isinstance(training.tuning_result, TuningRecord)
    assert isinstance(training.tuning_result.fold_results[0].metrics, RegressionMetrics)
    assert isinstance(training.tuning_result.final_test_metrics.mimic_test, BootstrapRegressionMetrics)
    assert isinstance(restored.model_runs[0].evaluation.test_results[0].metrics, RegressionMetrics)
    assert pipeline_result_to_dict(restored) == serialized


def test_pipeline_result_serialization_rejects_unknown_and_non_finite_values():
    unsupported = pipeline_result(tuned=True)
    unsupported.model_runs[0].training_result.tuning_result.best_params["invalid"] = object()
    with pytest.raises(TypeError, match="Unsupported value"):
        pipeline_result_to_dict(unsupported)

    non_finite = pipeline_result()
    non_finite.model_runs[0].training_result.fit_time = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        pipeline_result_to_dict(non_finite)

    unsupported_version = pipeline_result_to_dict(pipeline_result())
    unsupported_version["tracking_schema_version"] = "0"
    with pytest.raises(ValueError, match="Unsupported MLflow tracking schema version"):
        pipeline_result_from_json(canonical_json(unsupported_version))


def test_pipeline_result_reader_rejects_unknown_nested_keys_and_non_finite_values():
    unknown_metric = pipeline_result_to_dict(pipeline_result())
    metrics = unknown_metric["pipeline_result"]["model_runs"][0]["evaluation"]["test_results"][0]["metrics"]
    metrics["unexpected"] = 1.0
    with pytest.raises(ValueError, match="metrics has invalid keys"):
        pipeline_result_from_dict(unknown_metric)

    non_finite = pipeline_result_to_dict(pipeline_result())
    non_finite["pipeline_result"]["model_runs"][0]["training_result"]["fit_time"] = float("inf")
    with pytest.raises(ValueError, match="must be finite"):
        pipeline_result_from_dict(non_finite)


def test_pipeline_serialization_rejects_task_summary_and_metric_family_mismatches():
    wrong_task = pipeline_result()
    wrong_task.model_runs[0].training_result.task_type = "regression"
    with pytest.raises(ValueError, match="does not match target task"):
        pipeline_result_to_dict(wrong_task)

    wrong_metrics = pipeline_result(tuned=True)
    wrong_metrics.model_runs[0].training_result.tuning_result.fold_results[0].metrics = RegressionMetrics(
        r2=0.8,
        mae=0.2,
        mse=0.1,
        rmse=0.3,
    )
    with pytest.raises(ValueError, match="does not match task type"):
        pipeline_result_to_dict(wrong_metrics)

    wrong_bootstrap = pipeline_result(tuned=True)
    tuning = wrong_bootstrap.model_runs[0].training_result.tuning_result
    final_metrics = tuning.final_test_metrics
    regression_metrics = RegressionMetrics(r2=0.8, mae=0.2, mse=0.1, rmse=0.3)
    tuning.final_test_metrics = replace(
        final_metrics,
        mimic_test=replace(final_metrics.mimic_test, metrics=regression_metrics),
    )
    with pytest.raises(ValueError, match="does not match task type"):
        pipeline_result_to_dict(wrong_bootstrap)

    malformed_reader_payload = pipeline_result_to_dict(pipeline_result())
    malformed_reader_payload["pipeline_result"]["model_runs"][0]["training_result"]["task_type"] = "regression"
    with pytest.raises(ValueError, match="invalid keys"):
        pipeline_result_from_dict(malformed_reader_payload)

    invalid_summary = regression_result(tuned=False)
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
    serialized = cv_result_to_dict("logistic-regression", "classification", tuning_result())

    restored = cv_result_from_json(canonical_json(serialized))

    assert restored.tracking_schema_version == TRACKING_SCHEMA_VERSION
    assert restored.model_instance_id == "logistic-regression"
    assert restored.task_type == "classification"
    assert isinstance(restored.tuning_result.fold_results[0].metrics, ClassificationMetrics)

    serialized["task_type"] = "regression"
    with pytest.raises(ValueError, match="invalid for regression"):
        cv_result_from_dict(serialized)

    regression_tuning = regression_result().model_runs[0].training_result.tuning_result
    regression = cv_result_from_dict(cv_result_to_dict("linear-regression", "regression", regression_tuning))
    assert regression.task_type == "regression"
    assert isinstance(regression.tuning_result.fold_results[0].metrics, RegressionMetrics)


def test_bootstrap_final_metrics_round_trip_as_typed_results():
    source = bootstrap_result()

    restored = pipeline_result_from_json(canonical_json(pipeline_result_to_dict(source))).pipeline_result
    metrics = restored.model_runs[0].training_result.tuning_result.final_test_metrics.mimic_test

    assert isinstance(metrics, BootstrapClassificationMetrics)
    assert metrics.metrics.accuracy == pytest.approx(0.85)
    assert metrics.n_bootstrap == 5000
