import pytest

from src.mlflow.observation import assemble_pipeline_observation
from src.schemas.run_records import ModelRunRecord, PipelineRunRecord
from src.schemas.training_schemas import ModelConfig
from tests.factories import (
    bootstrap_result,
    failed_result,
    metric_value,
    pipeline_config,
    pipeline_result,
    regression_result,
)


def test_pipeline_projection_rejects_config_result_identity_mismatches():
    result = pipeline_result()
    with pytest.raises(ValueError, match="run_id mismatch"):
        assemble_pipeline_observation(pipeline_config("sqlite:///unused", run_id="other"), result)
    with pytest.raises(ValueError, match="target mismatch"):
        assemble_pipeline_observation(pipeline_config("sqlite:///unused", target="LOS"), result)
    with pytest.raises(ValueError, match="model instance mapping mismatch"):
        assemble_pipeline_observation(pipeline_config("sqlite:///unused", model_name="xgboost"), result)


def test_pipeline_projection_accepts_ordered_partial_model_prefix():
    params = pipeline_config("sqlite:///unused")
    params = params.model_copy(update={"training": (*params.training, ModelConfig(name="xgboost"))})

    observation = assemble_pipeline_observation(params, pipeline_result())

    assert [child.run_name for child in observation.children] == ["logistic-regression"]


def test_observation_assembly_describes_parent_model_and_cv_runs():
    observation = assemble_pipeline_observation(
        pipeline_config("sqlite:///unused", run_name="friendly-run"),
        pipeline_result(tuned=True),
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
    assert metric_value(observation.metrics, "pipeline.total_time") == 0.5
    assert {row.dataset for row in observation.table_rows} == {
        "mimic",
        "tudd",
        "mimic_minus_tudd",
    }

    model_run = observation.children[0]
    assert model_run.run_name == "logistic-regression"
    assert model_run.tags["status"] == "success"
    assert model_run.params["model.tuning.best_params"] == '{"C": 1.0}'
    assert metric_value(model_run.metrics, "test.mimic_minus_tudd.accuracy") == 0.0
    assert metric_value(model_run.metrics, "test.mimic.accuracy") == 0.95
    assert model_run.params["model.final_evaluation.method"] == "bootstrap"

    cv0, cv1 = model_run.children
    assert cv0.run_name == "logistic-regression/cv00"
    assert cv1.tags["candidate_rank"] == "1"
    assert cv0.params["cv.params.C"] == "0.1"
    assert metric_value(cv0.metrics, "cv.rank") == 2.0
    assert [(metric.step, metric.value) for metric in cv0.metrics if metric.name == "cv.accuracy"] == [
        (0, 0.7),
        (1, 0.8),
    ]


def test_observation_assembly_marks_failed_model_without_evaluations():
    observation = assemble_pipeline_observation(
        pipeline_config("sqlite:///unused", run_name="failed-run"),
        failed_result(),
    )
    model_run = observation.children[0]

    assert model_run.tags["status"] == "failed"
    assert model_run.tags["failure_stage"] == "training"
    assert model_run.tags["error"] == "ValueError: bad params"
    assert metric_value(model_run.metrics, "train.fit_time") == 0.1
    assert model_run.evaluations == ()
    assert model_run.children == ()


def test_observation_logs_bootstrap_point_metrics_and_method():
    observation = assemble_pipeline_observation(
        pipeline_config("sqlite:///unused", run_name="bootstrap-run"),
        bootstrap_result(),
    )
    model_run = observation.children[0]

    assert model_run.params["model.final_evaluation.method"] == "bootstrap"
    assert model_run.params["model.final_evaluation.n_bootstrap"] == "5000"
    assert metric_value(model_run.metrics, "test.mimic.accuracy") == pytest.approx(0.85)
    assert metric_value(model_run.metrics, "test.mimic.ci_95_accuracy_lower") == pytest.approx(0.75)


def test_observation_only_logs_class_count_for_classification_metrics():
    regression = regression_result(tuned=False)

    observation = assemble_pipeline_observation(
        pipeline_config(
            "sqlite:///unused",
            target="LOS",
            model_name="linear-regression",
            run_id="regression-pipeline-id",
        ),
        regression,
    )

    assert all(not key.endswith(".n_classes") for key in observation.children[0].params)
    assert observation.tags["task_type"] == "regression"
    assert observation.children[0].tags["task_type"] == "regression"
    assert observation.params["dataset.train.target.count"] == "8"
    assert observation.params["dataset.train.target.mean"] == "4.0"
    assert all("class_balance" not in key for key in observation.params)


def test_observation_logs_regression_bootstrap_metrics_and_candidate():
    observation = assemble_pipeline_observation(
        pipeline_config(
            "sqlite:///unused",
            target="LOS",
            model_name="linear-regression",
            run_id="regression-pipeline-id",
        ),
        regression_result(),
    )
    model_run = observation.children[0]

    assert metric_value(model_run.metrics, "test.mimic.r2") == pytest.approx(0.75)
    assert metric_value(model_run.metrics, "test.mimic.rmse") == pytest.approx(0.35)
    assert metric_value(model_run.metrics, "test.mimic.ci_95_rmse_lower") <= 0.35
    assert metric_value(model_run.metrics, "test.mimic.ci_95_rmse_upper") >= 0.35
    assert model_run.children[0].tags["task_type"] == "regression"
    assert metric_value(model_run.children[0].metrics, "cv.mean.rmse") == pytest.approx(0.3)


def test_observation_assembly_keeps_cv_runs_for_failed_tuned_model():
    result = pipeline_result(tuned=True)
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
        pipeline_config("sqlite:///unused", run_name="failed-tuned-run"),
        failed_after_tuning,
    )
    model_run = observation.children[0]

    assert model_run.tags["status"] == "failed"
    assert metric_value(model_run.metrics, "cv.total_time") == 0.1
    assert [child.run_name for child in model_run.children] == [
        "logistic-regression/cv00",
        "logistic-regression/cv01",
    ]
