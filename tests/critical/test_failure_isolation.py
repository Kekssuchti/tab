"""Real model execution must survive local failures without leaking resources."""

from dataclasses import replace
from functools import partial

import numpy as np
import pandas as pd
import pytest

import src.run_pipeline as runner
from src.classes.data_registry import data_files_for_kind
from src.classes.pipeline import Pipeline
from src.config import config
from src.mlflow.mlflow_logger import MLflowPipelineLogger
from src.schemas.pipeline_schemas import PipelineConfig
from src.utils.prediction_metrics import evaluate_classification_models
from src.utils.prediction_tables import PredictionTableAccumulator
from tests.critical import test_training as training_support

# Share only this focused instrumentation, not a global autouse fixture.
model_audit = training_support.model_audit
model_config = training_support.model_config


@pytest.fixture
def pipeline_config(tmp_path, monkeypatch):
    """Redirect only data locations; Dataset, Trainer and Pipeline stay real."""
    filtered = tmp_path / "filtered"
    filtered.mkdir()
    labels = np.tile([0, 1], 20)
    for source in data_files_for_kind("normal"):
        pd.DataFrame(
            {
                "signal": 2 * labels - 1,
                "Age": 40 + np.arange(40) % 20,
                "mortality": labels,
            }
        ).to_csv(filtered / source.file_name, index=False)
    monkeypatch.setattr(config, "dir_data", tmp_path)
    return PipelineConfig.model_validate(
        {
            "random_states": {},
            "dataset": {
                "target": "mortality",
                "train_on": [{"dataset": "mimic", "fraction": 24}],
                "imputer": {"imputation_method": "none"},
                "scaler_encoder": {"type": "none"},
            },
            "training": [model_config(qualities=(1,)), model_config(qualities=(1,))],
            "mlflow": {
                "enabled": False,
                "tracking_uri": f"sqlite:///{tmp_path / 'tracking.db'}",
                "artifact_location": str(tmp_path / "artifacts"),
            },
        }
    )


@pytest.mark.parametrize(
    ("stage", "rows"),
    [
        ("fit", 12),
        ("predict", 12),
        ("evaluation", 12),
        ("fit", 24),
        ("predict", 24),
        ("evaluation", 24),
    ],
)
def test_fold_and_final_failures_release_immediately_and_later_model_runs(
    pipeline_config, model_audit, caplog, stage, rows
):
    """Distinct fit/predict/scoring boundaries exercise both cleanup finally blocks."""
    pipeline_config.training = (
        model_config(failure=stage, failure_rows=rows),
        model_config(qualities=(1,)),
    )
    pipeline = Pipeline(pipeline_config)
    callbacks = []

    def completed(partial_result, model_run):
        assert not model_audit.live  # Cleanup precedes callbacks, even on errors.
        callbacks.append((partial_result, model_run))

    result = pipeline.run(on_model_complete=completed)
    failed, succeeded = result.model_runs
    assert not failed.succeeded
    assert failed.training_result.error
    if stage != "evaluation":
        assert f"injected {stage} failure" in failed.training_result.error
    assert succeeded.succeeded
    assert succeeded.evaluation.metrics_by_test_set["mimic"].accuracy == 1.0
    assert pipeline.prediction_tables.model_instance_ids == (succeeded.model_instance_id,)
    assert [entry[1] for entry in callbacks] == list(result.model_runs)
    assert len(model_audit.fits) == (2 if rows == 12 else 6)
    # Both pytest log handlers hold these same records. Discard their traceback
    # frames, not application results/callbacks, before checking weak references.
    for record in caplog.records:
        record.exc_info = None
    caplog.clear()
    model_audit.assert_released()


@pytest.mark.parametrize("corruption", ["identity", "alignment", "accumulation"])
def test_invalid_prediction_accumulation_invalidates_only_affected_model(
    pipeline_config, model_audit, monkeypatch, caplog, corruption
):
    """Reject real malformed predictions atomically, never label them successful."""
    pipeline_config.training = (model_config(qualities=(1,)),) * 3
    original_add = PredictionTableAccumulator.add

    def corrupt_second_model(accumulator, instance_id, predictions):
        if instance_id.endswith("__1"):
            if corruption == "accumulation":
                raise OSError("injected accumulation failure")
            cohort = predictions.tudd
            if corruption == "identity":
                cohort = replace(cohort, test_set_id=cohort.test_set_id.astype(str))
            else:
                cohort = replace(cohort, y_true=1 - cohort.y_true)
            predictions = replace(predictions, tudd=cohort)
        return original_add(accumulator, instance_id, predictions)

    monkeypatch.setattr(PredictionTableAccumulator, "add", corrupt_second_model)
    pipeline = Pipeline(pipeline_config)
    callbacks = []
    result = pipeline.run(on_model_complete=lambda partial_result, model_run: callbacks.append(partial_result))
    assert [run.succeeded for run in result.model_runs] == [True, False, True]
    failed = result.model_runs[1]
    assert failed.evaluation is None  # It must not be exported as a successful model.
    expected_error = {
        "identity": "integer source-row positions",
        "alignment": "test set changed",
        "accumulation": "injected accumulation failure",
    }[corruption]
    assert expected_error in failed.training_result.error
    successful_ids = (result.model_runs[0].model_instance_id, result.model_runs[2].model_instance_id)
    assert pipeline.prediction_tables.model_instance_ids == successful_ids
    for table in pipeline.prediction_tables.frames().values():
        assert list(table.columns) == ["test_set_id", "y_true", *(f"y_pred_{name}" for name in successful_ids)]
    assert callbacks[-1].model_runs == result.model_runs
    caplog.clear()
    model_audit.assert_released()


def test_optional_tracking_outage_preserves_completed_success_and_releases_models(
    pipeline_config, model_audit, monkeypatch, caplog
):
    """One transport seam fails incremental and final logging after real training."""
    pipeline_config.mlflow.enabled = True

    def unavailable(_logger, _params):
        assert not model_audit.live
        raise OSError("tracking unavailable")

    monkeypatch.setattr(MLflowPipelineLogger, "_configure", unavailable)
    monkeypatch.setattr(
        runner, "evaluate_classification_models", partial(evaluate_classification_models, n_bootstrap=12)
    )
    result = runner.run_pipeline_params(pipeline_config)
    assert len(result.model_runs) == 2
    assert all(run.succeeded for run in result.model_runs)
    assert all(run.evaluation.metrics_by_test_set["tudd"].accuracy == 1.0 for run in result.model_runs)
    assert "tracking unavailable" in caplog.text
    caplog.clear()
    model_audit.assert_released()
