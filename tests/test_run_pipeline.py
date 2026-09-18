import numpy as np

import src.run_pipeline as run_pipeline_module
from src.run_pipeline import run_pipeline_params
from src.utils.prediction_tables import (
    BinaryTestPredictions,
    FinalTestPredictions,
    PredictionTableAccumulator,
)
from tests.factories import pipeline_config, pipeline_result


def test_run_pipeline_returns_result_when_final_mlflow_summary_fails(monkeypatch, tmp_path):
    expected = pipeline_result(tuned=True)

    class FakePipeline:
        def __init__(self, params):
            self.prediction_tables = PredictionTableAccumulator()

        def run(self, on_model_complete=None):
            return expected

    class FailingMLflowLogger:
        def log_model_run(self, *args, **kwargs):
            pass

        def log_pipeline_summary(self, *args, **kwargs):
            raise OSError("tracking unavailable")

    monkeypatch.setattr(run_pipeline_module, "Pipeline", FakePipeline)
    monkeypatch.setattr(run_pipeline_module, "MLflowPipelineLogger", FailingMLflowLogger)
    params = pipeline_config(
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifact_location=str(tmp_path / "artifacts"),
    )

    result = run_pipeline_params(params)

    assert result is expected


def test_run_pipeline_computes_confidence_intervals_after_models_finish(monkeypatch, tmp_path):
    expected = pipeline_result(tuned=True)
    captured = {}

    class FakePipeline:
        def __init__(self, params):
            self.prediction_tables = PredictionTableAccumulator({"mimic": "m", "tudd": "t"})
            predictions = FinalTestPredictions(
                mimic=BinaryTestPredictions(
                    np.array([0, 1, 2, 3]),
                    np.array([0, 0, 1, 1]),
                    np.array([0.1, 0.2, 0.8, 0.9]),
                ),
                tudd=BinaryTestPredictions(
                    np.array([0, 1, 2, 3]),
                    np.array([0, 0, 1, 1]),
                    np.array([0.2, 0.4, 0.6, 0.8]),
                ),
            )
            self.prediction_tables.add("logistic-regression", predictions)

        def run(self, on_model_complete=None):
            return expected

    class CapturingMLflowLogger:
        def log_model_run(self, *args, **kwargs):
            pass

        def log_pipeline_summary(self, params, result, prediction_tables, classification_metrics):
            captured["metrics"] = classification_metrics

    monkeypatch.setattr(run_pipeline_module, "Pipeline", FakePipeline)
    monkeypatch.setattr(run_pipeline_module, "MLflowPipelineLogger", CapturingMLflowLogger)
    params = pipeline_config(
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifact_location=str(tmp_path / "artifacts"),
    )

    result = run_pipeline_params(params)

    assert result is expected
    metrics = captured["metrics"]
    assert metrics[["scope", "dataset"]].to_records(index=False).tolist() == [
        ("test", "mimic"),
        ("test", "tudd"),
        ("test_delta", "mimic_minus_tudd"),
    ]
    assert metrics.loc[metrics["dataset"].eq("mimic"), "roc_auc_ci_lower"].notna().all()
