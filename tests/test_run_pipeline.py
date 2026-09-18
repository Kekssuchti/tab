import src.run_pipeline as run_pipeline_module
from src.run_pipeline import run_pipeline_params
from src.utils.prediction_tables import PredictionTableAccumulator
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
