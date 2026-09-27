from functools import partial

import numpy as np
import pytest

import mlflow
from src.config import config
from src.interfaces.model_interface import PreprocessedModelAdapter
from src.utils.prediction_metrics import evaluate_classification_models


@pytest.fixture
def isolated_runtime(tmp_path, monkeypatch):
    """Redirect data/tracking outputs, not dataset/training/evaluation behavior."""
    import src.run_pipeline as runner

    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.setattr(config, "dir_data", tmp_path / "data")
    # Do not upload the workspace's active log (which may contain private runs).
    monkeypatch.setattr(config, "dir_log", tmp_path / "logs")
    monkeypatch.setattr(
        runner,
        "evaluate_classification_models",
        partial(evaluate_classification_models, n_bootstrap=32),
    )
    yield tmp_path
    assert mlflow.active_run() is None
    mlflow.set_tracking_uri(previous_uri)


@pytest.fixture
def observed_adapter(monkeypatch):
    """Observe real adapter inputs without retaining any live estimator."""
    observations = {"fits": [], "predictions": []}
    original_fit = PreprocessedModelAdapter.fit
    original_predict = PreprocessedModelAdapter.predict

    def fit(self, X, y):
        elapsed = original_fit(self, X, y)
        observations["fits"].append(
            {"X": X.copy(), "y": np.asarray(y).copy(), "seed": self.random_state}
        )
        return elapsed

    def predict(self, X):
        prediction = original_predict(self, X)
        observations["predictions"].append(
            {"X": X.copy(), "values": prediction.values.copy(), "seed": self.random_state}
        )
        return prediction

    monkeypatch.setattr(PreprocessedModelAdapter, "fit", fit)
    monkeypatch.setattr(PreprocessedModelAdapter, "predict", predict)
    return observations
