from typing import ClassVar

import numpy as np
import pytest

from src.interfaces.model_interface import LogTargetModelAdapter, TimedPrediction


class _RecordingRegressionAdapter:
    task_type = "regression"
    kwargs: ClassVar[dict] = {}
    model = object()

    def __init__(self, predictions):
        self.predictions = np.asarray(predictions)
        self.fit_targets = None
        self.released = False

    def fit(self, X_train, y_train):
        self.fit_targets = np.asarray(y_train)
        return 0.25

    def predict(self, X_test):
        return TimedPrediction(self.predictions, 0.1)

    def release(self):
        self.released = True


def test_log_target_adapter_transforms_fit_targets_and_inverts_predictions():
    inner = _RecordingRegressionAdapter(np.log([48.0, 96.0]))
    adapter = LogTargetModelAdapter(inner)

    fit_time = adapter.fit(np.zeros((2, 1)), np.array([24.0, 72.0]))
    prediction = adapter.predict(np.zeros((2, 1)))

    assert fit_time == pytest.approx(0.25)
    np.testing.assert_allclose(inner.fit_targets, np.log([24.0, 72.0]))
    np.testing.assert_allclose(prediction.values, [48.0, 96.0])
    assert prediction.seconds >= 0


def test_log_target_adapter_rejects_non_positive_targets_and_releases_inner_adapter():
    inner = _RecordingRegressionAdapter([0.0])
    adapter = LogTargetModelAdapter(inner)

    with pytest.raises(ValueError, match="finite, positive"):
        adapter.fit(np.zeros((1, 1)), np.array([0.0]))

    adapter.release()
    assert inner.released


def test_log_target_adapter_rejects_non_finite_inverse_predictions():
    adapter = LogTargetModelAdapter(_RecordingRegressionAdapter([1000.0]))

    with pytest.raises(ValueError, match="non-finite predictions"):
        adapter.predict(np.zeros((1, 1)))
