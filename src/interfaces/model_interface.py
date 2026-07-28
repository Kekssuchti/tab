from abc import ABC, abstractmethod
from dataclasses import dataclass
from timeit import default_timer as timer
from typing import Any

import numpy as np

from src.schemas.base_schemas import TaskType


@dataclass(frozen=True)
class TimedPrediction:
    values: np.ndarray
    seconds: float


def _prediction_array(values: Any) -> np.ndarray:
    detach = getattr(values, "detach", None)
    if callable(detach):
        values = detach()

    cpu = getattr(values, "cpu", None)
    if callable(cpu):
        values = cpu()

    to_numpy = getattr(values, "numpy", None)
    if callable(to_numpy):
        values = to_numpy()

    return np.asarray(values)


class ModelAdapter(ABC):
    """Common interface for trainable tabular model adapters."""

    task_type: TaskType
    kwargs: dict
    model: Any

    @abstractmethod
    def fit(self, X_train, y_train) -> float:
        """
        Fit the model.

        Tabular foundation model adapters may only cache the training data here.

        Returns:
            Fit time in seconds.
        """
        pass

    @abstractmethod
    def predict(self, X_test) -> TimedPrediction:
        """
        Predict for a fitted model.

        Classification adapters must return class probabilities with shape
        (n_samples, n_classes). Regression adapters return predictions.

        Returns:
            Prediction values and prediction time in seconds.
        """
        pass

    def release(self) -> None:
        estimator = getattr(self, "model", None)
        if estimator is None:
            return

        close = getattr(estimator, "close", None)
        if callable(close):
            close()

        cpu = getattr(estimator, "cpu", None)
        if callable(cpu):
            cpu()

        self.model = None
        for attr in ("X_train", "y_train"):
            if hasattr(self, attr):
                setattr(self, attr, None)

    def predict_from_estimator(self, X_test) -> np.ndarray:
        if self.task_type == "classification" and hasattr(self.model, "predict_proba"):
            values = self.model.predict_proba(X_test)
        else:
            values = self.model.predict(X_test)
        return _prediction_array(values)

    def timed_prediction(self, values: Any, started_at: float) -> TimedPrediction:
        return TimedPrediction(values=_prediction_array(values), seconds=timer() - started_at)


class PreprocessedModelAdapter(ModelAdapter):
    """Adapter wrapper that applies sklearn preprocessing around a model."""

    def __init__(self, adapter: ModelAdapter, preprocess_pipeline) -> None:
        self.adapter = adapter
        self.preprocess_pipeline = preprocess_pipeline
        self.task_type = adapter.task_type
        self.kwargs = adapter.kwargs
        self.model = adapter.model

    def fit(self, X_train, y_train) -> float:
        start = timer()
        X_train_processed = self.preprocess_pipeline.fit_transform(X_train)
        self.adapter.fit(X_train_processed, y_train)
        return timer() - start

    def predict(self, X_test) -> TimedPrediction:
        start = timer()
        X_test_processed = self.preprocess_pipeline.transform(X_test)
        prediction = self.adapter.predict(X_test_processed)
        return TimedPrediction(values=prediction.values, seconds=timer() - start)

    def release(self) -> None:
        adapter = getattr(self, "adapter", None)
        if adapter is not None:
            adapter.release()
        self.adapter = None
        self.model = None
        self.preprocess_pipeline = None
