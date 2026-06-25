from abc import ABC, abstractmethod
from timeit import default_timer as timer
from typing import Any

import numpy as np
from numpy import ndarray

from src.schemas.base_schemas import TaskType

PredictionOutput = tuple[ndarray, float]


class ModelAdapter(ABC):
    name: str
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
    def predict(self, X_test) -> PredictionOutput:
        """
        Predict for a fitted model.

        Classification adapters must return class probabilities with shape
        (n_samples, n_classes). Regression adapters return predictions.

        Returns:
            Prediction values and prediction time in seconds.
        """
        pass

    def estimator_for_training(self):
        return self.model

    def set_trained_estimator(self, estimator) -> None:
        self.model = estimator

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

    def predict_from_estimator(self, X_test) -> ndarray:
        if self.task_type == "classification" and hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(X_test))
        return np.asarray(self.model.predict(X_test))


class PreprocessedModelAdapter(ModelAdapter):
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

    def predict(self, X_test) -> PredictionOutput:
        start = timer()
        X_test_processed = self.preprocess_pipeline.transform(X_test)
        predictions, _ = self.adapter.predict(X_test_processed)
        return predictions, timer() - start

    def release(self) -> None:
        adapter = getattr(self, "adapter", None)
        if adapter is not None:
            adapter.release()
        self.adapter = None
        self.model = None
        self.preprocess_pipeline = None
