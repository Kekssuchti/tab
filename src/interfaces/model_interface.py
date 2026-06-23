from abc import ABC, abstractmethod
from timeit import default_timer as timer
from typing import Any, Literal

import numpy as np
from numpy import ndarray

from src.schemas.base_schemas import TaskType


class ModelAdapter(ABC):
    name: str
    task_type: TaskType
    kwargs: dict
    model: Any

    @abstractmethod
    def fit(self, X_train, y_train) -> float:
        """
        Fit model.
        For tabular foundation model this doesnt do much except the preprocessing steps for training data

        Returns:
            Training time in ms
        """
        pass

    @abstractmethod
    def predict(self, X_test) -> tuple[ndarray, float]:
        """
        Fit model.
        For tabular foundation model this is where the ICL happens

        Returns:
            Prediction probability for classification model
            Prediction for regression model

            and
            Prediction time in ms (since this is realistically our "train" time compared to classical ML)
        """
        pass

    def estimator_for_training(self):
        return self.model

    def set_trained_estimator(self, estimator) -> None:
        self.model = estimator

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

    def predict(self, X_test) -> tuple[ndarray, float]:
        start = timer()
        X_test_processed = self.preprocess_pipeline.transform(X_test)
        predictions, _ = self.adapter.predict(X_test_processed)
        return predictions, timer() - start
