from abc import ABC, abstractmethod
from dataclasses import dataclass
from timeit import default_timer as timer
from typing import Any, Generic, Literal, NewType, TypeVar, cast, overload

import numpy as np

from src.schemas.base_schemas import TaskType

ClassificationPredictions = NewType("ClassificationPredictions", np.ndarray)
RegressionPredictions = NewType("RegressionPredictions", np.ndarray)
PredictionValues = ClassificationPredictions | RegressionPredictions
PredictionT = TypeVar("PredictionT", bound=PredictionValues)


@dataclass(frozen=True)
class TimedPrediction(Generic[PredictionT]):
    values: PredictionT
    seconds: float


@overload
def prediction_values_for_task(
    task_type: Literal["classification"],
    values: Any,
) -> ClassificationPredictions: ...


@overload
def prediction_values_for_task(
    task_type: Literal["regression"],
    values: Any,
) -> RegressionPredictions: ...


@overload
def prediction_values_for_task(task_type: TaskType, values: Any) -> PredictionValues: ...


def prediction_values_for_task(task_type: TaskType, values: Any) -> PredictionValues:
    detach = getattr(values, "detach", None)
    if callable(detach):
        values = detach()

    cpu = getattr(values, "cpu", None)
    if callable(cpu):
        values = cpu()

    to_numpy = getattr(values, "numpy", None)
    if callable(to_numpy):
        values = to_numpy()

    array = np.asarray(values)
    if task_type == "classification":
        return ClassificationPredictions(array)
    return RegressionPredictions(array)


class ModelAdapter(ABC, Generic[PredictionT]):
    """Common interface for trainable tabular model adapters."""

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
    def predict(self, X_test) -> TimedPrediction[PredictionT]:
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

    def predict_from_estimator(self, X_test) -> PredictionT:
        if self.task_type == "classification" and hasattr(self.model, "predict_proba"):
            values = self.model.predict_proba(X_test)
        else:
            values = self.model.predict(X_test)
        return cast(PredictionT, prediction_values_for_task(self.task_type, values))

    def timed_prediction(self, values: Any, started_at: float) -> TimedPrediction[PredictionT]:
        normalized = prediction_values_for_task(self.task_type, values)
        return TimedPrediction(values=cast(PredictionT, normalized), seconds=timer() - started_at)


class PreprocessedModelAdapter(ModelAdapter[PredictionT], Generic[PredictionT]):
    """Adapter wrapper that applies sklearn preprocessing around a model."""

    def __init__(self, adapter: ModelAdapter[PredictionT], preprocess_pipeline) -> None:
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

    def predict(self, X_test) -> TimedPrediction[PredictionT]:
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
