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


def seed_kwargs(seed_param: str | None, random_state: int | None) -> dict[str, Any]:
    """Return the wrapped estimator's seed keyword, or nothing when unseeded.

    Adapters merge the result into their default parameters, so a `None` seed
    omits the keyword entirely instead of forcing the estimator into an
    unseeded explicit state.
    """
    if seed_param is None or random_state is None:
        return {}
    return {seed_param: random_state}


class ModelAdapter(ABC):
    """Common interface for trainable tabular model adapters.

    Every adapter accepts the two pipeline seeds under the canonical keyword
    names ``random_state`` and ``inference_state`` and maps them to whatever
    keyword its wrapped estimator expects. Wrapped estimators disagree here
    (scikit-learn, XGBoost, EBM, TabPFN, TabICL and TabSwift use
    ``random_state``, Mitra, EXAONE and LimiX use ``seed``), so callers such as
    the trainer never need to know the local name.

    ---
    Attributes:
        random_state: int or None
            Seed for the estimator the adapter constructs. It is forwarded to the
            wrapped estimator, so it governs the randomness of fitting.

        inference_state: int or None
            Seed for randomness drawn while predicting. Adapters whose library
            exposes a predict-time seed apply it; the rest keep it for
            provenance, because their predictions only depend on the
            construction seed.
    """

    task_type: TaskType
    kwargs: dict
    model: Any
    random_state: int | None
    inference_state: int | None

    @abstractmethod
    def fit(self, X_train, y_train) -> float:
        """
        Fit the model.

        Tabular foundation model adapters may only cache the training data here.

        Returns:
            Fit time in seconds.
        """

    @abstractmethod
    def predict(self, X_test) -> TimedPrediction:
        """
        Predict for a fitted model.

        Classification adapters must return class probabilities with shape
        (n_samples, n_classes), ordered by encoded class label. For binary
        classification, column 1 is therefore the probability of class 1.
        Regression adapters return predictions.

        Returns:
            Prediction values and prediction time in seconds.
        """

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
        self.random_state = getattr(adapter, "random_state", None)
        self.inference_state = getattr(adapter, "inference_state", None)

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


class LogTargetModelAdapter(ModelAdapter):
    """Train a regression adapter in log space and return predictions in original units."""

    def __init__(self, adapter: ModelAdapter) -> None:
        if adapter.task_type != "regression":
            raise ValueError("Log target transformation is only supported for regression")
        self.adapter = adapter
        self.task_type = adapter.task_type
        self.kwargs = adapter.kwargs
        self.model = adapter.model
        self.random_state = getattr(adapter, "random_state", None)
        self.inference_state = getattr(adapter, "inference_state", None)

    def fit(self, X_train, y_train) -> float:
        targets = np.asarray(y_train, dtype=float)
        if not np.all(np.isfinite(targets)) or np.any(targets <= 0):
            raise ValueError("Log target transformation requires finite, positive targets")
        return self.adapter.fit(X_train, np.log(targets))

    def predict(self, X_test) -> TimedPrediction:
        start = timer()
        prediction = self.adapter.predict(X_test)
        try:
            with np.errstate(over="raise", invalid="raise"):
                values = np.exp(np.asarray(prediction.values, dtype=float))
        except FloatingPointError as exc:
            raise ValueError("Inverse log target transformation produced non-finite predictions") from exc
        if not np.all(np.isfinite(values)):
            raise ValueError("Inverse log target transformation produced non-finite predictions")
        return TimedPrediction(values=values, seconds=timer() - start)

    def release(self) -> None:
        adapter = getattr(self, "adapter", None)
        if adapter is not None:
            adapter.release()
        self.adapter = None
        self.model = None
