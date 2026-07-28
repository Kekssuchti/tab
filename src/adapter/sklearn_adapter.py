from timeit import default_timer as timer
from typing import Literal

from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

from src.config import config
from src.interfaces.model_interface import (
    ClassificationPredictions,
    ModelAdapter,
    PredictionValues,
    TimedPrediction,
)
from src.schemas.base_schemas import TaskType


class LinearModelAdapter(ModelAdapter[ClassificationPredictions]):
    def __init__(
        self,
        task_type: TaskType = "regression",
        **kwargs,
    ) -> None:
        self.task_type = task_type
        default_params = {
            "random_state": config.seed,
            "penalty": "l2",
        }
        self.kwargs = {**default_params, **kwargs}
        self.model = self._load_model()

    def _load_model(self):
        if self.task_type == "classification":
            return LogisticRegression(**self.kwargs)
        raise NotImplementedError

    def fit(self, X_train, y_train) -> float:
        start = timer()
        self.model.fit(X_train, y_train)
        return timer() - start

    def predict(self, X_test) -> TimedPrediction[ClassificationPredictions]:
        start = timer()
        return self.timed_prediction(self.predict_from_estimator(X_test), start)


class XGBoostAdapter(ModelAdapter[PredictionValues]):
    def __init__(
        self,
        task_type: Literal["classification", "regression"] = "classification",
        **kwargs,
    ) -> None:
        self.task_type = task_type
        default_params = {"random_state": config.seed, "eval_metric": "logloss"}
        self.kwargs = {**default_params, **kwargs}
        self.model = self._load_model()

    def _load_model(self):
        if self.task_type == "classification":
            return XGBClassifier(**self.kwargs)

        return XGBRegressor(**self.kwargs)

    def fit(self, X_train, y_train) -> float:
        start = timer()
        self.model.fit(X_train, y_train)
        return timer() - start

    def predict(self, X_test) -> TimedPrediction[PredictionValues]:
        start = timer()
        return self.timed_prediction(self.predict_from_estimator(X_test), start)


class EBMAdapter(ModelAdapter[PredictionValues]):
    def __init__(
        self,
        task_type: Literal["classification", "regression"] = "classification",
        **kwargs,
    ) -> None:
        self.task_type = task_type
        default_params = {
            "random_state": config.seed,
            "interactions": 0,
        }
        self.kwargs = {**default_params, **kwargs}
        self.model = self._load_model()

    def _load_model(self):
        if self.task_type == "classification":
            return ExplainableBoostingClassifier(**self.kwargs)

        return ExplainableBoostingRegressor(**self.kwargs)

    def fit(self, X_train, y_train) -> float:
        start = timer()
        self.model.fit(X_train, y_train)
        return timer() - start

    def predict(self, X_test) -> TimedPrediction[PredictionValues]:
        start = timer()
        return self.timed_prediction(self.predict_from_estimator(X_test), start)
