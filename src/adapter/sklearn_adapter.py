from timeit import default_timer as timer
from typing import Literal

from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

from src.config import config
from src.interfaces.model_interface import ModelAdapter, TimedPrediction
from src.schemas.base_schemas import TaskType


class LinearModelAdapter(ModelAdapter):
    def __init__(
        self,
        task_type: TaskType = "regression",
        **kwargs,
    ) -> None:
        self.task_type = task_type
        default_params = (
            {
                "random_state": config.seed,
                "penalty": "l2",
            }
            if task_type == "classification"
            else {}
        )
        self.kwargs = {**default_params, **kwargs}
        self.model = self._load_model()

    def _load_model(self):
        if self.task_type == "classification":
            return LogisticRegression(**self.kwargs)
        return LinearRegression(**self.kwargs)

    def fit(self, X_train, y_train) -> float:
        start = timer()
        self.model.fit(X_train, y_train)
        return timer() - start

    def predict(self, X_test) -> TimedPrediction:
        start = timer()
        return self.timed_prediction(self.predict_from_estimator(X_test), start)


class XGBoostAdapter(ModelAdapter):
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

    def predict(self, X_test) -> TimedPrediction:
        start = timer()
        return self.timed_prediction(self.predict_from_estimator(X_test), start)


class EBMAdapter(ModelAdapter):
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

    def predict(self, X_test) -> TimedPrediction:
        start = timer()
        return self.timed_prediction(self.predict_from_estimator(X_test), start)
