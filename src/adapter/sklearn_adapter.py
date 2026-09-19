from timeit import default_timer as timer
from typing import Literal

from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

from src.interfaces.model_interface import ModelAdapter, TimedPrediction, seed_kwargs
from src.schemas.base_schemas import TaskType


class LinearModelAdapter(ModelAdapter):
    def __init__(
        self,
        task_type: TaskType = "regression",
        random_state: int | None = None,
        inference_state: int | None = None,
        **kwargs,
    ) -> None:
        self.task_type = task_type
        self.random_state = random_state
        self.inference_state = inference_state
        # Plain LinearRegression accepts no seed at all.
        seed_param = "random_state" if task_type == "classification" else None
        default_params = {"penalty": "l2"} if task_type == "classification" else {}
        self.kwargs = {**seed_kwargs(seed_param, random_state), **default_params, **kwargs}
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
        random_state: int | None = None,
        inference_state: int | None = None,
        **kwargs,
    ) -> None:
        self.task_type = task_type
        self.random_state = random_state
        self.inference_state = inference_state
        default_params = {"eval_metric": "logloss", "n_jobs": 6}
        self.kwargs = {**seed_kwargs("random_state", random_state), **default_params, **kwargs}
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
        random_state: int | None = None,
        inference_state: int | None = None,
        **kwargs,
    ) -> None:
        self.task_type = task_type
        self.random_state = random_state
        self.inference_state = inference_state
        default_params = {"interactions": 0, "n_jobs": 6}
        self.kwargs = {**seed_kwargs("random_state", random_state), **default_params, **kwargs}
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
