from timeit import default_timer as timer
from typing import Literal

import numpy as np
from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

from src.interfaces.model_interface import ModelAdapter
from src.schemas.base_schemas import TaskType


class LinearModelAdapter(ModelAdapter):
    def __init__(
        self,
        task_type: TaskType = "regression",
        **kwargs,
    ) -> None:
        self.task_type = task_type
        self.kwargs = kwargs
        self.model = self._load_model()

    def _load_model(self):
        if self.task_type == "classification":
            params = {"max_iter": 1000, **self.kwargs}
            return LogisticRegression(**params)
        return LinearRegression(**self.kwargs)

    def fit(self, X_train, y_train) -> float:
        start = timer()
        self.model.fit(X_train, y_train)
        return timer() - start

    def predict(self, X_test):
        start = timer()
        return self.predict_from_estimator(X_test), timer() - start


class XGBoostAdapter(ModelAdapter):
    def __init__(
        self,
        task_type: Literal["classification", "regression"] = "classification",
        **kwargs,
    ) -> None:
        self.task_type = task_type
        self.kwargs = kwargs
        self.model = self._load_model()

    def _load_model(self):
        if self.task_type == "classification":
            params = {"eval_metric": "logloss", **self.kwargs}
            return XGBClassifier(**params)

        return XGBRegressor(**self.kwargs)

    def fit(self, X_train, y_train) -> float:
        start = timer()
        self.model.fit(X_train, y_train)
        return timer() - start

    def predict(self, X_test):
        start = timer()
        return self.predict_from_estimator(X_test), timer() - start


class EBMAdapter(ModelAdapter):
    def __init__(
        self,
        task_type: Literal["classification", "regression"] = "classification",
        **kwargs,
    ) -> None:
        self.task_type = task_type
        self.kwargs = kwargs
        self.model = self._load_model()

    def _load_model(self):
        if self.task_type == "classification":
            return ExplainableBoostingClassifier(**self.kwargs)

        return ExplainableBoostingRegressor(**self.kwargs)

    def fit(self, X_train, y_train) -> float:
        start = timer()
        self.model.fit(X_train, y_train)
        return timer() - start

    def predict(self, X_test):
        start = timer()
        return np.asarray(self.predict_from_estimator(X_test)), timer() - start
