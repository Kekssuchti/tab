from timeit import default_timer as timer

import numpy as np
from tabicl import TabICLClassifier, TabICLRegressor

from src.config import config
from src.interfaces.model_interface import ModelAdapter
from src.schemas.base_schemas import TaskType


class TabICLAdapter(ModelAdapter):
    def __init__(
        self,
        task_type: TaskType = "classification",
        **kwargs,
    ) -> None:
        super().__init__()
        self.task_type = task_type
        self.predict_batch_size = kwargs.pop("predict_batch_size", 9999999)
        if self.predict_batch_size is not None and self.predict_batch_size < 1:
            raise ValueError("predict_batch_size must be at least 1")

        # for full mimic tabicl runs OOM with more than 8 estimators
        # this is therefore the safety default
        # on experiments with less data this can be overwritten via kwargs
        n_estimators = kwargs.pop("n_estimators", 8)
        cache_type = "kv" if n_estimators <= 8 else "repr"
        kv_cache = kwargs.pop("kv_cache", cache_type)
        default_kwargs = {"random_state": config.seed, "kv_cache": kv_cache}

        self.kwargs = {**default_kwargs, **kwargs}
        self.model = self._load_model()

    def _load_model(self):
        if self.task_type == "classification":
            model = TabICLClassifier(**self.kwargs)
        else:
            model = TabICLRegressor(**self.kwargs)
        return model

    def fit(self, X_train, y_train):
        start_time = timer()
        self.model.fit(X_train, y_train)
        return timer() - start_time

    def predict(self, X_test):
        start_time = timer()
        if self.predict_batch_size is not None and len(X_test) > self.predict_batch_size:
            result = self._predict_batched(X_test)
            return np.asarray(result), timer() - start_time

        result = self._predict_single_batch(X_test)
        return np.asarray(result), timer() - start_time

    def _predict_batched(self, X_test):
        predictions = []
        for start in range(0, len(X_test), self.predict_batch_size):
            stop = start + self.predict_batch_size
            predictions.append(self._predict_single_batch(self._slice_rows(X_test, start, stop)))
        return np.concatenate(predictions, axis=0)

    def _predict_single_batch(self, X_test):
        if self.task_type == "classification":
            return self.model.predict_proba(X_test)
        return self.model.predict(X_test, output_type="mean", alphas=None)

    @staticmethod
    def _slice_rows(X, start: int, stop: int):
        if hasattr(X, "iloc"):
            return X.iloc[start:stop]
        return X[start:stop]
