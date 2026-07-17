from copy import deepcopy
from timeit import default_timer as timer

import numpy as np
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.classifier import ModelVersion

from src.config import config
from src.interfaces.model_interface import ModelAdapter
from src.schemas.base_schemas import TaskType


class TabPFNAdapter(ModelAdapter):
    def __init__(
        self,
        task_type: TaskType = "classification",
        version: ModelVersion = ModelVersion.V3,
        **kwargs,
    ) -> None:
        super().__init__()
        self.task_type = task_type
        self.version = version
        self.predict_batch_size = kwargs.pop(
            "predict_batch_size", 2048
        )  # default no batching
        default_kwargs = {
            "fit_mode": "fit_with_cache",
            "random_state": config.seed,
        }
        # override default kwargs with user-provided kwargs
        self.kwargs = self._normalize_kwargs({**default_kwargs, **kwargs})
        self.model = self._load_model()

    @staticmethod
    def _normalize_kwargs(kwargs):
        kwargs = dict(kwargs)
        inference_config = kwargs.get("inference_config")
        if not isinstance(inference_config, dict):
            return kwargs

        inference_config = deepcopy(inference_config)
        transforms = inference_config.get("PREPROCESS_TRANSFORMS")
        if isinstance(transforms, dict):
            inference_config["PREPROCESS_TRANSFORMS"] = [transforms]
        kwargs["inference_config"] = inference_config
        return kwargs

    def _load_model(self):
        if self.task_type == "classification":
            model = TabPFNClassifier.create_default_for_version(
                self.version, **self.kwargs
            )
        else:
            model = TabPFNRegressor.create_default_for_version(
                self.version, **self.kwargs
            )
        return model

    def fit(self, X_train, y_train):
        start_time = timer()
        self.model.fit(X_train, y_train)
        return timer() - start_time

    def predict(self, X_test):
        start_time = timer()
        if len(X_test) > self.predict_batch_size:
            result = self._predict_batched(X_test)
            return np.asarray(result), timer() - start_time

        result = self._predict_single_batch(X_test)
        return np.asarray(result), timer() - start_time

    def _predict_batched(self, X_test):
        predictions = []
        for start in range(0, len(X_test), self.predict_batch_size):
            stop = start + self.predict_batch_size
            predictions.append(
                self._predict_single_batch(self._slice_rows(X_test, start, stop))
            )
        return np.concatenate(predictions, axis=0)

    def _predict_single_batch(self, X_test):
        if self.task_type == "classification":
            return self.model.predict_proba(X_test)
        return self.model.predict(X_test, output_type="mean", quantiles=None)

    @staticmethod
    def _slice_rows(X, start: int, stop: int):
        if hasattr(X, "iloc"):
            return X.iloc[start:stop]
        return X[start:stop]
