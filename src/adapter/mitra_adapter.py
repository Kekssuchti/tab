from timeit import default_timer as timer
from typing import Literal

import numpy as np
from autogluon.tabular.models.mitra.sklearn_interface import (
    MitraClassifier,
    MitraRegressor,
)

from src.interfaces.model_interface import TFModelInterface
from src.utils.logger import logger


class MitraAdapter(TFModelInterface):
    def __init__(
        self,
        task_type: Literal["classification", "regression"] = "classification",
        **kwargs,
    ) -> None:
        super().__init__()
        self.task_type = task_type
        self.kwargs = kwargs
        self.model = self._load_model()

    def _load_model(self):
        defaults = {"verbose": False, "fine_tune": False}
        params = {**defaults, **self.kwargs}

        if self.task_type == "regression":
            return MitraRegressor(**params)
        else:
            return MitraClassifier(**params)

    def fit(self, X_train, y_train):
        start_time = timer()
        self.model.fit(X_train, y_train)
        return timer() - start_time

    def predict(self, X_test):
        logger.info(f"Predicting with Mitra (n_estimators={self.model.n_estimators})")
        start_time = timer()

        if isinstance(self.model, MitraClassifier):
            result = np.asarray(self.model.predict_proba(X_test))
        else:
            result = np.asarray(self.model.predict(X_test))

        logger.info("Mitra Prediction done")
        return result, timer() - start_time
