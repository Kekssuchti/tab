from timeit import default_timer as timer
from typing import Literal

import numpy as np
from tabicl import TabICLClassifier, TabICLRegressor

from src.interfaces.model_interface import ModelAdapter
from src.utils.logger import logger


class TabICLAdapter(ModelAdapter):
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
        logger.info("Predicting with TabICL")
        start_time = timer()
        if self.task_type == "classification":
            result = np.asarray(self.model.predict_proba(X_test))
        else:
            result = self.model.predict(X_test, output_type="mean", alphas=None)
            result = np.asarray(result)

        logger.info("TabICL Prediction done")
        return result, timer() - start_time
