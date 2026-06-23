from timeit import default_timer as timer

import numpy as np
from tabpfn import TabPFNClassifier, TabPFNRegressor

from src.interfaces.model_interface import ModelAdapter
from src.schemas.base_schemas import TaskType
from src.utils.logger import logger


class TabPFNAdapter(ModelAdapter):
    def __init__(
        self,
        task_type: TaskType = "classification",
        **kwargs,
    ) -> None:
        super().__init__()
        self.task_type = task_type
        self.kwargs = kwargs
        self.model = self._load_model()

    def _load_model(self):
        if self.task_type == "classification":
            model = TabPFNClassifier(**self.kwargs)
        else:
            model = TabPFNRegressor(**self.kwargs)
        return model

    def fit(self, X_train, y_train):
        start_time = timer()
        self.model.fit(X_train, y_train)
        return timer() - start_time

    def predict(self, X_test):
        logger.info("Predicting with TabPFN")
        start_time = timer()
        if self.task_type == "classification":
            result = self.model.predict_proba(X_test)
        else:
            result = self.model.predict(X_test, output_type="mean", quantiles=None)

        logger.info("TabPFN Prediction done")
        return np.asarray(result), timer() - start_time
