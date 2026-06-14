from timeit import default_timer as timer
from typing import Literal

from tabpfn import TabPFNClassifier, TabPFNRegressor

from src.interfaces.model_interface import TFModelInterface
from src.utils.logger import logger


class TabPFNAdapter(TFModelInterface):
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
            model = TabPFNClassifier(**self.kwargs)
        else:
            model = TabPFNRegressor(**self.kwargs)
        return model

    def fit(self, X_train, y_train):
        start_time = timer()
        self.model.fit(X_train, y_train)
        return timer() - start_time

    def predict(self, X_test):
        logger.info(f"Predicting with: {self.model.configs_}")
        start_time = timer()
        if isinstance(self.model, TabPFNClassifier):
            result = self.model.predict_proba(X_test)
        else:
            result = self.model.predict(X_test, output_type="mean", quantiles=None)

        logger.info("TabPFN Prediction done")
        return result, timer() - start_time
