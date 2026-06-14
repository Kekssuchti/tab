from timeit import default_timer as timer
from typing import Literal

from orion_msp import OrionMSPClassifier

from src.interfaces.model_interface import TFModelInterface
from src.utils.logger import logger


class OrionMSPAdapter(TFModelInterface):
    def __init__(
        self, task_type: Literal["classification"] = "classification", **kwargs
    ) -> None:
        super().__init__()
        if task_type != "classification":
            logger.error("Got wrong task type for Orion Bix model")
            raise ValueError
        self.task_type = "classification"  # regression not supported
        self.kwargs = kwargs
        self.model = self._load_model()

    def _load_model(self):
        model = OrionMSPClassifier(**self.kwargs)
        return model

    def fit(self, X_train, y_train):
        start_time = timer()
        self.model.fit(X_train, y_train)
        return timer() - start_time

    def predict(self, X_test):
        logger.info(f"Predicting with: {self.model.get_params()}")
        start_time = timer()
        result = self.model.predict_proba(X_test)

        logger.info("OrionMSP Prediction done")
        return result, timer() - start_time
