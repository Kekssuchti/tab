from timeit import default_timer as timer

from orion_bix import OrionBixClassifier

from src.interfaces.model_interface import TFModelInterface
from src.utils.logger import logger


class TabICLAdapter(TFModelInterface):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.task_type = "classification"  # regression not supported
        self.kwargs = kwargs
        self.model = self._load_model()

    def _load_model(self):
        model = OrionBixClassifier(**self.kwargs)
        return model

    def fit(self, X_train, y_train):
        start_time = timer()
        self.model.fit(X_train, y_train)
        return timer() - start_time

    def predict(self, X_test):
        logger.info(f"Predicting with: {self.model.inference_config}")
        start_time = timer()
        result = self.model.predict_proba(X_test)

        logger.info("TabICL Prediction done")
        return result, timer() - start_time
