from timeit import default_timer as timer
from typing import Literal

from orion_bix import OrionBixClassifier

from src.config import config
from src.interfaces.model_interface import ModelAdapter, TimedPrediction
from src.utils.logger import logger


class OrionBixAdapter(ModelAdapter):
    def __init__(self, task_type: Literal["classification"] = "classification", **kwargs) -> None:
        super().__init__()
        if task_type != "classification":
            logger.error("Got wrong task type for Orion Bix model")
            raise ValueError(f"task_type must be 'classification', got {task_type!r}")
        self.task_type = "classification"  # regression not supported
        default_kwargs = {
            "random_state": config.seed,
            "feat_shuffle_method": "latin",
            "use_amp": True,
        }
        self.kwargs = {**default_kwargs, **kwargs}
        self.model = self._load_model()

    def _load_model(self):
        model = OrionBixClassifier(**self.kwargs)
        return model

    def fit(self, X_train, y_train):
        start_time = timer()
        self.model.fit(X_train, y_train)
        return timer() - start_time

    def predict(self, X_test) -> TimedPrediction:
        start_time = timer()
        result = self.model.predict_proba(X_test)

        return self.timed_prediction(result, start_time)
