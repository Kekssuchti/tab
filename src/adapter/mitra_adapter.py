from timeit import default_timer as timer

from autogluon.tabular.models.mitra.sklearn_interface import (
    MitraClassifier,
    MitraRegressor,
)

from src.config import config
from src.interfaces.model_interface import ModelAdapter, PredictionValues, TimedPrediction
from src.schemas.base_schemas import TaskType


class MitraAdapter(ModelAdapter[PredictionValues]):
    def __init__(
        self,
        task_type: TaskType = "classification",
        **kwargs,
    ) -> None:
        super().__init__()
        self.task_type = task_type
        default_params = {
            "device": "cuda",
            "fine_tune": False,
            "fine_tune_steps": 0,
            "seed": config.seed,
            "n_estimators": 1,
        }
        self.kwargs = {**default_params, **kwargs}
        self.model = self._load_model()

    def _load_model(self):
        if self.task_type == "regression":
            return MitraRegressor(**self.kwargs)
        else:
            return MitraClassifier(**self.kwargs)

    def fit(self, X_train, y_train):
        start_time = timer()
        self.model.fit(X_train, y_train)
        return timer() - start_time

    def predict(self, X_test) -> TimedPrediction[PredictionValues]:
        start_time = timer()
        result = self.predict_from_estimator(X_test)

        return self.timed_prediction(result, start_time)
