from timeit import default_timer as timer

from exaonetabular.classifier import EXAONETabularClassifier
from exaonetabular.regressor import EXAONETabularRegressor

from src.interfaces.model_interface import ModelAdapter, TimedPrediction, seed_kwargs
from src.schemas.base_schemas import TaskType


class EXAONEAdapter(ModelAdapter):
    def __init__(
        self,
        task_type: TaskType = "classification",
        random_state: int | None = None,
        inference_state: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.task_type = task_type
        self.random_state = random_state
        default_kwargs = {**seed_kwargs("seed", random_state), "device": "cuda"}

        self.kwargs = {**default_kwargs, **kwargs}
        self.model = self._load_model()

    def _load_model(self):
        if self.task_type == "classification":
            model = EXAONETabularClassifier.from_pretrained(**self.kwargs)
        else:
            model = EXAONETabularRegressor.from_pretrained(**self.kwargs)
        return model

    def fit(self, X_train, y_train):
        start_time = timer()
        self.model.fit(X_train, y_train)
        return timer() - start_time

    def predict(self, X_test) -> TimedPrediction:
        start_time = timer()
        result = self._predict_single_batch(X_test)
        return self.timed_prediction(result, start_time)

    def _predict_single_batch(self, X_test):
        if self.task_type == "classification":
            return self.model.predict_proba(X_test)
        return self.model.predict(X_test, output_type="mean", alphas=None)
