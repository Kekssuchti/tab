from timeit import default_timer as timer

from external.TabSwift.TALENT.model.lib.tabswift.classifier import TabSwiftClassifier
from external.TabSwift.TALENT.model.lib.tabswift.regressor import TabSwiftRegressor
from src.interfaces.model_interface import ModelAdapter, TimedPrediction, seed_kwargs
from src.schemas.base_schemas import TaskType


class TabSwiftAdapter(ModelAdapter):
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
        default_params = {
            "model_path": "swift.ckpt",
            **seed_kwargs("random_state", random_state),
        }
        self.kwargs = {**default_params, **kwargs}
        self.model = self._load_model()

    def _load_model(self):
        if self.task_type == "regression":
            return TabSwiftRegressor(**self.kwargs)
        else:
            return TabSwiftClassifier(**self.kwargs)

    def fit(self, X_train, y_train):
        start_time = timer()
        self.model.fit(X_train, y_train)
        return timer() - start_time

    def predict(self, X_test) -> TimedPrediction:
        start_time = timer()
        result = self.predict_from_estimator(X_test)

        return self.timed_prediction(result, start_time)
