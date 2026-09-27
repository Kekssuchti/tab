from timeit import default_timer as timer

from sklearn.base import BaseEstimator
from sklearn.utils.validation import validate_data

from external.TabSwift.TALENT.model.lib.tabswift.classifier import TabSwiftClassifier
from external.TabSwift.TALENT.model.lib.tabswift.regressor import TabSwiftRegressor
from src.interfaces.model_interface import ModelAdapter, TimedPrediction, seed_kwargs
from src.schemas.base_schemas import TaskType


def _restore_validate_data() -> None:
    """Give scikit-learn >= 1.7 back the ``BaseEstimator._validate_data`` method.

    The vendored TabSwift estimators and their preprocessing transformers call
    ``self._validate_data(...)``. Scikit-learn deprecated that method in 1.6 and
    removed it in 1.7, where the module-level ``validate_data(estimator, ...)``
    helper took over. Drop this shim once vendored TabSwift requires 1.7 itself.
    """
    if hasattr(BaseEstimator, "_validate_data"):
        return

    def _validate_data(self, X="no_validation", y="no_validation", reset=True, **check_params):
        return validate_data(self, X, y, reset=reset, **check_params)

    BaseEstimator._validate_data = _validate_data


_restore_validate_data()


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
