from timeit import default_timer as timer

import sdm
import torch

from src.interfaces.model_interface import ModelAdapter, TimedPrediction, as_feature_array
from src.schemas.base_schemas import TaskType


class KumoTabularAdapter(ModelAdapter):
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
        self.inference_state = inference_state

        default_kwargs = {
            "device": "cuda:0",
            "size": "small",
            "n_estimators": 8,
        }

        self.kwargs = {**default_kwargs, **kwargs}
        self.n_estimators = self.kwargs.pop("n_estimators")
        self.model = self._load_model()

    def _load_model(self):
        if self.task_type == "classification":
            model = sdm.models.KumoTabular(task="classification", **self.kwargs)
        else:
            model = sdm.models.KumoTabular(task="regression", **self.kwargs)
        return model

    def fit(self, X_train, y_train):
        start_time = timer()
        features = self._feature_tensor(X_train)
        # SDM infers the target's semantic type from its tensor dtype.
        dtype = torch.int64 if self.task_type == "classification" else torch.float32
        targets = torch.as_tensor(as_feature_array(y_train), dtype=dtype, device=features.device).reshape(-1, 1)
        generator = None
        if self.random_state is not None:
            generator = torch.Generator(device=features.device).manual_seed(self.random_state)
        self.model.fit(features, targets, num_estimators=self.n_estimators, generator=generator)
        return timer() - start_time

    def predict(self, X_test) -> TimedPrediction:
        start_time = timer()
        result = self._predict_single_batch(X_test)
        return self.timed_prediction(result, start_time)

    def _predict_single_batch(self, X_test):
        result = self.model.predict(self._feature_tensor(X_test)).numerical
        if self.task_type == "classification":
            # Kumo's default recipe already applies softmax and averages estimators.
            return result
        # Kumo returns q001..q999; their average approximates the predictive mean.
        return result.mean(dim=-1)

    def _feature_tensor(self, X):
        return torch.as_tensor(as_feature_array(X), dtype=torch.float32, device=self.kwargs["device"])

    def release(self) -> None:
        if self.model is not None:
            self.model.clear()
        super().release()
