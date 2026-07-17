import sys

from src.config import config

sys.path.insert(0, str(config.dir_external_tabfm))

from timeit import default_timer as timer

import numpy as np
import torch

from external.tabfm.tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch
from src.interfaces.model_interface import ModelAdapter, PredictionOutput
from src.schemas.base_schemas import TaskType


class TabfmAdapter(ModelAdapter):
    def __init__(
        self,
        task_type: TaskType = "classification",
        **kwargs,
    ) -> None:
        super().__init__()
        self.task_type = task_type
        self.predict_batch_size = kwargs.pop("predict_batch_size", 99999999)

        default_params = {"random_state": config.seed, "n_estimators": 1}
        self.kwargs = {**default_params, **kwargs}
        self.model = self._load_model()

    def _load_model(self):
        model_raw = tabfm_v1_0_0_pytorch.load(
            model_type="classification",
            use_cache=True,
            device="cuda",
        )

        model = TabFMClassifier(model=model_raw, **self.kwargs)

        return model

    def fit(self, X_train, y_train):
        start_time = timer()
        self.model.fit(X_train, y_train)
        return timer() - start_time

    def predict(self, X_test) -> PredictionOutput:
        start_time = timer()
        if (
            self.predict_batch_size is not None
            and len(X_test) > self.predict_batch_size
        ):
            result = self._predict_batched(X_test)

            return result, timer() - start_time

        result = self._predict_single_batch(X_test)

        return result, timer() - start_time

    def _predict_batched(self, X_test):
        predictions = []
        for start in range(0, len(X_test), self.predict_batch_size):
            stop = start + self.predict_batch_size
            predictions.append(
                self._predict_single_batch(self._slice_rows(X_test, start, stop))
            )
        return self._concat_predictions(predictions)

    def _predict_single_batch(self, X_test):
        result = self.model.predict_proba(X_test)
        return result

    @staticmethod
    def _concat_predictions(predictions):
        if torch.is_tensor(predictions[0]):
            return torch.cat(predictions, dim=0)
        return np.concatenate(
            [np.asarray(prediction) for prediction in predictions], axis=0
        )

    @staticmethod
    def _slice_rows(X, start: int, stop: int):
        if hasattr(X, "iloc"):
            return X.iloc[start:stop]
        return X[start:stop]


if __name__ == "__main__":
    TabfmAdapter()
