import sys

from src.config import config

sys.path.insert(0, str(config.dir_external_limix))
import os
from copy import deepcopy
from timeit import default_timer as timer
from typing import Literal

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from external.limix.inference.predictor import LimiXPredictor
from src.interfaces.model_interface import ModelAdapter, PredictionValues, TimedPrediction
from src.schemas.base_schemas import TaskType


class LimixAdapter(ModelAdapter[PredictionValues]):
    def __init__(
        self,
        task_type: TaskType = "classification",
        size: Literal["2M", "16M"] = "2M",
        **kwargs,
    ) -> None:
        super().__init__()
        self.task_type = task_type
        if size not in ("2M", "16M"):
            raise ValueError(f"size must be '2M' or '16M', got '{size}'")
        self.size = size  # only "2M" and "16M" are valid
        self.model_path = str(config.dir_cache / f"LimiX-{self.size}.ckpt")

        seed = kwargs.pop("random_state", config.seed)
        self.inference_config = kwargs.pop("inference_config", None)
        self.n_estimators = kwargs.pop("n_estimators", None)
        self.retrieval_config_overrides = kwargs.pop("retrieval_config_overrides", None)
        default_params = {
            "seed": seed,
            "mask_prediction": False,
        }
        self.predict_batch_size = kwargs.pop("predict_batch_size", 99999999)
        if self.predict_batch_size is not None and self.predict_batch_size < 1:
            raise ValueError("predict_batch_size must be at least 1")
        self.kwargs = {**default_params, **kwargs}
        self.model = self._load_model()
        self._configure_inference()

    def _load_model(self):
        device = torch.device("cuda")

        if not os.path.exists(self.model_path):
            hf_hub_download(
                repo_id=f"stable-ai/LimiX-{self.size}",
                filename=f"LimiX-{self.size}.ckpt",
                local_dir="./cache",
            )

        # TODO: find out what the diff in retrieval vs noretrieval is here.
        # All sized models (2m / 16m) only have retrieval json configs.
        config_name = {
            "classification": f"cls_default_{self.size}_retrieval.json",
            "regression": f"reg_default_{self.size}_retrieval.json",
        }[self.task_type.lower()]

        config_path = str(config.dir_external_limix / "config" / config_name)

        model = LimiXPredictor(
            device=device,
            model_path=self.model_path,
            inference_config=self.inference_config or config_path,
            **self.kwargs,
        )

        return model

    def _configure_inference(self):
        if self.n_estimators is None and self.retrieval_config_overrides is None:
            return

        inference_config = deepcopy(self.model.inference_config)
        if self.n_estimators is not None:
            if self.n_estimators < 1:
                raise ValueError("n_estimators must be at least 1")
            if self.n_estimators > len(inference_config):
                raise ValueError(
                    f"n_estimators={self.n_estimators} exceeds available "
                    f"LimiX inference pipelines ({len(inference_config)})"
                )
            inference_config = inference_config[: self.n_estimators]

        if self.retrieval_config_overrides is not None:
            for config_item in inference_config:
                config_item["retrieval_config"].update(self.retrieval_config_overrides)

        self.model.set_inference_config(inference_config)

    def fit(self, X_train, y_train):
        # this model does not have a fit() function
        # most models still provide it even tho its not needed to keep sklearns known interfaces in takt
        # or put some data preprocessing into fit() but LimiX does not
        # this model does not, thus we only return 0.0 for the training time (none since not done)
        self.X_train = X_train
        self.y_train = y_train
        return 0.0

    def predict(self, X_test) -> TimedPrediction[PredictionValues]:
        start_time = timer()
        if self.predict_batch_size is not None and len(X_test) > self.predict_batch_size:
            result = self._predict_batched(X_test)
            return self.timed_prediction(result, start_time)

        result = self._predict_single_batch(X_test)

        return self.timed_prediction(result, start_time)

    def _predict_batched(self, X_test):
        predictions = []
        for start in range(0, len(X_test), self.predict_batch_size):
            stop = start + self.predict_batch_size
            predictions.append(self._predict_single_batch(self._slice_rows(X_test, start, stop)))
        return self._concat_predictions(predictions)

    def _predict_single_batch(self, X_test):
        result = self.model.predict(self.X_train, self.y_train, X_test, task_type=self.task_type.capitalize())
        return result

    @staticmethod
    def _concat_predictions(predictions):
        if torch.is_tensor(predictions[0]):
            return torch.cat(predictions, dim=0)
        return np.concatenate([np.asarray(prediction) for prediction in predictions], axis=0)

    @staticmethod
    def _slice_rows(X, start: int, stop: int):
        if hasattr(X, "iloc"):
            return X.iloc[start:stop]
        return X[start:stop]


if __name__ == "__main__":
    LimixAdapter()
