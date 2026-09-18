import sys

from src.config import config

sys.path.insert(0, str(config.dir_external_limix))
import os
from copy import deepcopy
from timeit import default_timer as timer

import kditransform.kdi_transformer as _kdi_transformer
import kditransform.ksum as _kdi_ksum
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from external.limix.inference.predictor import LimiXPredictor
from src.interfaces.model_interface import ModelAdapter, TimedPrediction, seed_kwargs
from src.schemas.base_schemas import TaskType


def _use_pure_python_kdi_kernel_sum() -> None:
    """Run kditransform's kernel sum without numba.

    The ROCm SDK wheels link libLLVM.so.23 into every compute library, and
    importing torch loads it into the global symbol scope. llvmlite bundles
    LLVM 20 instead and its internal calls are preemptible, so the first numba
    JIT compilation after torch is imported binds against LLVM 23 and corrupts
    the heap: the process aborts with "free(): invalid pointer" and no Python
    traceback. LimiX preprocesses with kditransform (the KDIX pipelines), so
    limix-2 died inside predict().

    Swapping the dispatcher for its Python implementation is numerically
    identical for KDITransformer and costs roughly a factor of two on fit.
    """
    for module in (_kdi_transformer, _kdi_ksum):
        kernel_sum = getattr(module, "ksum_numba", None)
        py_func = getattr(kernel_sum, "py_func", None)
        if py_func is not None:
            module.ksum_numba = py_func


_use_pure_python_kdi_kernel_sum()


class LimixV2Adapter(ModelAdapter):
    def __init__(
        self,
        task_type: TaskType = "classification",
        random_state: int | None = None,
        inference_state: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # from docu
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        # LimiX defaults its inference cache to an upstream cluster path; keep it in the project cache.
        os.environ.setdefault("LIMIX_CACHE_DIR", str(config.dir_cache / "limix2_inference"))

        self.task_type = task_type
        self.model_path = str(config.dir_cache / "LimiX-2.ckpt")

        self.random_state = random_state
        self.inference_state = inference_state
        self.inference_config = kwargs.pop("inference_config", None)
        self.n_estimators = kwargs.pop("n_estimators", None)
        self.retrieval_config_overrides = kwargs.pop("retrieval_config_overrides", None)
        default_params = {
            # "mask_prediction": False,
            **seed_kwargs("seed", random_state),
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
                repo_id="stable-ai/LimiX-2",
                filename="LimiX-2.ckpt",
                local_dir=str(config.dir_cache),
            )

        config_name = {
            "classification": "cls_default_noretrieval_v2.json",
            "regression": "reg_default_noretrieval_v2.json",
        }[self.task_type.lower()]

        config_path = str(config.dir_external_limix / "config" / config_name)

        model = LimiXPredictor(
            device=device,
            model_path=self.model_path,
            inference_config=config_path,  # self.inference_config or
            **self.kwargs,
        )

        return model

    def _configure_inference(self):
        if self.n_estimators is None and self.retrieval_config_overrides is None and self.inference_state is None:
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

        # LimiX derives the per-pipeline predict-time seeds from this value, so
        # the inference state is the seed for randomness drawn while predicting.
        self.model.set_inference_config(inference_config, seed=self.inference_state)

    def fit(self, X_train, y_train):
        # this model does not have a fit() function
        # most models still provide it even tho its not needed to keep sklearns known interfaces in takt
        # or put some data preprocessing into fit() but LimiX does not
        # this model does not, thus we only return 0.0 for the training time (none since not done)
        self.X_train = X_train
        self.y_train = y_train
        return 0.0

    def predict(self, X_test) -> TimedPrediction:
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
