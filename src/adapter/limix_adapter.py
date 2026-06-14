import sys

from src.config import config

sys.path.insert(0, str(config.external_limix_dir))


import os
from timeit import default_timer as timer
from typing import Literal

import torch
from huggingface_hub import hf_hub_download

from external.limix.inference.predictor import LimiXPredictor
from src.interfaces.model_interface import TFModelInterface
from src.utils.logger import logger


class LimixAdapter(TFModelInterface):
    def __init__(
        self, task_type="classification", size: Literal["2M", "16M"] = "2M", **kwargs
    ) -> None:
        super().__init__()
        self.task_type = task_type
        if size not in ("2M", "16M"):
            raise ValueError(f"size must be '2M' or '16M', got '{size}'")
        self.size = size  # only "2M" and "16M" are valid
        self.kwargs = kwargs
        self.model_path = str(config.cache_dir / f"LimiX-{self.size}.ckpt")
        self.model = self._load_model()

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

        config_path = str(config.external_limix_dir / "config" / config_name)

        model = LimiXPredictor(
            device=device,
            model_path=self.model_path,
            inference_config=config_path,
            **self.kwargs,
        )

        return model

    def fit(self, X_train, y_train):
        # this model does not have a fit() function
        # most models still provide it even tho its not needed to keep sklearns known interfaces in takt
        # or put some data preprocessing into fit() but LimiX does not
        # this model does not, thus we only return 0.0 for the training time (none since not done)
        self.X_train = X_train
        self.y_train = y_train
        return 0.0

    def predict(self, X_test):
        logger.info(f"Predicting with: {self.model.inference_config}")
        start_time = timer()
        result = self.model.predict(
            self.X_train, self.y_train, X_test, task_type=self.task_type.capitalize()
        )

        logger.info(f"LimiX-{self.size} Prediction done")
        return result, timer() - start_time


if __name__ == "__main__":
    LimixAdapter()
