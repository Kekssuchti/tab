from timeit import default_timer as timer
from typing import Literal

import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

from src.interfaces.model_interface import TFModelInterface
from src.utils.logger import logger


class MitraAdapter(TFModelInterface):
    def __init__(
        self,
        task_type: Literal["classification", "regression"] = "classification",
        **kwargs,
    ) -> None:
        super().__init__()
        self.task_type = task_type
        self.kwargs = kwargs
        self.model = self._load_model()

    def _load_model(self):
        if self.task_type == "regression":
            problem_type = self.task_type
        else:
            # Note: right now we probably only have binary classification so
            # it is not viable for multiclass can later be extended if needed
            problem_type = "binary"

        model = TabularPredictor(
            label="target", problem_type=problem_type, path="cache/mitra", verbosity=0
        )

        return model

    def fit(self, X_train, y_train):
        # combine them again for tab dataset
        df = pd.DataFrame(X_train)
        df["target"] = y_train
        data = TabularDataset(df)

        hyperparameters_default = {
            "fine_tune": False,
        }
        hyperparameters_combined = {**hyperparameters_default, **self.kwargs}

        start_time = timer()
        self.model.fit(data, hyperparameters={"MITRA": hyperparameters_combined})
        return timer() - start_time

    def predict(self, X_test):
        logger.info(f"Predicting with: {self.model.model_info}")
        start_time = timer()

        X_test = TabularDataset(pd.DataFrame(X_test))
        result = np.asarray(self.model.predict_proba(X_test))

        logger.info("Mitra Prediction done")
        return result, timer() - start_time
