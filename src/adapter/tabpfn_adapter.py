from timeit import default_timer as timer

from tabpfn import TabPFNClassifier, TabPFNRegressor

from src.interfaces.model_interface import TFModelInterface
from src.utils.logger import logger


class TabPFN(TFModelInterface):
    def __init__(self) -> None:
        super().__init__()

    def load_model(self, task_type="classifier", **kwargs):
        if isinstance(task_type, str):
            if task_type == "classifier":
                self.model = TabPFNClassifier(**kwargs)
            elif task_type == "regression":
                self.model = TabPFNRegressor(**kwargs)
            else:
                logger.error(f"Unknown type: {task_type}")
        else:
            logger.error(f"Type has wrong type: {type(task_type)}")

    def fit(self, X_train, y_train):
        if not self.model:
            logger.info(
                "Using default model configuration, load model beforehand to adjust parameters"
            )
            self.load_model()

        start_time = timer()
        self.model.fit(X_train, y_train)
        return timer() - start_time

    def predict(self, X_test):
        logger.info(f"Predicting with: {self.model.configs_}")
        start_time = timer()
        if isinstance(self.model, TabPFNClassifier):
            result = self.model.predict_proba(X_test)
        else:
            result = self.model.predict(X_test, output_type="mean", quantiles=None)

        logger.info("Prediction done")
        return result, timer() - start_time
