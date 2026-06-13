from abc import ABC, abstractmethod

from numpy import ndarray


class TFModelInterface(ABC):
    """
    Interface to unify usage of all tabular foundation models.

    Basic functions needed:
        fit(X_train, y_train)
        predict(X_test)
    """

    def __init__(self) -> None:
        self.model = None

    @abstractmethod
    def load_model(self, task_type="classifier", **kwargs):
        pass

    @abstractmethod
    def fit(self, X_train, y_train) -> float:
        """
        Fit model.
        For tabular foundation model this doesnt do much except the preprocessing steps for training data

        Returns:
            Training time in ms
        """
        pass

    @abstractmethod
    def predict(self, X_test) -> tuple[ndarray, float]:
        """
        Fit model.
        For tabular foundation model this is where the ICL happens

        Returns:
            Prediction probability for classification model
            Prediction for regression model

            and
            Prediction time in ms (since this is realistically our "train" time compared to classical ML)
        """
        pass
