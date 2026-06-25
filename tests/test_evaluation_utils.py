import numpy as np
import pytest

from src.utils.evaluation_utils import evaluate_classification_predictions


def test_classification_evaluation_requires_probability_matrix():
    with pytest.raises(ValueError, match="2D class-probability array"):
        evaluate_classification_predictions(np.array([0, 1, 1]), np.array([0, 1, 1]))


def test_classification_evaluation_validates_prediction_count_and_values():
    with pytest.raises(ValueError, match="row count"):
        evaluate_classification_predictions(
            np.array([[0.7, 0.3], [0.1, 0.9]]),
            np.array([0, 1, 1]),
        )

    with pytest.raises(ValueError, match="finite"):
        evaluate_classification_predictions(
            np.array([[0.7, 0.3], [np.nan, 0.9]]),
            np.array([0, 1]),
        )
