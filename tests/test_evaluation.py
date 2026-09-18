import numpy as np
import pandas as pd
import pytest

from src.interfaces.model_interface import TimedPrediction
from src.schemas.dataset_schemas import DatasetBundle, XYDataset
from src.schemas.metrics import RegressionMetrics
from src.utils.evaluation import evaluate_trained_model
from src.utils.evaluation_utils import classification_prediction_batch


class _PredictionModel:
    def predict(self, X):
        return TimedPrediction(values=X.to_numpy(), seconds=0.125)


def test_trained_model_evaluation_returns_point_metrics_and_probabilities():
    probabilities = pd.DataFrame(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
        ]
    )
    test_set = XYDataset(X=probabilities, y=pd.Series([0, 0, 1, 1], index=[40, 41, 45, 49]))
    data = DatasetBundle(train_data=test_set, test_mimic=test_set, test_tudd=test_set)

    result = evaluate_trained_model(_PredictionModel(), "classification", data)

    assert result.metrics.mimic_test.accuracy == 1.0
    assert result.metrics.mimic_prediction_time == 0.125
    assert result.metrics.tudd_prediction_time == 0.125
    assert result.test_predictions is not None
    np.testing.assert_array_equal(result.test_predictions.mimic.test_set_id, [40, 41, 45, 49])
    np.testing.assert_array_equal(result.test_predictions.mimic.y_true, [0, 0, 1, 1])
    np.testing.assert_allclose(result.test_predictions.mimic.positive_class_probability, [0.1, 0.2, 0.8, 0.9])


def test_regression_evaluation_returns_only_point_metrics():
    class RegressionPredictionModel:
        def predict(self, X):
            return TimedPrediction(values=np.array([1.1, 2.2, 2.8, 4.1]), seconds=0.05)

    test_set = XYDataset(
        X=pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]}),
        y=pd.Series([1.0, 2.0, 3.0, 4.0], index=[40, 41, 45, 49]),
    )
    data = DatasetBundle(train_data=test_set, test_mimic=test_set, test_tudd=test_set)

    result = evaluate_trained_model(RegressionPredictionModel(), "regression", data)

    assert isinstance(result.metrics.mimic_test, RegressionMetrics)
    assert result.test_predictions is None


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        (np.array([[1.1, -0.1], [0.2, 0.8]]), "between 0 and 1"),
        (np.array([[0.8, 0.3], [0.2, 0.8]]), "sum to 1"),
    ],
)
def test_classification_evaluation_rejects_invalid_probabilities(probabilities, message):
    with pytest.raises(ValueError, match=message):
        classification_prediction_batch(probabilities, np.array([0, 1]))
