import numpy as np
import pandas as pd
import pytest

from src.interfaces.model_interface import TimedPrediction
from src.schemas.dataset_schemas import DatasetBundle, XYDataset
from src.utils.evaluation import evaluate_trained_model
from src.utils.evaluation_utils import classification_prediction_batch


class _PredictionModel:
    def __init__(self):
        self.predict_calls = 0

    def predict(self, X):
        self.predict_calls += 1
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
    retriever_set = XYDataset(X=probabilities, y=pd.Series([0, 0, 1, 1], index=[50, 51, 55, 59]))
    data = DatasetBundle(
        train_data=test_set,
        test_mimic=test_set,
        test_tudd=test_set,
        test_retriever=retriever_set,
    )

    model = _PredictionModel()
    result = evaluate_trained_model(model, "classification", data)

    assert model.predict_calls == 1
    assert result.metrics.mimic_test.accuracy == 0.0
    assert result.metrics.mimic_prediction_time == 0.0
    assert result.metrics.tudd_prediction_time == 0.0
    assert result.metrics.retriever_test is not None
    assert result.metrics.retriever_test.accuracy == 1.0
    assert result.metrics.retriever_prediction_time == 0.125
    assert result.test_predictions is not None
    assert result.test_predictions.retriever is not None
    assert result.test_predictions.mimic.test_set_id.size == 0
    assert result.test_predictions.tudd.test_set_id.size == 0
    np.testing.assert_array_equal(result.test_predictions.retriever.test_set_id, [50, 51, 55, 59])
    np.testing.assert_array_equal(result.test_predictions.retriever.y_true, [0, 0, 1, 1])
    np.testing.assert_allclose(result.test_predictions.retriever.positive_class_probability, [0.1, 0.2, 0.8, 0.9])


def test_trained_model_evaluation_keeps_standard_holdouts_without_retriever():
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
    model = _PredictionModel()

    result = evaluate_trained_model(model, "classification", data)

    assert model.predict_calls == 2
    assert result.metrics.mimic_test.accuracy == 1.0
    assert result.metrics.tudd_test.accuracy == 1.0
    assert result.metrics.retriever_test is None
    np.testing.assert_array_equal(result.test_predictions.mimic.test_set_id, [40, 41, 45, 49])
    assert result.test_predictions.retriever is None


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
