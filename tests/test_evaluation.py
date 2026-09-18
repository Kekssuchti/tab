import numpy as np
import pandas as pd
import pytest

from src.interfaces.model_interface import TimedPrediction
from src.schemas.dataset_schemas import DatasetBundle, XYDataset
from src.schemas.metrics import (
    BootstrapClassificationMetrics,
    BootstrapFinalTestMetrics,
    BootstrapRegressionMetrics,
)
from src.utils.evaluation import (
    _evaluate_bootstrap_classification,
    _evaluate_bootstrap_regression,
    evaluate_trained_model_bootstrap,
    evaluate_trained_model_bootstrap_with_predictions,
)


class _PredictionModel:
    def predict(self, X):
        return TimedPrediction(values=X.to_numpy(), seconds=0.125)


def test_bootstrap_classification_uses_full_test_metrics_and_sampled_confidence_intervals():
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.7, 0.3],
            [0.4, 0.6],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.3, 0.7],
            [0.6, 0.4],
            [0.1, 0.9],
        ]
    )
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 1])

    result = _evaluate_bootstrap_classification(
        probabilities,
        y_true,
        n_bootstrap=100,
        rng=np.random.default_rng(7),
    )

    assert isinstance(result, BootstrapClassificationMetrics)
    assert result.n_bootstrap == 100
    assert result.metrics.accuracy == 0.875
    assert result.metrics.roc_auc == 1.0
    assert set(result.confidence_intervals) == {
        "roc_auc",
        "prc_auc",
        "f1",
        "accuracy",
        "sensitivity",
        "precision",
    }
    assert all(lower <= upper for lower, upper in result.confidence_intervals.values())


def test_bootstrap_regression_records_resample_count():
    result = _evaluate_bootstrap_regression(
        predictions=np.array([1.2, 1.8, 3.1, 3.9, 5.2, 5.8]),
        y_true=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        n_bootstrap=50,
        rng=np.random.default_rng(9),
    )

    assert isinstance(result, BootstrapRegressionMetrics)
    assert result.n_bootstrap == 50
    assert result.metrics.mae > 0
    assert set(result.confidence_intervals) == {"r2", "mae", "mse", "rmse"}


def test_trained_model_bootstrap_consumes_timed_predictions():
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

    result = evaluate_trained_model_bootstrap_with_predictions(
        _PredictionModel(),
        "classification",
        data,
        n_bootstrap=20,
        random_state=3,
    )

    assert isinstance(result.metrics.mimic_test, BootstrapClassificationMetrics)
    assert result.metrics.mimic_test.metrics.accuracy == 1.0
    assert result.metrics.mimic_prediction_time == 0.125
    assert result.metrics.tudd_prediction_time == 0.125
    assert result.test_predictions is not None
    np.testing.assert_array_equal(result.test_predictions.mimic.test_set_id, [40, 41, 45, 49])
    np.testing.assert_array_equal(result.test_predictions.mimic.y_true, [0, 0, 1, 1])
    np.testing.assert_allclose(result.test_predictions.mimic.positive_class_probability, [0.1, 0.2, 0.8, 0.9])


def test_regression_evaluation_does_not_emit_probability_tables():
    class RegressionPredictionModel:
        def predict(self, X):
            return TimedPrediction(values=np.array([1.1, 2.2, 2.8, 4.1]), seconds=0.05)

    test_set = XYDataset(
        X=pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]}),
        y=pd.Series([1.0, 2.0, 3.0, 4.0], index=[40, 41, 45, 49]),
    )
    data = DatasetBundle(train_data=test_set, test_mimic=test_set, test_tudd=test_set)

    result = evaluate_trained_model_bootstrap_with_predictions(
        RegressionPredictionModel(),
        "regression",
        data,
        n_bootstrap=20,
        random_state=3,
    )

    assert isinstance(result.metrics.mimic_test, BootstrapRegressionMetrics)
    assert result.test_predictions is None

    metrics_only = evaluate_trained_model_bootstrap(
        RegressionPredictionModel(),
        "regression",
        data,
        n_bootstrap=20,
        random_state=3,
    )
    assert isinstance(metrics_only, BootstrapFinalTestMetrics)


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        (np.array([[1.1, -0.1], [0.2, 0.8]]), "between 0 and 1"),
        (np.array([[0.8, 0.3], [0.2, 0.8]]), "sum to 1"),
    ],
)
def test_classification_evaluation_rejects_invalid_probabilities(probabilities, message):
    with pytest.raises(ValueError, match=message):
        _evaluate_bootstrap_classification(
            probabilities,
            np.array([0, 1]),
            n_bootstrap=5,
            rng=np.random.default_rng(1),
        )
