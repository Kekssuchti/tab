import pytest

from src.schemas.metrics import ClassificationMetrics, RegressionMetrics, calculate_metric_diff
from tests.factories import (
    bootstrap_classification_metrics,
    bootstrap_regression_metrics,
    classification_metrics,
)


def _regression(*, r2: float = 0.9, mae: float = 0.1, mse: float = 0.2, rmse: float = 0.3) -> RegressionMetrics:
    return RegressionMetrics(r2=r2, mae=mae, mse=mse, rmse=rmse)


def test_calculate_metric_diff_subtracts_classification_metrics():
    result = calculate_metric_diff(classification_metrics(0.9), classification_metrics(0.4))

    assert result.roc_auc == pytest.approx(0.5)
    assert result.prc_auc == pytest.approx(0.5)
    assert result.f1 == pytest.approx(0.5)
    assert result.accuracy == pytest.approx(0.5)
    assert result.sensitivity == pytest.approx(0.5)
    assert result.precision == pytest.approx(0.5)
    assert result.n_classes == 2
    assert result.confusion_matrix is None


def test_calculate_metric_diff_subtracts_regression_metrics():
    result = calculate_metric_diff(
        _regression(r2=0.9, mae=0.1, mse=0.2, rmse=0.3),
        _regression(r2=0.3, mae=0.05, mse=0.1, rmse=0.15),
    )

    assert result.r2 == pytest.approx(0.6)
    assert result.mae == pytest.approx(0.05)
    assert result.mse == pytest.approx(0.1)
    assert result.rmse == pytest.approx(0.15)


def test_calculate_metric_diff_unwraps_bootstrap_classification():
    result = calculate_metric_diff(
        bootstrap_classification_metrics(0.8),
        bootstrap_classification_metrics(0.2),
    )

    assert isinstance(result, ClassificationMetrics)
    assert result.accuracy == pytest.approx(0.6)


def test_calculate_metric_diff_unwraps_bootstrap_regression():
    result = calculate_metric_diff(
        bootstrap_regression_metrics(_regression(r2=0.9)),
        bootstrap_regression_metrics(_regression(r2=0.3)),
    )

    assert isinstance(result, RegressionMetrics)
    assert result.r2 == pytest.approx(0.6)


def test_calculate_metric_diff_rejects_mismatched_metric_types():
    with pytest.raises(ValueError, match="same type"):
        calculate_metric_diff(classification_metrics(), _regression())

    with pytest.raises(ValueError, match="same type"):
        calculate_metric_diff(bootstrap_classification_metrics(0.8), classification_metrics(0.2))


def test_calculate_metric_diff_reports_none_when_auc_is_unavailable():
    mimic = classification_metrics(0.9)
    mimic.roc_auc = None
    mimic.prc_auc = None

    result = calculate_metric_diff(mimic, classification_metrics(0.4))

    assert result.roc_auc is None
    assert result.prc_auc is None
    assert result.accuracy == pytest.approx(0.5)
