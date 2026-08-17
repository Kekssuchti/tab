from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, cast, overload

import numpy as np


@dataclass
class ClassificationMetrics:
    """
    Classification metrics for one prediction set.

    ---
    Attributes:
        roc_auc: float or None
            ROC AUC, or None when unavailable.

        prc_auc: float or None
            Precision-recall AUC, or None when unavailable.

        f1: float
            F1 score.

        accuracy: float
            Accuracy score.

        sensitivity: float
            True positive rate.

        precision: float
            Positive predictive value.

        confusion_matrix: numpy.ndarray
            Confusion matrix for predicted labels.

        n_classes: int
            Number of classes observed by the evaluator.
    """

    roc_auc: float | None
    prc_auc: float | None
    f1: float
    accuracy: float
    sensitivity: float
    precision: float
    n_classes: int
    confusion_matrix: np.ndarray | None

    @property
    def scores(self) -> dict[str, float]:
        scores = {
            "f1": self.f1,
            "accuracy": self.accuracy,
            "sensitivity": self.sensitivity,
            "precision": self.precision,
        }
        if self.roc_auc is not None:
            scores["roc_auc"] = self.roc_auc
        if self.prc_auc is not None:
            scores["prc_auc"] = self.prc_auc
        return scores


@dataclass
class RegressionMetrics:
    """
    Regression metrics for one prediction set.

    ---
    Attributes:
        r2: float
            Coefficient of determination.

        mae: float
            Mean absolute error.

        mse: float
            Mean squared error.
    """

    r2: float
    mae: float
    mse: float
    rmse: float

    @property
    def scores(self) -> dict[str, float]:
        return {
            "r2": self.r2,
            "mae": self.mae,
            "mse": self.mse,
            "rmse": self.rmse,
        }


class ConfidenceMetrics(Protocol):
    @property
    def metrics(self) -> ClassificationMetrics | RegressionMetrics: ...

    @property
    def scores(self) -> dict[str, float]: ...

    @property
    def confidence_intervals(self) -> dict[str, tuple[float, float]]: ...


MetricT = TypeVar("MetricT", ClassificationMetrics, RegressionMetrics)
ConfidenceMetricT = TypeVar("ConfidenceMetricT", bound=ConfidenceMetrics)


@dataclass(frozen=True)
class FinalTestMetrics(Generic[MetricT]):
    """
    Final classification metrics for both held-out test sets.

    ---
    Attributes:
        mimic_test: ClassificationMetrics | RegressionMetrics
            Metrics on the MIMIC test set.

        mimic_prediction_time: float
            Prediction time on the MIMIC test set, in seconds.

        tudd_test: ClassificationMetrics | RegressionMetrics
            Metrics on the TUDD test set.

        tudd_prediction_time: float
            Prediction time on the TUDD test set, in seconds.
    """

    mimic_test: MetricT
    mimic_prediction_time: float
    tudd_test: MetricT
    tudd_prediction_time: float

    @property
    def mimic_minus_tudd(self) -> MetricT:
        return cast(MetricT, calculate_metric_diff(self.mimic_test, self.tudd_test))


@dataclass(frozen=True)
class BootstrapFinalTestMetrics(Generic[ConfidenceMetricT]):
    """
    Final-test point metrics and confidence intervals for a tuned model.

    The intervals come from bootstrap resampling of one model's full-test
    predictions.

    ---
    Attributes:
        mimic_test: ConfidenceMetrics
            Point metrics and confidence intervals on the MIMIC test set.

        mimic_prediction_time: float
            Prediction time in seconds.

        tudd_test: ConfidenceMetrics
            Point metrics and confidence intervals on the TUDD test set.

        tudd_prediction_time: float
            Prediction time in seconds.
    """

    mimic_test: ConfidenceMetricT
    mimic_prediction_time: float
    tudd_test: ConfidenceMetricT
    tudd_prediction_time: float

    @property
    def mimic_minus_tudd(self) -> ClassificationMetrics | RegressionMetrics:
        return calculate_metric_diff(self.mimic_test, self.tudd_test)


@overload
def calculate_metric_diff(
    mimic_metrics: ClassificationMetrics,
    tudd_metrics: ClassificationMetrics,
) -> ClassificationMetrics: ...


@overload
def calculate_metric_diff(
    mimic_metrics: RegressionMetrics,
    tudd_metrics: RegressionMetrics,
) -> RegressionMetrics: ...


@overload
def calculate_metric_diff(
    mimic_metrics: BootstrapClassificationMetrics,
    tudd_metrics: BootstrapClassificationMetrics,
) -> ClassificationMetrics: ...


@overload
def calculate_metric_diff(
    mimic_metrics: BootstrapRegressionMetrics,
    tudd_metrics: BootstrapRegressionMetrics,
) -> RegressionMetrics: ...


def calculate_metric_diff(
    mimic_metrics: ClassificationMetrics
    | RegressionMetrics
    | BootstrapClassificationMetrics
    | BootstrapRegressionMetrics,
    tudd_metrics: ClassificationMetrics
    | RegressionMetrics
    | BootstrapClassificationMetrics
    | BootstrapRegressionMetrics,
) -> ClassificationMetrics | RegressionMetrics:
    """
    Calculate the difference between two sets of metrics.

    Both inputs must be matching classification or regression metric variants.
    Classification diffs always report confusion_matrix as None since it is not
    subtracted.
    """

    if isinstance(mimic_metrics, BootstrapClassificationMetrics) and isinstance(
        tudd_metrics, BootstrapClassificationMetrics
    ):
        return calculate_metric_diff(mimic_metrics.metrics, tudd_metrics.metrics)

    if isinstance(mimic_metrics, BootstrapRegressionMetrics) and isinstance(tudd_metrics, BootstrapRegressionMetrics):
        return calculate_metric_diff(mimic_metrics.metrics, tudd_metrics.metrics)

    if isinstance(mimic_metrics, RegressionMetrics) and isinstance(tudd_metrics, RegressionMetrics):
        return RegressionMetrics(
            r2=mimic_metrics.r2 - tudd_metrics.r2,
            mae=mimic_metrics.mae - tudd_metrics.mae,
            mse=mimic_metrics.mse - tudd_metrics.mse,
            rmse=mimic_metrics.rmse - tudd_metrics.rmse,
        )

    if isinstance(mimic_metrics, ClassificationMetrics) and isinstance(tudd_metrics, ClassificationMetrics):
        if mimic_metrics.roc_auc is not None and tudd_metrics.roc_auc is not None:
            roc_auc = mimic_metrics.roc_auc - tudd_metrics.roc_auc
        else:
            roc_auc = None

        if mimic_metrics.prc_auc is not None and tudd_metrics.prc_auc is not None:
            prc_auc = mimic_metrics.prc_auc - tudd_metrics.prc_auc
        else:
            prc_auc = None

        return ClassificationMetrics(
            roc_auc=roc_auc,
            prc_auc=prc_auc,
            f1=mimic_metrics.f1 - tudd_metrics.f1,
            accuracy=mimic_metrics.accuracy - tudd_metrics.accuracy,
            sensitivity=mimic_metrics.sensitivity - tudd_metrics.sensitivity,
            precision=mimic_metrics.precision - tudd_metrics.precision,
            n_classes=mimic_metrics.n_classes,
            confusion_matrix=None,
        )

    raise ValueError("mimic_metrics and tudd_metrics must be of the same type")


@dataclass(frozen=True)
class BootstrapClassificationMetrics:
    metrics: ClassificationMetrics
    ci_95_roc_auc_lower: float
    ci_95_roc_auc_upper: float
    ci_95_prc_auc_lower: float
    ci_95_prc_auc_upper: float
    ci_95_f1_lower: float
    ci_95_f1_upper: float
    ci_95_accuracy_lower: float
    ci_95_accuracy_upper: float
    ci_95_sensitivity_lower: float
    ci_95_sensitivity_upper: float
    ci_95_precision_lower: float
    ci_95_precision_upper: float
    n_bootstrap: int

    @property
    def roc_auc(self) -> float | None:
        return self.metrics.roc_auc

    @property
    def prc_auc(self) -> float | None:
        return self.metrics.prc_auc

    @property
    def f1(self) -> float:
        return self.metrics.f1

    @property
    def accuracy(self) -> float:
        return self.metrics.accuracy

    @property
    def sensitivity(self) -> float:
        return self.metrics.sensitivity

    @property
    def precision(self) -> float:
        return self.metrics.precision

    @property
    def n_classes(self) -> int:
        return self.metrics.n_classes

    @property
    def scores(self) -> dict[str, float]:
        return self.metrics.scores

    @property
    def confidence_intervals(self) -> dict[str, tuple[float, float]]:
        return {
            "roc_auc": (self.ci_95_roc_auc_lower, self.ci_95_roc_auc_upper),
            "prc_auc": (self.ci_95_prc_auc_lower, self.ci_95_prc_auc_upper),
            "f1": (self.ci_95_f1_lower, self.ci_95_f1_upper),
            "accuracy": (self.ci_95_accuracy_lower, self.ci_95_accuracy_upper),
            "sensitivity": (self.ci_95_sensitivity_lower, self.ci_95_sensitivity_upper),
            "precision": (self.ci_95_precision_lower, self.ci_95_precision_upper),
        }


@dataclass(frozen=True)
class BootstrapRegressionMetrics:
    metrics: RegressionMetrics
    ci_95_r2_lower: float
    ci_95_r2_upper: float
    ci_95_mae_lower: float
    ci_95_mae_upper: float
    ci_95_mse_lower: float
    ci_95_mse_upper: float
    ci_95_rmse_lower: float
    ci_95_rmse_upper: float
    n_bootstrap: int

    @property
    def r2(self) -> float:
        return self.metrics.r2

    @property
    def mae(self) -> float:
        return self.metrics.mae

    @property
    def mse(self) -> float:
        return self.metrics.mse

    @property
    def rmse(self) -> float:
        return self.metrics.rmse

    @property
    def scores(self) -> dict[str, float]:
        return self.metrics.scores

    @property
    def confidence_intervals(self) -> dict[str, tuple[float, float]]:
        return {
            "r2": (self.ci_95_r2_lower, self.ci_95_r2_upper),
            "mae": (self.ci_95_mae_lower, self.ci_95_mae_upper),
            "mse": (self.ci_95_mse_lower, self.ci_95_mse_upper),
            "rmse": (self.ci_95_rmse_lower, self.ci_95_rmse_upper),
        }
