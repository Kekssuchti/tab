from __future__ import annotations

from dataclasses import dataclass
from typing import cast, overload

import numpy as np


@dataclass
class ClassificationMetrics:
    """Point metrics for one binary classification prediction set."""

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
    """Point metrics for one regression prediction set."""

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


@dataclass(frozen=True)
class FinalTestMetrics[MetricT: (ClassificationMetrics, RegressionMetrics)]:
    """Point metrics and prediction times for the two held-out datasets."""

    mimic_test: MetricT
    mimic_prediction_time: float
    tudd_test: MetricT
    tudd_prediction_time: float

    @property
    def mimic_minus_tudd(self) -> MetricT:
        return cast(MetricT, calculate_metric_diff(self.mimic_test, self.tudd_test))


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


def calculate_metric_diff(
    mimic_metrics: ClassificationMetrics | RegressionMetrics,
    tudd_metrics: ClassificationMetrics | RegressionMetrics,
) -> ClassificationMetrics | RegressionMetrics:
    """Subtract matching TUDD point metrics from MIMIC point metrics."""

    if isinstance(mimic_metrics, RegressionMetrics) and isinstance(tudd_metrics, RegressionMetrics):
        return RegressionMetrics(
            r2=mimic_metrics.r2 - tudd_metrics.r2,
            mae=mimic_metrics.mae - tudd_metrics.mae,
            mse=mimic_metrics.mse - tudd_metrics.mse,
            rmse=mimic_metrics.rmse - tudd_metrics.rmse,
        )

    if isinstance(mimic_metrics, ClassificationMetrics) and isinstance(tudd_metrics, ClassificationMetrics):
        roc_auc = (
            mimic_metrics.roc_auc - tudd_metrics.roc_auc
            if mimic_metrics.roc_auc is not None and tudd_metrics.roc_auc is not None
            else None
        )
        prc_auc = (
            mimic_metrics.prc_auc - tudd_metrics.prc_auc
            if mimic_metrics.prc_auc is not None and tudd_metrics.prc_auc is not None
            else None
        )
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

    raise TypeError("mimic_metrics and tudd_metrics must have the same metric type")
