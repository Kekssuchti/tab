from dataclasses import dataclass

import numpy as np


@dataclass
class ClassificationMetrics:
    """Classification metrics for one prediction set."""

    roc_auc: float | None
    prc_auc: float | None
    f1: float
    accuracy: float
    sensitivity: float
    precision: float
    confusion_matrix: np.ndarray
    n_classes: int

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
class ClassificationMetricsAggregate:
    mean_roc_auc: float
    mean_prc_auc: float
    mean_f1: float
    mean_accuracy: float
    mean_sensitivity: float
    mean_precision: float
    mean_confusion_matrix: np.ndarray
    n_classes: int
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

    def __init__(self, metrics: list[ClassificationMetrics]) -> None:
        from src.utils.evaluation_utils import calculate_mean_ci

        self.mean_roc_auc, self.ci_95_roc_auc_lower, self.ci_95_roc_auc_upper = (
            calculate_mean_ci(
                [m.roc_auc for m in metrics if m.roc_auc is not None], 0.95
            )
        )
        self.mean_prc_auc, self.ci_95_prc_auc_lower, self.ci_95_prc_auc_upper = (
            calculate_mean_ci(
                [m.prc_auc for m in metrics if m.prc_auc is not None], 0.95
            )
        )
        self.mean_f1, self.ci_95_f1_lower, self.ci_95_f1_upper = calculate_mean_ci(
            [m.f1 for m in metrics], 0.95
        )
        self.mean_accuracy, self.ci_95_accuracy_lower, self.ci_95_accuracy_upper = (
            calculate_mean_ci([m.accuracy for m in metrics], 0.95)
        )
        (
            self.mean_sensitivity,
            self.ci_95_sensitivity_lower,
            self.ci_95_sensitivity_upper,
        ) = calculate_mean_ci([m.sensitivity for m in metrics], 0.95)
        self.mean_precision, self.ci_95_precision_lower, self.ci_95_precision_upper = (
            calculate_mean_ci([m.precision for m in metrics], 0.95)
        )
        self.mean_confusion_matrix = np.mean(
            np.stack([m.confusion_matrix for m in metrics]), axis=0
        )
        self.n_classes = metrics[0].n_classes

    @property
    def scores(self) -> dict[str, float]:
        return {
            "mean_roc_auc": self.mean_roc_auc,
            "mean_prc_auc": self.mean_prc_auc,
            "mean_f1": self.mean_f1,
            "mean_accuracy": self.mean_accuracy,
            "mean_sensitivity": self.mean_sensitivity,
            "mean_precision": self.mean_precision,
        }


@dataclass
class RegressionMetrics:
    r2: float
    mae: float
    mse: float

    @property
    def scores(self) -> dict[str, float]:
        return {
            "r2": self.r2,
            "mae": self.mae,
            "mse": self.mse,
        }


@dataclass(frozen=True)
class ClassificationMetricDeltas:
    """Signed final-test deltas. Positive means mimic outperformed tudd."""

    roc_auc: float | None
    prc_auc: float | None
    f1: float
    accuracy: float
    sensitivity: float
    precision: float

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


@dataclass(frozen=True)
class FinalTestMetrics:
    mimic_test: ClassificationMetrics
    mimic_prediction_time: float
    tudd_test: ClassificationMetrics
    tudd_prediction_time: float

    @property
    def mimic_minus_tudd(self) -> ClassificationMetricDeltas:
        return ClassificationMetricDeltas(
            roc_auc=self.mimic_test.roc_auc - self.tudd_test.roc_auc
            if self.mimic_test.roc_auc is not None and self.tudd_test.roc_auc is not None
            else None,
            prc_auc=self.mimic_test.prc_auc - self.tudd_test.prc_auc
            if self.mimic_test.prc_auc is not None and self.tudd_test.prc_auc is not None
            else None,
            f1=self.mimic_test.f1 - self.tudd_test.f1,
            accuracy=self.mimic_test.accuracy - self.tudd_test.accuracy,
            sensitivity=self.mimic_test.sensitivity - self.tudd_test.sensitivity,
            precision=self.mimic_test.precision - self.tudd_test.precision,
        )


@dataclass(frozen=True)
class AggregatedFinalTestMetrics:
    mimic_test: ClassificationMetricsAggregate
    mimic_prediction_time: float
    tudd_test: ClassificationMetricsAggregate
    tudd_prediction_time: float

    @property
    def mimic_minus_tudd(self) -> ClassificationMetricDeltas:
        return ClassificationMetricDeltas(
            roc_auc=self.mimic_test.mean_roc_auc - self.tudd_test.mean_roc_auc,
            prc_auc=self.mimic_test.mean_prc_auc - self.tudd_test.mean_prc_auc,
            f1=self.mimic_test.mean_f1 - self.tudd_test.mean_f1,
            accuracy=self.mimic_test.mean_accuracy - self.tudd_test.mean_accuracy,
            sensitivity=self.mimic_test.mean_sensitivity
            - self.tudd_test.mean_sensitivity,
            precision=self.mimic_test.mean_precision - self.tudd_test.mean_precision,
        )


@dataclass(frozen=True)
class ClassificationPredictionBatch:
    probabilities: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    n_classes: int
