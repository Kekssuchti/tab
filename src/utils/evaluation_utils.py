from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder, label_binarize

from src.utils.logger import logger

ScoringMethodCLS = Literal["roc_auc", "f1", "accuracy"]
ScoringMethodREG = Literal["r2", "mae", "mse"]


@dataclass
class ClassificationMetrics:
    """Classification metrics for one prediction set."""

    roc_auc: float
    prc_auc: float
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
class CVClassificationMetrics:
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
        # Calculate confidence intervals
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
            roc_auc=self.mimic_test.roc_auc - self.tudd_test.roc_auc,
            prc_auc=self.mimic_test.prc_auc - self.tudd_test.prc_auc,
            f1=self.mimic_test.f1 - self.tudd_test.f1,
            accuracy=self.mimic_test.accuracy - self.tudd_test.accuracy,
            sensitivity=self.mimic_test.sensitivity - self.tudd_test.sensitivity,
            precision=self.mimic_test.precision - self.tudd_test.precision,
        )


@dataclass(frozen=True)
class CVFinalTestMetrics:
    mimic_test: CVClassificationMetrics
    mimic_prediction_time: float
    tudd_test: CVClassificationMetrics
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


def evaluate_classification_predictions(
    predictions: np.ndarray,
    y_true,
) -> ClassificationMetrics:
    batch = classification_prediction_batch(predictions, y_true)

    if batch.n_classes == 2:
        roc_auc = float(roc_auc_score(batch.y_true, batch.probabilities[:, 1]))
        prc_auc = float(
            average_precision_score(batch.y_true, batch.probabilities[:, 1])
        )
    else:
        roc_auc = float(
            roc_auc_score(batch.y_true, batch.probabilities, multi_class="ovr")
        )
        classes = np.arange(batch.n_classes)
        y_binary = label_binarize(batch.y_true, classes=classes)
        prc_auc = float(
            average_precision_score(y_binary, batch.probabilities, average="macro")
        )

    average = _classification_average(batch.n_classes)
    f1 = float(f1_score(batch.y_true, batch.y_pred, average=average))
    accuracy = float(accuracy_score(batch.y_true, batch.y_pred))
    sensitivity = float(recall_score(batch.y_true, batch.y_pred, average=average))
    precision = float(
        precision_score(batch.y_true, batch.y_pred, average=average, zero_division=0)
    )
    confusion = confusion_matrix(batch.y_true, batch.y_pred)

    return ClassificationMetrics(
        roc_auc=roc_auc,
        prc_auc=prc_auc,
        f1=f1,
        accuracy=accuracy,
        sensitivity=sensitivity,
        precision=precision,
        confusion_matrix=confusion,
        n_classes=batch.n_classes,
    )


def classification_prediction_batch(
    predictions: np.ndarray,
    y_true,
) -> ClassificationPredictionBatch:
    y_true = np.asarray(y_true).ravel()
    probabilities = np.asarray(predictions)

    if probabilities.ndim != 2:
        raise ValueError(
            "Classification adapters must return a 2D class-probability array "
            "with shape (n_samples, n_classes)"
        )
    if probabilities.shape[0] != y_true.shape[0]:
        raise ValueError(
            "Classification prediction row count does not match y_true: "
            f"got {probabilities.shape[0]} predictions for {y_true.shape[0]} labels"
        )
    if probabilities.shape[1] < 2:
        raise ValueError(
            "Classification probabilities must include at least two classes"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("Classification probabilities must be finite")

    y_true = LabelEncoder().fit_transform(y_true)
    return ClassificationPredictionBatch(
        probabilities=probabilities,
        y_true=y_true,
        y_pred=probabilities.argmax(axis=1),
        n_classes=probabilities.shape[1],
    )


def classification_score(
    metrics: ClassificationMetrics | CVClassificationMetrics,
    scoring: ScoringMethodCLS,
) -> float:
    scoring_name = (
        f"mean_{scoring}" if isinstance(metrics, CVClassificationMetrics) else scoring
    )
    score = metrics.scores.get(scoring_name)
    if score is None:
        raise ValueError(f"Scoring method '{scoring}' requires 2D class probabilities")
    return score


def mean_classification_metrics(
    metrics: list[ClassificationMetrics],
) -> ClassificationMetrics:
    if not metrics:
        raise ValueError("Cannot aggregate empty metric list")

    return ClassificationMetrics(
        roc_auc=float(np.mean([metric.roc_auc for metric in metrics])),
        prc_auc=float(np.mean([metric.prc_auc for metric in metrics])),
        f1=float(np.mean([metric.f1 for metric in metrics])),
        accuracy=float(np.mean([metric.accuracy for metric in metrics])),
        sensitivity=float(np.mean([metric.sensitivity for metric in metrics])),
        precision=float(np.mean([metric.precision for metric in metrics])),
        confusion_matrix=np.mean(
            np.stack([metric.confusion_matrix for metric in metrics]), axis=0
        ),
        n_classes=metrics[0].n_classes,
    )


def final_test_metrics(
    mimic_test: ClassificationMetrics,
    tudd_test: ClassificationMetrics,
    mimic_prediction_time: float = 0.0,
    tudd_prediction_time: float = 0.0,
) -> FinalTestMetrics:
    return FinalTestMetrics(
        mimic_test=mimic_test,
        mimic_prediction_time=mimic_prediction_time,
        tudd_test=tudd_test,
        tudd_prediction_time=tudd_prediction_time,
    )


def _classification_average(n_classes: int) -> str:
    return "binary" if n_classes == 2 else "macro"


def calculate_mean_ci(
    values: list[float], confidence: float = 0.95
) -> tuple[float, float, float]:
    if not values:
        logger.error("Cannot calculate mean CI from empty list")
        return 0, 0, 0

    mean = np.mean(values)
    n = len(values)
    if n == 1:
        mean = float(mean)
        return mean, mean, mean

    std = np.std(values, ddof=1)  # use /n-1 for sample std
    z = norm.ppf((1 + confidence) / 2)
    ci_lower = mean - z * std / np.sqrt(n)
    ci_upper = mean + z * std / np.sqrt(n)

    return float(mean), ci_lower, ci_upper


def _format_metrics(
    cv_results: list[FinalTestMetrics],
) -> CVFinalTestMetrics:
    mimic_test = CVClassificationMetrics([result.mimic_test for result in cv_results])
    tudd_test = CVClassificationMetrics([result.tudd_test for result in cv_results])

    return CVFinalTestMetrics(
        mimic_test=mimic_test,
        tudd_test=tudd_test,
        mimic_prediction_time=float(
            np.mean([result.mimic_prediction_time for result in cv_results])
        ),
        tudd_prediction_time=float(
            np.mean([result.tudd_prediction_time for result in cv_results])
        ),
    )
