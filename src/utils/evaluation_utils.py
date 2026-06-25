from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, label_binarize

ScoringMethodCLS = Literal["roc_auc", "f1", "accuracy"]
ScoringMethodREG = Literal["r2", "mae", "mse"]


@dataclass
class ClassificationMetrics:
    """Classification metrics for one prediction set."""

    roc_auc: float | None
    prc_auc: float | None
    f1: float
    accuracy: float
    sensitivity: float
    precision: float
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
    tudd_test: ClassificationMetrics
    mimic_minus_tudd: ClassificationMetricDeltas


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
        prc_auc = float(average_precision_score(batch.y_true, batch.probabilities[:, 1]))
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

    return ClassificationMetrics(
        roc_auc=roc_auc,
        prc_auc=prc_auc,
        f1=f1,
        accuracy=accuracy,
        sensitivity=sensitivity,
        precision=precision,
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
        raise ValueError("Classification probabilities must include at least two classes")
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
    metrics: ClassificationMetrics,
    scoring: ScoringMethodCLS,
) -> float:
    score = metrics.scores.get(scoring)
    if score is None:
        raise ValueError(f"Scoring method '{scoring}' requires 2D class probabilities")
    return score


def mean_classification_metrics(
    metrics: list[ClassificationMetrics],
) -> ClassificationMetrics:
    if not metrics:
        raise ValueError("Cannot aggregate empty metric list")

    roc_auc = _mean_optional([metric.roc_auc for metric in metrics])
    prc_auc = _mean_optional([metric.prc_auc for metric in metrics])
    f1 = float(np.mean([metric.f1 for metric in metrics]))
    accuracy = float(np.mean([metric.accuracy for metric in metrics]))
    sensitivity = float(np.mean([metric.sensitivity for metric in metrics]))
    precision = float(np.mean([metric.precision for metric in metrics]))

    return ClassificationMetrics(
        roc_auc=roc_auc,
        prc_auc=prc_auc,
        f1=f1,
        accuracy=accuracy,
        sensitivity=sensitivity,
        precision=precision,
        n_classes=metrics[0].n_classes,
    )


def final_test_metrics(
    mimic_test: ClassificationMetrics,
    tudd_test: ClassificationMetrics,
) -> FinalTestMetrics:
    return FinalTestMetrics(
        mimic_test=mimic_test,
        tudd_test=tudd_test,
        mimic_minus_tudd=ClassificationMetricDeltas(
            roc_auc=_optional_delta(mimic_test.roc_auc, tudd_test.roc_auc),
            prc_auc=_optional_delta(mimic_test.prc_auc, tudd_test.prc_auc),
            f1=mimic_test.f1 - tudd_test.f1,
            accuracy=mimic_test.accuracy - tudd_test.accuracy,
            sensitivity=mimic_test.sensitivity - tudd_test.sensitivity,
            precision=mimic_test.precision - tudd_test.precision,
        ),
    )


def _optional_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _mean_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return float(np.mean(present))


def _classification_average(n_classes: int) -> str:
    return "binary" if n_classes == 2 else "macro"
