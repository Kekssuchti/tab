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

    primary_metric: ScoringMethodCLS
    primary_score: float
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

    @property
    def side_scores(self) -> dict[str, float]:
        return {
            name: score
            for name, score in self.scores.items()
            if name != self.primary_metric
        }


def evaluate_classification_predictions(
    scoring: ScoringMethodCLS,
    predictions: np.ndarray,
    y_true,
) -> ClassificationMetrics:
    y_true = np.asarray(y_true).ravel()
    predictions = np.asarray(predictions)

    if predictions.ndim == 2:
        y_true = LabelEncoder().fit_transform(y_true)
        y_pred = predictions.argmax(axis=1)
        n_classes = predictions.shape[1]
    else:
        encoder = LabelEncoder().fit(np.concatenate([y_true, predictions.ravel()]))
        y_true = encoder.transform(y_true)
        y_pred = encoder.transform(predictions.ravel())
        n_classes = len(np.unique(y_true))

    roc_auc = None
    prc_auc = None
    if predictions.ndim == 2:
        if n_classes == 2:
            roc_auc = float(roc_auc_score(y_true, predictions[:, 1]))
            prc_auc = float(average_precision_score(y_true, predictions[:, 1]))
        else:
            roc_auc = float(roc_auc_score(y_true, predictions, multi_class="ovr"))
            classes = np.arange(n_classes)
            y_binary = label_binarize(y_true, classes=classes)
            prc_auc = float(
                average_precision_score(y_binary, predictions, average="macro")
            )

    average = _classification_average(n_classes)
    f1 = float(f1_score(y_true, y_pred, average=average))
    accuracy = float(accuracy_score(y_true, y_pred))
    sensitivity = float(recall_score(y_true, y_pred, average=average))
    precision = float(
        precision_score(y_true, y_pred, average=average, zero_division=0)
    )

    scores = {"roc_auc": roc_auc, "f1": f1, "accuracy": accuracy}
    primary_score = scores[scoring]
    if primary_score is None:
        raise ValueError(f"Scoring method '{scoring}' requires 2D class probabilities")

    return ClassificationMetrics(
        primary_metric=scoring,
        primary_score=primary_score,
        roc_auc=roc_auc,
        prc_auc=prc_auc,
        f1=f1,
        accuracy=accuracy,
        sensitivity=sensitivity,
        precision=precision,
        n_classes=n_classes,
    )


def mean_classification_metrics(
    scoring: ScoringMethodCLS, metrics: list[ClassificationMetrics]
) -> ClassificationMetrics:
    if not metrics:
        raise ValueError("Cannot aggregate empty metric list")

    roc_auc = _mean_optional([metric.roc_auc for metric in metrics])
    prc_auc = _mean_optional([metric.prc_auc for metric in metrics])
    f1 = float(np.mean([metric.f1 for metric in metrics]))
    accuracy = float(np.mean([metric.accuracy for metric in metrics]))
    sensitivity = float(np.mean([metric.sensitivity for metric in metrics]))
    precision = float(np.mean([metric.precision for metric in metrics]))

    scores = {"roc_auc": roc_auc, "f1": f1, "accuracy": accuracy}
    primary_score = scores[scoring]
    if primary_score is None:
        raise ValueError(f"Scoring method '{scoring}' requires 2D class probabilities")

    return ClassificationMetrics(
        primary_metric=scoring,
        primary_score=primary_score,
        roc_auc=roc_auc,
        prc_auc=prc_auc,
        f1=f1,
        accuracy=accuracy,
        sensitivity=sensitivity,
        precision=precision,
        n_classes=metrics[0].n_classes,
    )


def _mean_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return float(np.mean(present))


def _classification_average(n_classes: int) -> str:
    return "binary" if n_classes == 2 else "macro"


@dataclass
class RegressionMetrics:
    primary_metric: ScoringMethodREG
    primary_score: float
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

    @property
    def side_scores(self) -> dict[str, float]:
        return {
            name: score
            for name, score in self.scores.items()
            if name != self.primary_metric
        }
