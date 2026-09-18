from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from src.schemas.metrics import ClassificationMetrics


@dataclass(frozen=True)
class ClassificationPredictionBatch:
    probabilities: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    n_classes: int


def classification_metrics(
    batch: ClassificationPredictionBatch,
) -> ClassificationMetrics:
    values, _ = _metric_values(
        batch.y_true,
        batch.probabilities[:, 1],
        batch.y_pred,
    )
    roc_auc, prc_auc, f1, accuracy, sensitivity, precision = values

    return ClassificationMetrics(
        roc_auc=roc_auc,
        prc_auc=prc_auc,
        f1=f1,
        accuracy=accuracy,
        sensitivity=sensitivity,
        precision=precision,
        n_classes=2,
        confusion_matrix=None,
    )


def bootstrap_scores_classification(
    batch: ClassificationPredictionBatch,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Evaluate class-stratified bootstrap samples with a conventional loop."""
    negative = batch.y_true == 0
    positive = batch.y_true == 1
    negative_scores = batch.probabilities[negative, 1]
    positive_scores = batch.probabilities[positive, 1]
    negative_predictions = batch.y_pred[negative]
    positive_predictions = batch.y_pred[positive]
    n_negative = negative_scores.size
    n_positive = positive_scores.size
    sample_y_true = np.r_[np.zeros(n_negative, dtype=np.int8), np.ones(n_positive, dtype=np.int8)]
    output = np.empty((6, n_bootstrap))

    for bootstrap_index in range(n_bootstrap):
        negative_sample = rng.integers(n_negative, size=n_negative)
        positive_sample = rng.integers(n_positive, size=n_positive)
        sample_scores = np.concatenate((negative_scores[negative_sample], positive_scores[positive_sample]))

        roc_auc, prc_auc = _bootstrap_ranking_metrics(sample_y_true, sample_scores)
        false_positive = np.count_nonzero(negative_predictions[negative_sample])
        true_positive = np.count_nonzero(positive_predictions[positive_sample])
        confusion_scores = _confusion_scores(
            n_negative - false_positive,
            false_positive,
            n_positive - true_positive,
            true_positive,
        )
        output[:, bootstrap_index] = roc_auc, prc_auc, *confusion_scores

    return output


def _bootstrap_ranking_metrics(
    y_true: np.ndarray,
    positive_probability: np.ndarray,
) -> tuple[float, float]:
    """Calculate AUROC and average precision with one shared score sort."""
    order = np.argsort(positive_probability, kind="stable")[::-1]
    ordered_y_true = y_true[order]
    ordered_probability = positive_probability[order]
    group_ends = np.r_[
        np.flatnonzero(ordered_probability[1:] != ordered_probability[:-1]),
        y_true.size - 1,
    ]

    true_positive = np.cumsum(ordered_y_true)[group_ends]
    false_positive = group_ends + 1 - true_positive
    n_positive = true_positive[-1]
    n_negative = false_positive[-1]

    roc_auc = np.trapezoid(
        np.r_[0, true_positive],
        np.r_[0, false_positive],
    ) / (n_positive * n_negative)
    precision = true_positive / (true_positive + false_positive)
    prc_auc = np.sum(np.diff(np.r_[0, true_positive]) * precision) / n_positive
    return float(roc_auc), float(prc_auc)


def _metric_values(
    y_true: np.ndarray,
    positive_probability: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[tuple[float, float, float, float, float, float], np.ndarray]:
    """Calculate metrics in the public result order."""
    roc_auc = float(roc_auc_score(y_true, positive_probability))
    prc_auc = float(average_precision_score(y_true, positive_probability))

    confusion = confusion_matrix(y_true, y_pred, labels=(0, 1))
    true_negative, false_positive, false_negative, true_positive = confusion.ravel()

    confusion_scores = _confusion_scores(
        true_negative,
        false_positive,
        false_negative,
        true_positive,
    )
    return (roc_auc, prc_auc, *confusion_scores), confusion


def _confusion_scores(
    true_negative: int,
    false_positive: int,
    false_negative: int,
    true_positive: int,
) -> tuple[float, float, float, float]:
    f1_denominator = 2 * true_positive + false_positive + false_negative
    precision_denominator = true_positive + false_positive
    sensitivity_denominator = true_positive + false_negative
    n_samples = true_positive + true_negative + false_positive + false_negative

    f1 = float(2 * true_positive / f1_denominator) if f1_denominator else 0.0
    accuracy = float((true_positive + true_negative) / n_samples)
    sensitivity = float(true_positive / sensitivity_denominator) if sensitivity_denominator else 0.0
    precision = float(true_positive / precision_denominator) if precision_denominator else 0.0
    return f1, accuracy, sensitivity, precision
