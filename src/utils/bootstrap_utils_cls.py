from dataclasses import dataclass

import numpy as np

from src.schemas.metrics import ClassificationMetrics


@dataclass(frozen=True)
class _ClassificationPredictionBatch:
    probabilities: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    n_classes: int


def classification_metrics(
    batch: _ClassificationPredictionBatch,
) -> ClassificationMetrics:
    order, group_starts = _score_groups(batch.probabilities[:, 1])
    ordered_positive = batch.y_true[order]

    positive_by_group = _sum_by_group(ordered_positive, group_starts)
    total_by_group = np.diff(np.r_[group_starts, batch.y_true.size])
    roc_auc, prc_auc = _binary_ranking_metrics(positive_by_group, total_by_group - positive_by_group)

    confusion = np.bincount(2 * batch.y_true + batch.y_pred, minlength=4).reshape(2, 2)
    true_negative, false_positive, false_negative, true_positive = confusion.ravel()
    f1, accuracy, sensitivity, precision = _binary_confusion_metrics(
        true_negative,
        false_positive,
        false_negative,
        true_positive,
    )

    return ClassificationMetrics(
        roc_auc=roc_auc,
        prc_auc=prc_auc,
        f1=float(f1),
        accuracy=float(accuracy),
        sensitivity=float(sensitivity),
        precision=float(precision),
        n_classes=2,
        confusion_matrix=confusion,
    )


def bootstrap_scores_classification(
    batch: _ClassificationPredictionBatch,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Evaluate stratified bootstrap samples from per-observation sample weights."""
    order, group_starts = _score_groups(batch.probabilities[:, 1])
    ordered_positive = batch.y_true[order].astype(bool, copy=False)
    ordered_predicted_positive = batch.y_pred[order] == 1

    n_positive = int(ordered_positive.sum())
    n_negative = ordered_positive.size - n_positive
    n_samples = n_negative + n_positive

    group_ids = np.repeat(
        np.arange(group_starts.size),
        np.diff(np.r_[group_starts, n_samples]),
    )
    negative_group_ids = group_ids[~ordered_positive]
    positive_group_ids = group_ids[ordered_positive]

    batch_size = 250  # found through testing
    output = np.empty((6, n_bootstrap))
    roc_auc, prc_auc, f1, accuracy, sensitivity, precision = output

    for start in range(0, n_bootstrap, batch_size):
        stop = min(start + batch_size, n_bootstrap)
        size = stop - start
        negative_weights = _draw_bootstrap_sample_weights(rng, size, n_negative)
        positive_weights = _draw_bootstrap_sample_weights(rng, size, n_positive)

        false_positive = negative_weights[:, ordered_predicted_positive[~ordered_positive]].sum(axis=1)
        true_positive = positive_weights[:, ordered_predicted_positive[ordered_positive]].sum(axis=1)
        false_negative = n_positive - true_positive
        (
            f1[start:stop],
            accuracy[start:stop],
            sensitivity[start:stop],
            precision[start:stop],
        ) = _binary_confusion_metrics(
            n_negative - false_positive,
            false_positive,
            false_negative,
            true_positive,
        )

        negative_by_group = _sample_weights_by_score_group(
            negative_weights,
            negative_group_ids,
            group_starts.size,
        )
        positive_by_group = _sample_weights_by_score_group(
            positive_weights,
            positive_group_ids,
            group_starts.size,
        )
        roc_auc[start:stop], prc_auc[start:stop] = _binary_ranking_metrics(
            positive_by_group,
            negative_by_group,
        )

    return output


def _score_groups(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(scores, kind="stable")[::-1]
    ordered_scores = scores[order]
    group_starts = np.r_[
        0,
        np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]) + 1,
    ]
    return order, group_starts


def _sum_by_group(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    if starts.size == values.shape[-1]:
        return values
    return np.add.reduceat(values, starts, axis=-1)


def _binary_ranking_metrics(
    positive_by_group: np.ndarray,
    negative_by_group: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate AUROC and average precision from descending score groups."""
    cumulative_positive = np.cumsum(positive_by_group, axis=-1)
    cumulative_total = np.cumsum(positive_by_group + negative_by_group, axis=-1)
    n_positive = cumulative_positive[..., -1]
    n_negative = np.sum(negative_by_group, axis=-1)

    roc_auc = np.sum(
        negative_by_group * (cumulative_positive - 0.5 * positive_by_group),
        axis=-1,
    ) / (n_negative * n_positive)

    precision_at_threshold = _safe_ratio(cumulative_positive, cumulative_total)
    average_precision = (
        np.sum(
            positive_by_group * precision_at_threshold,
            axis=-1,
        )
        / n_positive
    )
    return roc_auc, average_precision


def _safe_ratio(
    numerator: np.ndarray | int,
    denominator: np.ndarray | int,
) -> np.ndarray:
    numerator = np.asarray(numerator)
    denominator = np.asarray(denominator)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator + denominator, dtype=float),
        where=denominator != 0,
    )


def _binary_confusion_metrics(
    true_negative: np.ndarray | int,
    false_positive: np.ndarray | int,
    false_negative: np.ndarray | int,
    true_positive: np.ndarray | int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate F1, accuracy, sensitivity, and precision from confusion counts."""
    f1 = _safe_ratio(
        2 * true_positive,
        2 * true_positive + false_positive + false_negative,
    )
    accuracy = _safe_ratio(
        true_positive + true_negative,
        true_positive + true_negative + false_positive + false_negative,
    )
    sensitivity = _safe_ratio(true_positive, true_positive + false_negative)
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    return f1, accuracy, sensitivity, precision


def _draw_bootstrap_sample_weights(
    rng: np.random.Generator,
    n_bootstrap: int,
    class_size: int,
) -> np.ndarray:
    """Draw samples as multiplicities rather than materialized row indices."""
    sampled_indices = rng.integers(
        class_size,
        size=(n_bootstrap, class_size),
        dtype=np.int32,
    )
    # Offsets let one bincount calculate observation counts for every sample.
    sampled_indices += np.arange(n_bootstrap, dtype=np.int32)[:, None] * class_size
    sample_weights = np.bincount(
        sampled_indices.ravel(),
        minlength=n_bootstrap * class_size,
    )
    return sample_weights.reshape(n_bootstrap, class_size).astype(
        np.int32,
        copy=False,
    )


def _sample_weights_by_score_group(
    sample_weights: np.ndarray,
    group_ids: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """Sum one class's sample weights into the shared score groups."""
    starts = np.r_[0, np.flatnonzero(group_ids[1:] != group_ids[:-1]) + 1]
    grouped_weights = np.zeros(
        (sample_weights.shape[0], n_groups),
        dtype=sample_weights.dtype,
    )
    grouped_weights[:, group_ids[starts]] = _sum_by_group(sample_weights, starts)
    return grouped_weights
