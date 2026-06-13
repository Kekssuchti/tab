"""
Evaluation module for classification models.

Provides standard metrics for assessing prediction quality:
AUROC, AUPRC, F1-score, and Accuracy.
"""

import numpy as np
from numpy import ndarray
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder


def evaluate_predictions(
    predictions: ndarray,
    y_test: ndarray,
) -> dict[str, float]:
    """Evaluate classification predictions against ground truth.

    Works for binary and multi-class problems.  String labels in ``y_test``
    are encoded automatically.

    Args:
        predictions:
            2D array of shape ``(n_samples, n_classes)`` with class
            probabilities (output of ``predict_proba``).
        y_test:
            1D array-like of shape ``(n_samples,)`` with true class labels.
            May be a numpy array, pandas Series, or list.

    Returns:
        dict with keys::

            "auroc"     — Area Under the ROC Curve (One-vs-Rest for multi-class)
            "auprc"     — Area Under the Precision-Recall Curve
            "f1"        — F1 score (macro-averaged for multi-class)
            "accuracy"  — Accuracy
            "n_classes" — Number of classes detected

    Raises:
        ValueError: If ``predictions`` is not 2D.
    """
    if predictions.ndim != 2:
        raise ValueError(
            f"predictions must be 2D (n_samples, n_classes), "
            f"got shape {predictions.shape}"
        )

    # --- Normalise y_test into a flat 1D numpy array of integers ---
    # Handles: pandas Series, list, 2D column vectors, string labels
    y_test = np.asarray(y_test).ravel()

    # If labels are strings (e.g. "Low"/"Medium"/"High"), encode to ints
    if not np.issubdtype(y_test.dtype, np.number):
        y_test = LabelEncoder().fit_transform(y_test)

    n_classes = predictions.shape[1]

    # Hard predictions via argmax (works for any n_classes, no threshold
    # shenanigans)
    y_pred = predictions.argmax(axis=1)

    # --- AUROC ---
    if n_classes == 2:
        auroc = float(roc_auc_score(y_test, predictions[:, 1]))
        auprc = float(average_precision_score(y_test, predictions[:, 1]))
    else:
        auroc = float(roc_auc_score(y_test, predictions, multi_class="ovr"))
        auprc = float(average_precision_score(y_test, predictions, average="macro"))

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": float(f1_score(y_test, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "n_classes": n_classes,
    }
