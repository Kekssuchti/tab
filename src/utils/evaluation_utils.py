import numpy as np

from src.schemas.metrics import (
    ClassificationMetrics,
    RegressionMetrics,
)
from src.schemas.training_schemas import ClassificationScoring, RegressionScoring
from src.utils.bootstrap_utils_cls import (
    _ClassificationPredictionBatch,
    bootstrap_scores_classification,
    classification_metrics,
)
from src.utils.bootstrap_utils_reg import bootstrap_scores_regression, regression_metrics


def evaluate_classification_predictions(
    predictions: np.ndarray,
    y_true,
) -> ClassificationMetrics:
    batch = classification_prediction_batch(predictions, y_true)

    return classification_metrics(batch)


def classification_prediction_batch(
    predictions: np.ndarray,
    y_true,
) -> _ClassificationPredictionBatch:
    probabilities = np.asarray(predictions)
    labels = np.asarray(y_true).ravel()

    if probabilities.ndim != 2:
        raise ValueError(
            "Classification adapters must return a 2D class-probability array with shape (n_samples, n_classes)"
        )
    if probabilities.shape[0] != labels.size:
        raise ValueError(
            "Classification prediction row count does not match y_true: "
            f"got {probabilities.shape[0]} predictions for {labels.size} labels"
        )
    if probabilities.shape[1] < 2:
        raise ValueError("Classification probabilities must include at least two classes")
    if probabilities.shape[1] != 2:
        raise NotImplementedError("Multi-class ROC/AUC not yet supported")

    if not np.isfinite(probabilities).all():
        raise ValueError("Classification probabilities must be finite")

    _, encoded_labels = np.unique(labels, return_inverse=True)

    return _ClassificationPredictionBatch(
        probabilities=probabilities,
        y_true=encoded_labels,
        y_pred=probabilities.argmax(axis=1),
        n_classes=probabilities.shape[1],  # 2
    )


def classification_score(metrics: ClassificationMetrics, scoring: ClassificationScoring) -> float:
    score = metrics.scores.get(scoring)
    if score is None:
        raise ValueError(f"Scoring method '{scoring}' requires 2D class probabilities")
    return score


def regression_score(metrics: RegressionMetrics, scoring: RegressionScoring) -> float:
    return metrics.scores[scoring]


def mean_classification_metrics(metrics: list[ClassificationMetrics]) -> ClassificationMetrics:
    if not metrics:
        raise ValueError("Cannot average an empty metric list")

    confusion_matrices = [metric.confusion_matrix for metric in metrics if metric.confusion_matrix is not None]
    return ClassificationMetrics(
        roc_auc=float(np.mean([metric.roc_auc for metric in metrics if metric.roc_auc is not None])),
        prc_auc=float(np.mean([metric.prc_auc for metric in metrics if metric.prc_auc is not None])),
        f1=float(np.mean([metric.f1 for metric in metrics])),
        accuracy=float(np.mean([metric.accuracy for metric in metrics])),
        sensitivity=float(np.mean([metric.sensitivity for metric in metrics])),
        precision=float(np.mean([metric.precision for metric in metrics])),
        confusion_matrix=(np.mean(np.stack(confusion_matrices), axis=0) if confusion_matrices else None),
        n_classes=metrics[0].n_classes,
    )


def mean_regression_metrics(metrics: list[RegressionMetrics]) -> RegressionMetrics:
    if not metrics:
        raise ValueError("Cannot average an empty metric list")

    return RegressionMetrics(
        r2=float(np.mean([metric.r2 for metric in metrics])),
        mae=float(np.mean([metric.mae for metric in metrics])),
        mse=float(np.mean([metric.mse for metric in metrics])),
        rmse=float(np.mean([metric.rmse for metric in metrics])),
    )


def evaluate_bootstrap_classification(
    predictions: np.ndarray,
    y_true: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[ClassificationMetrics, np.ndarray, np.ndarray]:
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")

    batch = classification_prediction_batch(predictions, y_true)
    metrics = classification_metrics(batch)
    scores = bootstrap_scores_classification(
        batch,
        n_bootstrap,
        rng,
    )
    lower, upper = np.percentile(scores, [2.5, 97.5], axis=1)
    return metrics, lower, upper


def evaluate_regression_predictions(
    predictions: np.ndarray,
    true_values: np.ndarray,
) -> RegressionMetrics:
    return regression_metrics(predictions, true_values)


def evaluate_bootstrap_regression(
    predictions: np.ndarray,
    y_true: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[RegressionMetrics, np.ndarray, np.ndarray]:
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")

    metrics = regression_metrics(predictions, y_true)

    scores = bootstrap_scores_regression(
        predictions,
        y_true,
        n_bootstrap,
        rng,
    )
    lower, upper = np.percentile(scores, [2.5, 97.5], axis=1)
    return metrics, lower, upper
