from typing import Literal

import numpy as np
from scipy.stats import norm
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.preprocessing import LabelEncoder, label_binarize

from src.schemas.metrics import (
    AggregatedFinalTestMetrics,
    ClassificationMetrics,
    ClassificationMetricsAggregate,
    ClassificationPredictionBatch,
    FinalTestMetrics,
    RegressionMetrics,
    RegressionMetricsAggregate,
)
from src.utils.logger import logger

ScoringMethodCLS = Literal["roc_auc", "f1", "accuracy"]
ScoringMethodREG = Literal["r2", "mae", "mse", "rmse"]


def evaluate_classification_predictions(
    predictions: np.ndarray,
    y_true,
) -> ClassificationMetrics:
    batch = classification_prediction_batch(predictions, y_true)

    if batch.n_classes == 2:
        roc_auc = float(roc_auc_score(batch.y_true, batch.probabilities[:, 1]))
        prc_auc = float(average_precision_score(batch.y_true, batch.probabilities[:, 1]))
    else:
        roc_auc = float(roc_auc_score(batch.y_true, batch.probabilities, multi_class="ovr"))
        classes = np.arange(batch.n_classes)
        y_binary = label_binarize(batch.y_true, classes=classes)
        prc_auc = float(average_precision_score(y_binary, batch.probabilities, average="macro"))

    average = _classification_average(batch.n_classes)
    f1 = float(f1_score(batch.y_true, batch.y_pred, average=average))
    accuracy = float(accuracy_score(batch.y_true, batch.y_pred))
    sensitivity = float(recall_score(batch.y_true, batch.y_pred, average=average))
    precision = float(precision_score(batch.y_true, batch.y_pred, average=average, zero_division=0))
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


def evaluate_regression_predictions(
    predictions: np.ndarray,
    true_values: np.ndarray,
) -> RegressionMetrics:
    rmse = float(root_mean_squared_error(true_values, predictions))
    mae = float(mean_absolute_error(true_values, predictions))
    mse = float(mean_squared_error(true_values, predictions))
    r2 = float(r2_score(true_values, predictions))

    return RegressionMetrics(
        rmse=rmse,
        mae=mae,
        mse=mse,
        r2=r2,
    )


def classification_prediction_batch(
    predictions: np.ndarray,
    y_true,
) -> ClassificationPredictionBatch:
    y_true = np.asarray(y_true).ravel()
    probabilities = np.asarray(predictions)

    if probabilities.ndim != 2:
        raise ValueError(
            "Classification adapters must return a 2D class-probability array with shape (n_samples, n_classes)"
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
    metrics: ClassificationMetrics | ClassificationMetricsAggregate,
    scoring: ScoringMethodCLS,
) -> float:
    scoring_name = f"mean_{scoring}" if isinstance(metrics, ClassificationMetricsAggregate) else scoring
    score = metrics.scores.get(scoring_name)
    if score is None:
        raise ValueError(f"Scoring method '{scoring}' requires 2D class probabilities")
    return score


def regression_score(
    metrics: RegressionMetrics | RegressionMetricsAggregate,
    scoring: ScoringMethodREG,
) -> float:
    scoring_name = f"mean_{scoring}" if isinstance(metrics, RegressionMetricsAggregate) else scoring
    score = metrics.scores.get(scoring_name)
    if score is None:
        raise ValueError(f"Scoring method '{scoring}' requires regression predictions")
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
        confusion_matrix=np.mean(np.stack([metric.confusion_matrix for metric in metrics]), axis=0),
        n_classes=metrics[0].n_classes,
    )


def mean_regression_metrics(
    metrics: list[RegressionMetrics],
) -> RegressionMetrics:
    if not metrics:
        raise ValueError("Cannot aggregate empty metric list")

    return RegressionMetrics(
        rmse=float(np.mean([metric.rmse for metric in metrics])),
        mae=float(np.mean([metric.mae for metric in metrics])),
        mse=float(np.mean([metric.mse for metric in metrics])),
        r2=float(np.mean([metric.r2 for metric in metrics])),
    )


def final_test_metrics(
    mimic_test: ClassificationMetrics | RegressionMetrics,
    tudd_test: ClassificationMetrics | RegressionMetrics,
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


def calculate_mean_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float, float]:
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
) -> AggregatedFinalTestMetrics:
    if isinstance(cv_results[0].mimic_test, RegressionMetrics):
        mimic_test = RegressionMetricsAggregate([result.mimic_test for result in cv_results])
        tudd_test = RegressionMetricsAggregate([result.tudd_test for result in cv_results])
    else:
        mimic_test = ClassificationMetricsAggregate([result.mimic_test for result in cv_results])
        tudd_test = ClassificationMetricsAggregate([result.tudd_test for result in cv_results])

    return AggregatedFinalTestMetrics(
        mimic_test=mimic_test,
        tudd_test=tudd_test,
        mimic_prediction_time=float(np.mean([result.mimic_prediction_time for result in cv_results])),
        tudd_prediction_time=float(np.mean([result.tudd_prediction_time for result in cv_results])),
    )
