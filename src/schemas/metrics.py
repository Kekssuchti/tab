from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar, cast, overload

import numpy as np


@dataclass
class ClassificationMetrics:
    """
    Classification metrics for one prediction set.

    ---
    Attributes:
        roc_auc: float or None
            ROC AUC, or None when unavailable.

        prc_auc: float or None
            Precision-recall AUC, or None when unavailable.

        f1: float
            F1 score.

        accuracy: float
            Accuracy score.

        sensitivity: float
            True positive rate.

        precision: float
            Positive predictive value.

        confusion_matrix: numpy.ndarray
            Confusion matrix for predicted labels.

        n_classes: int
            Number of classes observed by the evaluator.
    """

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
class ClassificationMetricsAggregate:
    """
    Mean classification metrics with 95% confidence intervals.

    ---
    Attributes:
        metrics: list of ClassificationMetrics
            Per-run metrics to aggregate.
    """

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

        self.mean_roc_auc, self.ci_95_roc_auc_lower, self.ci_95_roc_auc_upper = calculate_mean_ci(
            [m.roc_auc for m in metrics if m.roc_auc is not None], 0.95
        )
        self.mean_prc_auc, self.ci_95_prc_auc_lower, self.ci_95_prc_auc_upper = calculate_mean_ci(
            [m.prc_auc for m in metrics if m.prc_auc is not None], 0.95
        )
        self.mean_f1, self.ci_95_f1_lower, self.ci_95_f1_upper = calculate_mean_ci([m.f1 for m in metrics], 0.95)
        self.mean_accuracy, self.ci_95_accuracy_lower, self.ci_95_accuracy_upper = calculate_mean_ci(
            [m.accuracy for m in metrics], 0.95
        )
        (
            self.mean_sensitivity,
            self.ci_95_sensitivity_lower,
            self.ci_95_sensitivity_upper,
        ) = calculate_mean_ci([m.sensitivity for m in metrics], 0.95)
        self.mean_precision, self.ci_95_precision_lower, self.ci_95_precision_upper = calculate_mean_ci(
            [m.precision for m in metrics], 0.95
        )
        self.mean_confusion_matrix = np.mean(np.stack([m.confusion_matrix for m in metrics]), axis=0)
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

    @property
    def confidence_intervals(self) -> dict[str, tuple[float, float]]:
        return {
            "roc_auc": (self.ci_95_roc_auc_lower, self.ci_95_roc_auc_upper),
            "prc_auc": (self.ci_95_prc_auc_lower, self.ci_95_prc_auc_upper),
            "f1": (self.ci_95_f1_lower, self.ci_95_f1_upper),
            "accuracy": (self.ci_95_accuracy_lower, self.ci_95_accuracy_upper),
            "sensitivity": (self.ci_95_sensitivity_lower, self.ci_95_sensitivity_upper),
            "precision": (self.ci_95_precision_lower, self.ci_95_precision_upper),
        }


@dataclass(frozen=True)
class ClassificationPredictionBatch:
    """
    Classification predictions and probabilities for one batch.

    ---
    Attributes:
        probabilities: numpy.ndarray
            Predicted class probabilities.

        y_true: numpy.ndarray
            Ground-truth labels.

        y_pred: numpy.ndarray
            Predicted labels.

        n_classes: int
            Number of classes represented in the batch.
    """

    probabilities: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    n_classes: int


@dataclass
class RegressionMetrics:
    """
    Regression metrics for one prediction set.

    ---
    Attributes:
        r2: float
            Coefficient of determination.

        mae: float
            Mean absolute error.

        mse: float
            Mean squared error.
    """

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


@dataclass
class RegressionMetricsAggregate:
    """
    Mean regression metrics with 95% confidence intervals.

    ---
    Attributes:
        metrics: list of RegressionMetrics
            Per-run metrics to aggregate.
    """

    mean_r2: float
    mean_mae: float
    mean_mse: float
    mean_rmse: float
    ci_95_r2_lower: float
    ci_95_r2_upper: float
    ci_95_mae_lower: float
    ci_95_mae_upper: float
    ci_95_mse_lower: float
    ci_95_mse_upper: float
    ci_95_rmse_lower: float
    ci_95_rmse_upper: float

    def __init__(self, metrics: list[RegressionMetrics]) -> None:
        from src.utils.evaluation_utils import calculate_mean_ci

        self.mean_r2, self.ci_95_r2_lower, self.ci_95_r2_upper = calculate_mean_ci([m.r2 for m in metrics])
        self.mean_mae, self.ci_95_mae_lower, self.ci_95_mae_upper = calculate_mean_ci([m.mae for m in metrics])
        self.mean_mse, self.ci_95_mse_lower, self.ci_95_mse_upper = calculate_mean_ci([m.mse for m in metrics])
        self.mean_rmse, self.ci_95_rmse_lower, self.ci_95_rmse_upper = calculate_mean_ci([m.rmse for m in metrics])

    @property
    def scores(self) -> dict[str, float]:
        return {
            "mean_r2": self.mean_r2,
            "mean_mae": self.mean_mae,
            "mean_mse": self.mean_mse,
            "mean_rmse": self.mean_rmse,
        }

    @property
    def confidence_intervals(self) -> dict[str, tuple[float, float]]:
        return {
            "r2": (self.ci_95_r2_lower, self.ci_95_r2_upper),
            "mae": (self.ci_95_mae_lower, self.ci_95_mae_upper),
            "mse": (self.ci_95_mse_lower, self.ci_95_mse_upper),
            "rmse": (self.ci_95_rmse_lower, self.ci_95_rmse_upper),
        }


MetricT = TypeVar("MetricT", ClassificationMetrics, RegressionMetrics)
AggregateMetricT = TypeVar("AggregateMetricT", ClassificationMetricsAggregate, RegressionMetricsAggregate)


@dataclass(frozen=True)
class FinalTestMetrics(Generic[MetricT]):
    """
    Final classification metrics for both held-out test sets.

    ---
    Attributes:
        mimic_test: ClassificationMetrics | RegressionMetrics
            Metrics on the MIMIC test set.

        mimic_prediction_time: float
            Prediction time on the MIMIC test set, in seconds.

        tudd_test: ClassificationMetrics | RegressionMetrics
            Metrics on the TUDD test set.

        tudd_prediction_time: float
            Prediction time on the TUDD test set, in seconds.
    """

    mimic_test: MetricT
    mimic_prediction_time: float
    tudd_test: MetricT
    tudd_prediction_time: float

    @property
    def mimic_minus_tudd(self) -> MetricT:
        return cast(MetricT, calculate_metric_diff(self.mimic_test, self.tudd_test))


ClassificationFinalTestMetrics: TypeAlias = FinalTestMetrics[ClassificationMetrics]
RegressionFinalTestMetrics: TypeAlias = FinalTestMetrics[RegressionMetrics]


@dataclass(frozen=True)
class AggregatedFinalTestMetrics(Generic[AggregateMetricT]):
    """
    Aggregated final-test metrics for tuned cross-validation runs.

    ---
    Attributes:
        mimic_test: ClassificationMetricsAggregate | RegressionMetricsAggregate
            Aggregated metrics on MIMIC test folds.

        mimic_prediction_time: float
            Mean prediction time on MIMIC test folds, in seconds.

        tudd_test: ClassificationMetricsAggregate | RegressionMetricsAggregate
            Aggregated metrics on TUDD test folds.

        tudd_prediction_time: float
            Mean prediction time on TUDD test folds, in seconds.
    """

    mimic_test: AggregateMetricT
    mimic_prediction_time: float
    tudd_test: AggregateMetricT
    tudd_prediction_time: float

    @property
    def mimic_minus_tudd(self) -> ClassificationMetrics | RegressionMetrics:
        return calculate_metric_diff(self.mimic_test, self.tudd_test)


ClassificationAggregatedFinalTestMetrics: TypeAlias = AggregatedFinalTestMetrics[ClassificationMetricsAggregate]
RegressionAggregatedFinalTestMetrics: TypeAlias = AggregatedFinalTestMetrics[RegressionMetricsAggregate]


@overload
def calculate_metric_diff(
    mimic_metrics: ClassificationMetrics,
    tudd_metrics: ClassificationMetrics,
) -> ClassificationMetrics: ...


@overload
def calculate_metric_diff(
    mimic_metrics: ClassificationMetricsAggregate,
    tudd_metrics: ClassificationMetricsAggregate,
) -> ClassificationMetrics: ...


@overload
def calculate_metric_diff(
    mimic_metrics: RegressionMetrics,
    tudd_metrics: RegressionMetrics,
) -> RegressionMetrics: ...


@overload
def calculate_metric_diff(
    mimic_metrics: RegressionMetricsAggregate,
    tudd_metrics: RegressionMetricsAggregate,
) -> RegressionMetrics: ...


def calculate_metric_diff(
    mimic_metrics: ClassificationMetrics
    | ClassificationMetricsAggregate
    | RegressionMetrics
    | RegressionMetricsAggregate,
    tudd_metrics: ClassificationMetrics
    | ClassificationMetricsAggregate
    | RegressionMetrics
    | RegressionMetricsAggregate,
) -> ClassificationMetrics | RegressionMetrics:
    """
    Calculate the difference between two sets of metrics.
    metrics must be of type: ClassificationMetrics, ClassificationMetricsAggregate, RegressionMetrics, or RegressionMetricsAggregate.
    """

    if isinstance(mimic_metrics, RegressionMetrics) and isinstance(tudd_metrics, RegressionMetrics):
        return RegressionMetrics(
            r2=mimic_metrics.r2 - tudd_metrics.r2,
            mae=mimic_metrics.mae - tudd_metrics.mae,
            mse=mimic_metrics.mse - tudd_metrics.mse,
            rmse=mimic_metrics.rmse - tudd_metrics.rmse,
        )

    if isinstance(mimic_metrics, RegressionMetricsAggregate) and isinstance(tudd_metrics, RegressionMetricsAggregate):
        return RegressionMetrics(
            r2=mimic_metrics.mean_r2 - tudd_metrics.mean_r2,
            mae=mimic_metrics.mean_mae - tudd_metrics.mean_mae,
            mse=mimic_metrics.mean_mse - tudd_metrics.mean_mse,
            rmse=mimic_metrics.mean_rmse - tudd_metrics.mean_rmse,
        )

    if isinstance(mimic_metrics, ClassificationMetrics) and isinstance(tudd_metrics, ClassificationMetrics):
        if mimic_metrics.roc_auc is not None and tudd_metrics.roc_auc is not None:
            roc_auc = mimic_metrics.roc_auc - tudd_metrics.roc_auc
        else:
            roc_auc = None

        if mimic_metrics.prc_auc is not None and tudd_metrics.prc_auc is not None:
            prc_auc = mimic_metrics.prc_auc - tudd_metrics.prc_auc
        else:
            prc_auc = None

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

    if isinstance(mimic_metrics, ClassificationMetricsAggregate) and isinstance(
        tudd_metrics, ClassificationMetricsAggregate
    ):
        return ClassificationMetrics(
            roc_auc=mimic_metrics.mean_roc_auc - tudd_metrics.mean_roc_auc,
            prc_auc=mimic_metrics.mean_prc_auc - tudd_metrics.mean_prc_auc,
            f1=mimic_metrics.mean_f1 - tudd_metrics.mean_f1,
            accuracy=mimic_metrics.mean_accuracy - tudd_metrics.mean_accuracy,
            sensitivity=mimic_metrics.mean_sensitivity - tudd_metrics.mean_sensitivity,
            precision=mimic_metrics.mean_precision - tudd_metrics.mean_precision,
            n_classes=mimic_metrics.n_classes,
            confusion_matrix=None,
        )

    raise ValueError("mimic_metrics and tudd_metrics must be of the same type")
