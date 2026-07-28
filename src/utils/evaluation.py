import numpy as np

from src.config import config
from src.schemas.base_schemas import TaskType
from src.schemas.dataset_schemas import DatasetBundle, XYDataset
from src.schemas.metrics import (
    AggregatedFinalTestMetrics,
    BootstrapClassificationMetrics,
    BootstrapRegressionMetrics,
    FinalTestMetrics,
)
from src.schemas.run_records import TestSetEvaluationRecord
from src.utils.evaluation_utils import (
    classification_prediction_batch,
    evaluate_classification_predictions,
    evaluate_regression_predictions,
)
from src.utils.logger import logger


def evaluate_trained_model(
    trained_model,
    task_type: TaskType,
    data: DatasetBundle,
) -> FinalTestMetrics:
    """
    Runs standard evaluation on both mimic and tudd test set with full metrics

    Returns:
        FinalTestMetrics: metrics of tested model
    """
    mimic_test_results = _evaluate_test_set("mimic", trained_model, task_type, data.test_mimic)
    tudd_test_results = _evaluate_test_set("tudd", trained_model, task_type, data.test_tudd)

    return FinalTestMetrics(
        mimic_test=mimic_test_results.metrics,
        mimic_prediction_time=mimic_test_results.predict_time,
        tudd_test=tudd_test_results.metrics,
        tudd_prediction_time=tudd_test_results.predict_time,
    )


def _evaluate_test_set(
    dataset_name: str,
    trained_model,
    task_type: TaskType,
    test_set: XYDataset,
) -> TestSetEvaluationRecord:
    prediction = trained_model.predict(test_set.X)
    if task_type == "classification":
        metrics = evaluate_classification_predictions(prediction.values, test_set.y.to_numpy())
    else:
        metrics = evaluate_regression_predictions(prediction.values, test_set.y.to_numpy())

    return TestSetEvaluationRecord(
        dataset_name=dataset_name,
        metrics=metrics,
        predict_time=prediction.seconds,
    )


def _evaluate_bootstrap_classification(
    predictions: np.ndarray,
    y_true: np.ndarray,
    n_bootstrap: int = 5000,
    rng: np.random.Generator | None = None,
) -> BootstrapClassificationMetrics:
    logger.info("Started Bootstrap Classification Eval")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")

    metrics = evaluate_classification_predictions(predictions, y_true)
    batch = classification_prediction_batch(predictions, y_true)
    rng = rng or np.random.default_rng()
    indices_by_class = [np.flatnonzero(batch.y_true == label) for label in range(batch.n_classes)]

    bootstrap_scores = {
        "roc_auc": np.empty(n_bootstrap),
        "prc_auc": np.empty(n_bootstrap),
        "f1": np.empty(n_bootstrap),
        "accuracy": np.empty(n_bootstrap),
        "sensitivity": np.empty(n_bootstrap),
        "precision": np.empty(n_bootstrap),
    }

    for i in range(n_bootstrap):
        sampled_indices = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in indices_by_class]
        )
        sampled_metrics = evaluate_classification_predictions(
            batch.probabilities[sampled_indices],
            batch.y_true[sampled_indices],
        )
        for name, values in bootstrap_scores.items():
            score = sampled_metrics.scores.get(name)
            if score is None:
                raise ValueError(f"Bootstrap metric {name!r} is unavailable")
            values[i] = score

    confidence_intervals = {name: _percentile_ci(values) for name, values in bootstrap_scores.items()}

    logger.info("Finished Bootstrap Classification Eval")
    return BootstrapClassificationMetrics(
        metrics=metrics,
        ci_95_roc_auc_lower=confidence_intervals["roc_auc"][0],
        ci_95_roc_auc_upper=confidence_intervals["roc_auc"][1],
        ci_95_prc_auc_lower=confidence_intervals["prc_auc"][0],
        ci_95_prc_auc_upper=confidence_intervals["prc_auc"][1],
        ci_95_f1_lower=confidence_intervals["f1"][0],
        ci_95_f1_upper=confidence_intervals["f1"][1],
        ci_95_accuracy_lower=confidence_intervals["accuracy"][0],
        ci_95_accuracy_upper=confidence_intervals["accuracy"][1],
        ci_95_sensitivity_lower=confidence_intervals["sensitivity"][0],
        ci_95_sensitivity_upper=confidence_intervals["sensitivity"][1],
        ci_95_precision_lower=confidence_intervals["precision"][0],
        ci_95_precision_upper=confidence_intervals["precision"][1],
        n_bootstrap=n_bootstrap,
    )


def _evaluate_bootstrap_regression(
    predictions: np.ndarray,
    y_true: np.ndarray,
    n_bootstrap: int = 5000,
    rng: np.random.Generator | None = None,
) -> BootstrapRegressionMetrics:
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")

    predictions = np.asarray(predictions)
    y_true = np.asarray(y_true)
    metrics = evaluate_regression_predictions(predictions=predictions, true_values=y_true)
    rng = rng or np.random.default_rng()

    bootstrap_scores = {
        "r2": np.empty(n_bootstrap),
        "mae": np.empty(n_bootstrap),
        "mse": np.empty(n_bootstrap),
        "rmse": np.empty(n_bootstrap),
    }

    for i in range(n_bootstrap):
        indices = rng.choice(len(predictions), len(predictions), replace=True)
        sampled_metrics = evaluate_regression_predictions(predictions[indices], y_true[indices])
        for name, values in bootstrap_scores.items():
            values[i] = sampled_metrics.scores[name]

    confidence_intervals = {name: _percentile_ci(values) for name, values in bootstrap_scores.items()}

    return BootstrapRegressionMetrics(
        metrics=metrics,
        ci_95_r2_lower=confidence_intervals["r2"][0],
        ci_95_r2_upper=confidence_intervals["r2"][1],
        ci_95_mae_lower=confidence_intervals["mae"][0],
        ci_95_mae_upper=confidence_intervals["mae"][1],
        ci_95_mse_lower=confidence_intervals["mse"][0],
        ci_95_mse_upper=confidence_intervals["mse"][1],
        ci_95_rmse_lower=confidence_intervals["rmse"][0],
        ci_95_rmse_upper=confidence_intervals["rmse"][1],
        n_bootstrap=n_bootstrap,
    )


def evaluate_trained_model_bootstrap(
    trained_model,
    task_type: TaskType,
    data: DatasetBundle,
    n_bootstrap: int = 5000,
    random_state: int | None = config.seed,
) -> AggregatedFinalTestMetrics:
    mimic_prediction = trained_model.predict(data.test_mimic.X)
    tudd_prediction = trained_model.predict(data.test_tudd.X)
    rng = np.random.default_rng(random_state)

    if task_type == "classification":
        mimic_bootstrap_metrics = _evaluate_bootstrap_classification(
            predictions=mimic_prediction.values,
            y_true=data.test_mimic.y.to_numpy(),
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        tudd_bootstrap_metrics = _evaluate_bootstrap_classification(
            predictions=tudd_prediction.values,
            y_true=data.test_tudd.y.to_numpy(),
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
    else:
        mimic_bootstrap_metrics = _evaluate_bootstrap_regression(
            predictions=mimic_prediction.values,
            y_true=data.test_mimic.y.to_numpy(),
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        tudd_bootstrap_metrics = _evaluate_bootstrap_regression(
            predictions=tudd_prediction.values,
            y_true=data.test_tudd.y.to_numpy(),
            n_bootstrap=n_bootstrap,
            rng=rng,
        )

    return AggregatedFinalTestMetrics(
        mimic_test=mimic_bootstrap_metrics,
        mimic_prediction_time=mimic_prediction.seconds,
        tudd_test=tudd_bootstrap_metrics,
        tudd_prediction_time=tudd_prediction.seconds,
    )


def _percentile_ci(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.percentile(values, [2.5, 97.5])
    return float(lower), float(upper)
