import numpy as np

from src.config import config
from src.schemas.base_schemas import TaskType
from src.schemas.dataset_schemas import DatasetBundle
from src.schemas.metrics import (
    BootstrapClassificationMetrics,
    BootstrapFinalTestMetrics,
    BootstrapRegressionMetrics,
)
from src.utils.evaluation_utils import (
    evaluate_bootstrap_classification,
    evaluate_bootstrap_regression,
)


def _evaluate_bootstrap_classification(
    predictions: np.ndarray,
    y_true: np.ndarray,
    n_bootstrap: int = 10000,
    rng: np.random.Generator | None = None,
) -> BootstrapClassificationMetrics:
    rng = rng or np.random.default_rng()
    metrics, lower, upper = evaluate_bootstrap_classification(
        predictions,
        y_true,
        n_bootstrap,
        rng,
    )
    roc_auc_lower, prc_auc_lower, f1_lower, accuracy_lower, sensitivity_lower, precision_lower = lower
    roc_auc_upper, prc_auc_upper, f1_upper, accuracy_upper, sensitivity_upper, precision_upper = upper

    return BootstrapClassificationMetrics(
        metrics=metrics,
        ci_95_roc_auc_lower=float(roc_auc_lower),
        ci_95_roc_auc_upper=float(roc_auc_upper),
        ci_95_prc_auc_lower=float(prc_auc_lower),
        ci_95_prc_auc_upper=float(prc_auc_upper),
        ci_95_f1_lower=float(f1_lower),
        ci_95_f1_upper=float(f1_upper),
        ci_95_accuracy_lower=float(accuracy_lower),
        ci_95_accuracy_upper=float(accuracy_upper),
        ci_95_sensitivity_lower=float(sensitivity_lower),
        ci_95_sensitivity_upper=float(sensitivity_upper),
        ci_95_precision_lower=float(precision_lower),
        ci_95_precision_upper=float(precision_upper),
        n_bootstrap=n_bootstrap,
    )


def _evaluate_bootstrap_regression(
    predictions: np.ndarray,
    y_true: np.ndarray,
    n_bootstrap: int = 10000,
    rng: np.random.Generator | None = None,
) -> BootstrapRegressionMetrics:
    rng = rng or np.random.default_rng()
    metrics, lower, upper = evaluate_bootstrap_regression(
        predictions,
        y_true,
        n_bootstrap,
        rng,
    )
    r2_lower, mae_lower, mse_lower, rmse_lower = lower
    r2_upper, mae_upper, mse_upper, rmse_upper = upper

    return BootstrapRegressionMetrics(
        metrics=metrics,
        ci_95_r2_lower=float(r2_lower),
        ci_95_r2_upper=float(r2_upper),
        ci_95_mae_lower=float(mae_lower),
        ci_95_mae_upper=float(mae_upper),
        ci_95_mse_lower=float(mse_lower),
        ci_95_mse_upper=float(mse_upper),
        ci_95_rmse_lower=float(rmse_lower),
        ci_95_rmse_upper=float(rmse_upper),
        n_bootstrap=n_bootstrap,
    )


def evaluate_trained_model_bootstrap(
    trained_model,
    task_type: TaskType,
    data: DatasetBundle,
    n_bootstrap: int = 10000,
    random_state: int | None = config.seed,
) -> BootstrapFinalTestMetrics:
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

    return BootstrapFinalTestMetrics(
        mimic_test=mimic_bootstrap_metrics,
        mimic_prediction_time=mimic_prediction.seconds,
        tudd_test=tudd_bootstrap_metrics,
        tudd_prediction_time=tudd_prediction.seconds,
    )
