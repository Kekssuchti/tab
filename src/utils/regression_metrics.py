import numpy as np

from src.schemas.metrics import RegressionMetrics


def regression_metrics(
    predictions: np.ndarray,
    true_values: np.ndarray,
) -> RegressionMetrics:
    predictions = np.asarray(predictions, dtype=float).ravel()
    true_values = np.asarray(true_values, dtype=float).ravel()
    if predictions.size != true_values.size:
        raise ValueError(
            "Regression prediction count does not match true values: "
            f"got {predictions.size} predictions for {true_values.size} values"
        )
    if predictions.size == 0:
        raise ValueError("Regression predictions and true values must not be empty")
    if not np.isfinite(predictions).all() or not np.isfinite(true_values).all():
        raise ValueError("Regression predictions and true values must be finite")

    errors = true_values - predictions
    squared_errors = errors**2
    mse = np.mean(squared_errors)
    residual_sum_squares = np.sum(squared_errors)
    total_sum_squares = np.sum((true_values - np.mean(true_values)) ** 2)
    if total_sum_squares == 0:
        r2 = 1.0 if residual_sum_squares == 0 else 0.0
    else:
        r2 = 1.0 - residual_sum_squares / total_sum_squares

    return RegressionMetrics(
        rmse=float(np.sqrt(mse)),
        mae=float(np.mean(np.abs(errors))),
        mse=float(mse),
        r2=float(r2),
    )
