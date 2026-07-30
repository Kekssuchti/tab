import numpy as np

from src.schemas.metrics import RegressionMetrics


def regression_metrics(
    predictions: np.ndarray,
    true_values: np.ndarray,
) -> RegressionMetrics:
    predictions, true_values = _regression_arrays(predictions, true_values)
    errors = true_values - predictions
    squared_errors = errors**2
    mse = np.mean(squared_errors)
    mae = np.mean(np.abs(errors))
    r2 = _r2_score(
        np.sum(squared_errors),
        np.sum((true_values - np.mean(true_values)) ** 2),
    )

    return RegressionMetrics(
        rmse=float(np.sqrt(mse)),
        mae=float(mae),
        mse=float(mse),
        r2=float(r2),
    )


def bootstrap_scores_regression(
    predictions: np.ndarray,
    true_values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Evaluate bootstrap samples in vectorized, memory-bounded batches."""
    predictions, true_values = _regression_arrays(predictions, true_values)
    n_samples = true_values.size
    errors = true_values - predictions

    # Center once around any fixed value to calculate each sample's target
    # variance without the numerical cancellation caused by large targets.
    centered_true = true_values - np.mean(true_values)

    batch_size = 250  # found through testing
    output = np.empty((4, n_bootstrap))
    r2, mae, mse, rmse = output

    for start in range(0, n_bootstrap, batch_size):
        stop = min(start + batch_size, n_bootstrap)
        size = stop - start
        sampled_indices = rng.integers(
            n_samples,
            size=(size, n_samples),
            dtype=np.int32,
        )

        sampled_errors = errors[sampled_indices]
        squared_error_sums = np.einsum("ij,ij->i", sampled_errors, sampled_errors)
        mse[start:stop] = squared_error_sums / n_samples
        rmse[start:stop] = np.sqrt(mse[start:stop])
        mae[start:stop] = np.mean(np.abs(sampled_errors), axis=1)
        del sampled_errors

        sampled_true = centered_true[sampled_indices]
        centered_true_sums = np.sum(sampled_true, axis=1)
        total_sum_squares = np.einsum("ij,ij->i", sampled_true, sampled_true) - centered_true_sums**2 / n_samples
        # Roundoff can make a theoretically zero variance slightly negative.
        total_sum_squares = np.maximum(total_sum_squares, 0.0)
        r2[start:stop] = _r2_score(squared_error_sums, total_sum_squares)

    return output


def _regression_arrays(
    predictions: np.ndarray,
    true_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.asarray(predictions, dtype=float).ravel()
    true_values = np.asarray(true_values, dtype=float).ravel()
    if predictions.size != true_values.size:
        raise ValueError(
            "Regression prediction count does not match true_values: "
            f"got {predictions.size} predictions for {true_values.size} values"
        )
    if predictions.size == 0:
        raise ValueError("Regression predictions and true_values must not be empty")
    if not np.isfinite(predictions).all() or not np.isfinite(true_values).all():
        raise ValueError("Regression predictions and true_values must be finite")
    return predictions, true_values


def _r2_score(
    residual_sum_squares: np.ndarray | float,
    total_sum_squares: np.ndarray | float,
) -> np.ndarray:
    """Calculate sklearn-compatible finite R2 values from sums of squares."""
    residual_sum_squares = np.asarray(residual_sum_squares)
    total_sum_squares = np.asarray(total_sum_squares)
    r2 = np.ones_like(residual_sum_squares + total_sum_squares, dtype=float)
    nonzero_residual = residual_sum_squares != 0
    nonzero_total = total_sum_squares != 0
    valid = nonzero_residual & nonzero_total
    r2[valid] = 1.0 - residual_sum_squares[valid] / total_sum_squares[valid]
    r2[nonzero_residual & ~nonzero_total] = 0.0
    return r2
