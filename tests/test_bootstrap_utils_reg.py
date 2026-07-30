import numpy as np
from numpy.testing import assert_allclose
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

from src.utils.bootstrap_utils_reg import bootstrap_scores_regression, regression_metrics


def test_regression_metrics_match_sklearn():
    predictions = np.array([0.5, 2.5, 2.0, 4.5])
    true_values = np.array([1.0, 2.0, 3.0, 4.0])

    metrics = regression_metrics(predictions, true_values)

    assert_allclose(metrics.r2, r2_score(true_values, predictions))
    assert_allclose(metrics.mae, mean_absolute_error(true_values, predictions))
    assert_allclose(metrics.mse, mean_squared_error(true_values, predictions))
    assert_allclose(metrics.rmse, root_mean_squared_error(true_values, predictions))


def test_bootstrap_scores_regression_match_materialized_resamples_across_batches():
    predictions = np.array([1.2, 1.8, 3.1, 3.9, 5.2, 5.8])
    true_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    n_bootstrap = 503

    scores = bootstrap_scores_regression(
        predictions,
        true_values,
        n_bootstrap,
        np.random.default_rng(17),
    )

    reference_rng = np.random.default_rng(17)
    expected = np.empty_like(scores)
    for start in range(0, n_bootstrap, 250):
        stop = min(start + 250, n_bootstrap)
        sampled_indices = reference_rng.integers(
            true_values.size,
            size=(stop - start, true_values.size),
            dtype=np.int32,
        )
        for offset, indices in enumerate(sampled_indices):
            sample_predictions = predictions[indices]
            sample_true = true_values[indices]
            expected[:, start + offset] = [
                r2_score(sample_true, sample_predictions),
                mean_absolute_error(sample_true, sample_predictions),
                mean_squared_error(sample_true, sample_predictions),
                root_mean_squared_error(sample_true, sample_predictions),
            ]

    assert scores.shape == (4, n_bootstrap)
    assert_allclose(scores, expected, rtol=1e-12, atol=1e-12)
