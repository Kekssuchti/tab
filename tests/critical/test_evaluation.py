"""Scientific checks that an end-to-end run's plausible scores cannot establish."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import prediction_metrics
from src.utils.evaluation_utils import evaluate_classification_predictions

METRICS = ("roc_auc", "prc_auc", "f1", "accuracy", "sensitivity", "precision")
N_DRAWS = 41


def _tables():
    # Interleaved, imbalanced labels; ties cross classes and include the threshold.
    labels = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    a = np.array([.1, .7, .5, .2, .5, .8, .1, .3, .8, .2, .4, .5])
    b = np.array([.6, .3, .5, .2, .7, .8, .4, .3, .2, .7, .4, .5])
    return {
        "mimic": pd.DataFrame({"y_true": labels, "y_pred_a": a, "y_pred_b": b, "y_pred_copy": a}),
        "tudd": pd.DataFrame({"y_true": labels, "y_pred_a": b, "y_pred_b": a, "y_pred_copy": b}),
    }


def _oracle(labels, scores):
    # The public contract uses > .5, not >= .5 (equivalent to binary argmax).
    predicted = scores > .5
    return np.array([
        roc_auc_score(labels, scores),
        average_precision_score(labels, scores),
        f1_score(labels, predicted, zero_division=0),
        accuracy_score(labels, predicted),
        recall_score(labels, predicted, zero_division=0),
        precision_score(labels, predicted, zero_division=0),
    ])


@pytest.fixture
def recorded_evaluation(monkeypatch):
    """Observe real RNG draws at the existing evaluator boundary, never replace scores."""
    real_evaluate = prediction_metrics.evaluate_bootstrap_classification
    calls = []

    class RecordingRng:
        def __init__(self, rng):
            self.rng = rng
            self.draws = []

        def integers(self, *args, **kwargs):
            values = self.rng.integers(*args, **kwargs)
            self.draws.append(values.copy())
            return values

    def observe(probabilities, labels, n_bootstrap, rng):
        recorder = RecordingRng(rng)
        result = real_evaluate(probabilities, labels, n_bootstrap, recorder)
        negative, positive = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
        assert len(recorder.draws) == 2 * n_bootstrap
        indices = np.array([
            np.r_[negative[neg], positive[pos]]
            for neg, pos in zip(recorder.draws[::2], recorder.draws[1::2], strict=True)
        ])
        calls.append((probabilities[:, 1].copy(), labels.copy(), indices, result[3].copy()))
        return result

    monkeypatch.setattr(prediction_metrics, "evaluate_bootstrap_classification", observe)

    def run(tables, seed=17):
        calls.clear()
        result = prediction_metrics.evaluate_classification_models(tables, n_bootstrap=N_DRAWS, random_state=seed)
        rows = result.metrics.loc[result.metrics.scope.eq("test")]
        captured = {}
        for row, (scores, labels, indices, bootstrap) in zip(rows.itertuples(), calls, strict=True):
            table = tables[row.dataset]
            np.testing.assert_array_equal(scores, table[row.prediction_column])
            np.testing.assert_array_equal(labels, table.y_true)
            captured[row.model_instance, row.dataset] = (indices, bootstrap)
        return result, captured

    return run


def test_bootstrap_scores_intervals_deltas_and_wins_match_independent_oracles(recorded_evaluation):
    """Catch tied-score ranking errors, reversed comparisons, and plausible-but-wrong CIs."""
    tables = _tables()
    result, captured = recorded_evaluation(tables)
    oracle_scores = {}
    point_scores = {}
    for row in result.metrics.loc[result.metrics.scope.eq("test")].itertuples():
        table = tables[row.dataset]
        labels, probability = table.y_true.to_numpy(), table[row.prediction_column].to_numpy()
        indices, actual_scores = captured[row.model_instance, row.dataset]
        expected = np.array([_oracle(labels[draw], probability[draw]) for draw in indices]).T
        oracle_scores[row.model_instance, row.dataset] = expected
        point_scores[row.model_instance, row.dataset] = _oracle(labels, probability)
        np.testing.assert_allclose(actual_scores, expected, atol=1e-14)
        np.testing.assert_allclose([getattr(row, metric) for metric in METRICS], _oracle(labels, probability))
        bounds = np.percentile(expected, [2.5, 97.5], axis=1)
        np.testing.assert_allclose([getattr(row, f"{metric}_ci_lower") for metric in METRICS], bounds[0])
        np.testing.assert_allclose([getattr(row, f"{metric}_ci_upper") for metric in METRICS], bounds[1])
        for metric_index, metric in enumerate(METRICS[:2]):
            wide = result.bootstrap_scores.query("dataset == @row.dataset and metric == @metric")
            np.testing.assert_array_equal(wide.bootstrap_id, np.arange(N_DRAWS))
            np.testing.assert_allclose(wide[row.model_instance], expected[metric_index])

    for row in result.metrics.loc[result.metrics.scope.eq("test_delta")].itertuples():
        assert row.dataset == "mimic_minus_tudd"
        expected = point_scores[row.model_instance, "mimic"] - point_scores[row.model_instance, "tudd"]
        np.testing.assert_allclose([getattr(row, metric) for metric in METRICS], expected)
        assert all(pd.isna(getattr(row, f"{metric}_ci_lower")) for metric in METRICS)

    assert set(result.pairwise_wins) == {f"{dataset}_{metric}" for dataset in tables for metric in METRICS[:2]}
    for dataset in tables:
        for metric_index, metric in enumerate(METRICS[:2]):
            wins = result.pairwise_wins[f"{dataset}_{metric}"]
            assert list(wins.index) == list(wins.columns) == ["a", "b", "copy"]
            for left in wins.index:
                for right in wins.columns:
                    if left == right:
                        assert pd.isna(wins.loc[left, right])
                        continue
                    l_scores = oracle_scores[left, dataset][metric_index]
                    r_scores = oracle_scores[right, dataset][metric_index]
                    expected = sum(1 if l > r else .5 if l == r else 0 for l, r in zip(l_scores, r_scores))
                    assert wins.loc[left, right] == expected
                    assert wins.loc[left, right] + wins.loc[right, left] == N_DRAWS
            assert wins.loc["a", "copy"] == N_DRAWS / 2
            winner, loser = ("a", "b") if dataset == "mimic" else ("b", "a")
            assert wins.loc[winner, loser] > N_DRAWS / 2


def test_bootstrap_samples_are_paired_reproducible_and_model_order_independent(recorded_evaluation):
    """Inspect sampled row identities: equal duplicate scores alone cannot prove pairing."""
    tables = _tables()
    baseline, original = recorded_evaluation(tables)
    repeated, repeated_samples = recorded_evaluation(tables)
    pd.testing.assert_frame_equal(baseline.bootstrap_scores, repeated.bootstrap_scores)
    for dataset, table in tables.items():
        shared = original["a", dataset][0]
        for model in ("a", "b", "copy"):
            np.testing.assert_array_equal(original[model, dataset][0], shared)
            np.testing.assert_array_equal(repeated_samples[model, dataset][0], shared)
        # Sampling is stratified with replacement, and these are source-row positions.
        np.testing.assert_array_equal(table.y_true.to_numpy()[shared].sum(axis=1), np.full(N_DRAWS, 3))
        assert any(len(set(draw)) < len(draw) for draw in shared)

    reordered = {name: table[["y_true", "y_pred_copy", "y_pred_b", "y_pred_a"]] for name, table in tables.items()}
    added = {name: table.assign(y_pred_extra=np.linspace(.05, .95, len(table))) for name, table in reordered.items()}
    for variant in (reordered, added):
        result, samples = recorded_evaluation(variant)
        for key, (indices, scores) in original.items():
            np.testing.assert_array_equal(samples[key][0], indices)
            np.testing.assert_array_equal(samples[key][1], scores)
        pd.testing.assert_frame_equal(result.bootstrap_scores[baseline.bootstrap_scores.columns], baseline.bootstrap_scores)

    _, changed = recorded_evaluation(tables, seed=29)
    for dataset in tables:
        assert not np.array_equal(original["a", dataset][0], changed["a", dataset][0])


def test_half_probability_is_negative_and_no_positive_predictions_have_zero_precision():
    """Pin down the decision boundary and zero denominators, absent from ordinary E2Es."""
    labels = np.array([0, 0, 1, 0, 1])
    scores = np.array([.1, .5, .5, .2, .3])
    result = evaluate_classification_predictions(np.column_stack((1 - scores, scores)), labels)
    np.testing.assert_allclose([result.scores[metric] for metric in METRICS], _oracle(labels, scores))
    assert result.f1 == result.precision == result.sensitivity == 0
    assert result.accuracy == .6


@pytest.mark.parametrize("probabilities, message", [
    ([[1.1, -.1], [.2, .8]], "between 0 and 1"),
    ([[.8, .3], [.2, .8]], "sum to 1"),
    ([[np.nan, np.nan], [.2, .8]], "finite"),
])
def test_invalid_adapter_probabilities_are_not_reported_as_valid_scores(probabilities, message):
    with pytest.raises(ValueError, match=message):
        evaluate_classification_predictions(np.array(probabilities), np.array([0, 1]))
