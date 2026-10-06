"""Benchmark and compare the deterministic classification bootstrap evaluator.

This uses synthetic labels and predictions only. It exercises the production
metrics for four models and stores complete outputs so later implementations
can be checked bit-for-bit against a baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.prediction_metrics import ClassificationModelEvaluation, evaluate_classification_models

N_SAMPLES = 4_095
N_BOOTSTRAP = 10_000
DATA_SEED = 20_261_003
BOOTSTRAP_SEED = 1_337
MODEL_COLUMNS = ("y_pred_linear", "y_pred_noisy", "y_pred_tied", "y_pred_inverse")


def benchmark_tables() -> dict[str, pd.DataFrame]:
    """Create one deterministic, imbalanced cohort with four distinct models."""
    rng = np.random.default_rng(DATA_SEED)
    latent = rng.normal(size=N_SAMPLES)
    labels = (latent + rng.normal(scale=1.35, size=N_SAMPLES) > 1.25).astype(np.int8)

    def logistic(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-values))

    linear = logistic(1.15 * latent + rng.normal(scale=0.65, size=N_SAMPLES) - 1.0)
    noisy = logistic(0.65 * latent + rng.normal(scale=1.1, size=N_SAMPLES) - 0.8)
    tied = np.round(logistic(0.9 * latent - 0.9), 2)
    inverse = logistic(-0.35 * latent + rng.normal(scale=0.9, size=N_SAMPLES) - 0.7)
    return {
        "benchmark": pd.DataFrame(
            {
                "y_true": labels,
                "y_pred_linear": linear,
                "y_pred_noisy": noisy,
                "y_pred_tied": tied,
                "y_pred_inverse": inverse,
            }
        )
    }


def evaluation_payload(evaluation: ClassificationModelEvaluation) -> dict[str, object]:
    return {
        "metrics": evaluation.metrics,
        "bootstrap_scores": evaluation.bootstrap_scores,
        "pairwise_wins": evaluation.pairwise_wins,
    }


def assert_payload_equal(actual: dict[str, object], expected: dict[str, object]) -> None:
    pd.testing.assert_frame_equal(actual["metrics"], expected["metrics"], check_exact=True)
    pd.testing.assert_frame_equal(actual["bootstrap_scores"], expected["bootstrap_scores"], check_exact=True)
    actual_wins = actual["pairwise_wins"]
    expected_wins = expected["pairwise_wins"]
    assert isinstance(actual_wins, dict) and isinstance(expected_wins, dict)
    if actual_wins.keys() != expected_wins.keys():
        raise AssertionError(f"Pairwise matrix keys differ: {actual_wins.keys()} != {expected_wins.keys()}")
    for key in actual_wins:
        pd.testing.assert_frame_equal(actual_wins[key], expected_wins[key], check_exact=True)


def payload_digest(payload: dict[str, object]) -> str:
    return hashlib.sha256(pickle.dumps(payload, protocol=5)).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()

    tables = benchmark_tables()
    timings = []
    payload = None
    for _ in range(args.repeats):
        started = time.perf_counter()
        evaluation = evaluate_classification_models(
            tables,
            n_bootstrap=args.n_bootstrap,
            random_state=BOOTSTRAP_SEED,
            n_jobs=args.n_jobs,
        )
        timings.append(time.perf_counter() - started)
        current = evaluation_payload(evaluation)
        if payload is not None:
            assert_payload_equal(current, payload)
        payload = current

    assert payload is not None
    comparison = "not_requested"
    if args.reference is not None:
        with args.reference.open("rb") as handle:
            reference = pickle.load(handle)
        assert_payload_equal(payload, reference)
        comparison = "bitwise_identical"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(payload, handle, protocol=5)

    metrics = payload["metrics"]
    assert isinstance(metrics, pd.DataFrame)
    summary = {
        "label": args.label,
        "samples": N_SAMPLES,
        "models": list(MODEL_COLUMNS),
        "bootstrap_iterations": args.n_bootstrap,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_jobs": args.n_jobs,
        "repeats_seconds": timings,
        "median_seconds": float(np.median(timings)),
        "comparison_to_reference": comparison,
        "payload_sha256": payload_digest(payload),
        "point_metrics": metrics.loc[
            metrics["scope"].eq("test"),
            ["model_instance", "roc_auc", "prc_auc", "f1", "accuracy", "sensitivity", "precision"],
        ].to_dict(orient="records"),
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
