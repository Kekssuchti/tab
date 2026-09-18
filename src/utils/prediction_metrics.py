from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import config
from src.schemas.metrics import calculate_metric_diff
from src.utils.evaluation_utils import evaluate_bootstrap_classification
from src.utils.prediction_tables import (
    PREDICTION_COLUMN_PREFIX,
    PREDICTION_DATASETS,
    Y_TRUE_COLUMN,
    is_prediction_column,
)

CLASSIFICATION_METRICS = ("roc_auc", "prc_auc", "f1", "accuracy", "sensitivity", "precision")
PAIRWISE_METRICS = ("roc_auc", "prc_auc")
BOOTSTRAP_METADATA_COLUMNS = ("dataset", "metric", "bootstrap_id")
RESULT_COLUMNS = (
    "model_instance",
    "prediction_column",
    "scope",
    "dataset",
    "statistic",
    "ci_level",
    "n_bootstrap",
    *(column for metric in CLASSIFICATION_METRICS for column in (metric, f"{metric}_ci_lower", f"{metric}_ci_upper")),
)
from src.utils.logger import logger


@dataclass(frozen=True)
class ClassificationModelEvaluation:
    metrics: pd.DataFrame
    bootstrap_scores: pd.DataFrame
    pairwise_wins: dict[str, pd.DataFrame]


def evaluate_classification_models(
    tables: Mapping[str, pd.DataFrame],
    *,
    n_bootstrap: int = 10_000,
    random_state: int | None = config.seed,
) -> ClassificationModelEvaluation:
    """Evaluate every model and compare paired bootstrap AUROC/AUPRC scores."""
    logger.info("Evaluating classification models")

    model_columns = [column for column in tables["mimic"] if is_prediction_column(column)]
    bootstrap_seeds = _dataset_bootstrap_seeds(random_state)
    bootstrap_by_dataset = {dataset: {metric: {} for metric in PAIRWISE_METRICS} for dataset in PREDICTION_DATASETS}
    rows = []

    for column in model_columns:
        model_instance = column.removeprefix(PREDICTION_COLUMN_PREFIX)
        point_metrics = {}
        for dataset in PREDICTION_DATASETS:
            table = tables[dataset]
            probability = table[column].to_numpy(dtype=float)
            metrics, lower, upper, bootstrap = evaluate_bootstrap_classification(
                np.column_stack((1 - probability, probability)),
                table[Y_TRUE_COLUMN].to_numpy(dtype=int),
                n_bootstrap,
                np.random.default_rng(bootstrap_seeds[dataset]),
            )
            point_metrics[dataset] = metrics
            for metric in PAIRWISE_METRICS:
                metric_index = CLASSIFICATION_METRICS.index(metric)
                bootstrap_by_dataset[dataset][metric][model_instance] = bootstrap[metric_index]

            row = _result_row(column, "test", dataset, "point", 0.95, n_bootstrap)
            row.update(metrics.scores)
            for metric, lower_bound, upper_bound in zip(CLASSIFICATION_METRICS, lower, upper, strict=True):
                row[f"{metric}_ci_lower"] = float(lower_bound)
                row[f"{metric}_ci_upper"] = float(upper_bound)
            rows.append(row)

        difference = calculate_metric_diff(point_metrics["mimic"], point_metrics["tudd"])
        row = _result_row(
            column,
            "test_delta",
            "mimic_minus_tudd",
            "difference",
            None,
            None,
        )
        row.update(difference.scores)
        rows.append(row)

    bootstrap_scores = _bootstrap_scores_frame(bootstrap_by_dataset, n_bootstrap)

    return ClassificationModelEvaluation(
        metrics=pd.DataFrame(rows, columns=RESULT_COLUMNS),
        bootstrap_scores=bootstrap_scores,
        pairwise_wins=pairwise_win_matrices(bootstrap_scores),
    )


def pairwise_win_matrices(bootstrap_scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return matrices where each cell counts row-model wins over column-model.

    Ties contribute half a win to each model, so opposing off-diagonal cells
    always sum to the number of bootstrap iterations.
    """
    logger.info("Computing pairwise win matrices")
    model_columns = [column for column in bootstrap_scores.columns if column not in BOOTSTRAP_METADATA_COLUMNS]
    matrices = {}
    for (dataset, metric), scores in bootstrap_scores.groupby(["dataset", "metric"], sort=False):
        values = scores[model_columns].to_numpy()
        wins = (values[:, :, None] > values[:, None, :]).sum(axis=0).astype(float)
        ties = (values[:, :, None] == values[:, None, :]).sum(axis=0)
        wins += 0.5 * ties
        np.fill_diagonal(wins, np.nan)

        matrix = pd.DataFrame(wins, index=model_columns, columns=model_columns)
        matrix.index.name = "model_instance"
        matrices[f"{dataset}_{metric}"] = matrix
    logger.info("Pairwise win matrices computed")
    return matrices


def recompute_classification_metrics(
    tables: Mapping[str, pd.DataFrame],
    *,
    n_bootstrap: int = 10_000,
    random_state: int | None = config.seed,
) -> pd.DataFrame:
    """Backward-compatible wrapper returning only the summary metrics table."""
    return evaluate_classification_models(
        tables,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    ).metrics


def _dataset_bootstrap_seeds(random_state: int | None) -> dict[str, int]:
    rng = np.random.default_rng(random_state)
    return {dataset: int(rng.integers(np.iinfo(np.uint64).max, dtype=np.uint64)) for dataset in PREDICTION_DATASETS}


def _bootstrap_scores_frame(
    bootstrap_by_dataset: dict[str, dict[str, dict[str, np.ndarray]]],
    n_bootstrap: int,
) -> pd.DataFrame:
    frames = []
    for dataset in PREDICTION_DATASETS:
        for metric in PAIRWISE_METRICS:
            frame = pd.DataFrame(bootstrap_by_dataset[dataset][metric])
            frame.insert(0, "bootstrap_id", np.arange(n_bootstrap))
            frame.insert(0, "metric", metric)
            frame.insert(0, "dataset", dataset)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _result_row(
    prediction_column: str,
    scope: str,
    dataset: str,
    statistic: str,
    ci_level: float | None,
    n_bootstrap: int | None,
) -> dict[str, object]:
    return {
        "model_instance": prediction_column.removeprefix(PREDICTION_COLUMN_PREFIX),
        "prediction_column": prediction_column,
        "scope": scope,
        "dataset": dataset,
        "statistic": statistic,
        "ci_level": ci_level,
        "n_bootstrap": n_bootstrap,
    }
