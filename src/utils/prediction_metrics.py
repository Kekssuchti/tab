from __future__ import annotations

import multiprocessing
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import config
from src.schemas.metrics import ClassificationMetrics, calculate_metric_diff
from src.utils.evaluation_utils import evaluate_bootstrap_classification
from src.utils.prediction_tables import (
    PREDICTION_COLUMN_PREFIX,
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


@dataclass(frozen=True)
class _BootstrapTask:
    ordinal: int
    probability: np.ndarray
    labels: np.ndarray
    n_bootstrap: int
    seed: int


@dataclass(frozen=True)
class _BootstrapTaskResult:
    ordinal: int
    metrics: ClassificationMetrics
    lower: np.ndarray
    upper: np.ndarray
    bootstrap: np.ndarray


def evaluate_classification_models(
    tables: Mapping[str, pd.DataFrame],
    *,
    n_bootstrap: int = 10_000,
    random_state: int | None = config.seed,
    n_jobs: int = 4,
) -> ClassificationModelEvaluation:
    """Evaluate every model and compare paired bootstrap AUROC/AUPRC scores."""
    logger.info("Evaluating classification models")
    datasets = tuple(tables)
    first_table = next(iter(tables.values()))
    model_columns = [column for column in first_table if is_prediction_column(column)]
    bootstrap_seeds = _dataset_bootstrap_seeds(datasets, random_state)
    tasks = []
    task_keys = []
    for column in model_columns:
        for dataset in datasets:
            table = tables[dataset]
            tasks.append(
                _BootstrapTask(
                    ordinal=len(tasks),
                    probability=table[column].to_numpy(dtype=float),
                    labels=table[Y_TRUE_COLUMN].to_numpy(dtype=int),
                    n_bootstrap=n_bootstrap,
                    seed=bootstrap_seeds[dataset],
                )
            )
            task_keys.append((column, dataset))
    task_results = _run_bootstrap_tasks(tasks, n_jobs)
    results_by_key = {task_keys[result.ordinal]: result for result in task_results}

    bootstrap_by_dataset = {dataset: {metric: {} for metric in PAIRWISE_METRICS} for dataset in datasets}
    rows = []
    for column in model_columns:
        model_instance = column.removeprefix(PREDICTION_COLUMN_PREFIX)
        point_metrics = {}
        for dataset in datasets:
            result = results_by_key[column, dataset]
            point_metrics[dataset] = result.metrics
            for metric in PAIRWISE_METRICS:
                metric_index = CLASSIFICATION_METRICS.index(metric)
                bootstrap_by_dataset[dataset][metric][model_instance] = result.bootstrap[metric_index]

            row = _result_row(column, "test", dataset, "point", 0.95, n_bootstrap)
            row.update(result.metrics.scores)
            for metric, lower_bound, upper_bound in zip(
                CLASSIFICATION_METRICS,
                result.lower,
                result.upper,
                strict=True,
            ):
                row[f"{metric}_ci_lower"] = float(lower_bound)
                row[f"{metric}_ci_upper"] = float(upper_bound)
            rows.append(row)

        if "mimic" in point_metrics and "tudd" in point_metrics:
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

    bootstrap_scores = _bootstrap_scores_frame(bootstrap_by_dataset, datasets, n_bootstrap)

    return ClassificationModelEvaluation(
        metrics=pd.DataFrame(rows, columns=RESULT_COLUMNS),
        bootstrap_scores=bootstrap_scores,
        pairwise_wins=pairwise_win_matrices(bootstrap_scores),
    )


def _evaluate_bootstrap_task(task: _BootstrapTask) -> _BootstrapTaskResult:
    metrics, lower, upper, bootstrap = evaluate_bootstrap_classification(
        np.column_stack((1 - task.probability, task.probability)),
        task.labels,
        task.n_bootstrap,
        np.random.default_rng(task.seed),
    )
    return _BootstrapTaskResult(
        ordinal=task.ordinal,
        metrics=metrics,
        lower=lower,
        upper=upper,
        bootstrap=bootstrap,
    )


def _run_bootstrap_tasks(tasks: list[_BootstrapTask], n_jobs: int) -> list[_BootstrapTaskResult]:
    workers = min(n_jobs, len(tasks))
    if workers <= 1:
        return [_evaluate_bootstrap_task(task) for task in tasks]

    # Forkserver workers never inherit CUDA/ROCm state from model training. Preloading
    # the CPU evaluation module avoids repeating its NumPy/pandas/sklearn imports.
    multiprocessing.set_forkserver_preload(["src.utils.prediction_metrics"])
    context = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        results = list(executor.map(_evaluate_bootstrap_task, tasks))
    return sorted(results, key=lambda result: result.ordinal)


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
    n_jobs: int = 4,
) -> pd.DataFrame:
    """Backward-compatible wrapper returning only the summary metrics table."""
    return evaluate_classification_models(
        tables,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        n_jobs=n_jobs,
    ).metrics


def _dataset_bootstrap_seeds(datasets: tuple[str, ...], random_state: int | None) -> dict[str, int]:
    rng = np.random.default_rng(random_state)
    return {dataset: int(rng.integers(np.iinfo(np.uint64).max, dtype=np.uint64)) for dataset in datasets}


def _bootstrap_scores_frame(
    bootstrap_by_dataset: dict[str, dict[str, dict[str, np.ndarray]]],
    datasets: tuple[str, ...],
    n_bootstrap: int,
) -> pd.DataFrame:
    frames = []
    for dataset in datasets:
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
