from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.config import config
from src.schemas.metrics import BootstrapClassificationMetrics, ClassificationMetrics, calculate_metric_diff
from src.utils.evaluation import _evaluate_bootstrap_classification
from src.utils.prediction_tables import (
    PREDICTION_COLUMN_PREFIX,
    PREDICTION_DATASETS,
    TEST_SET_ID_COLUMN,
    Y_TRUE_COLUMN,
    BinaryTestPredictions,
    is_prediction_column,
)

_CLASSIFICATION_METRICS = ("roc_auc", "prc_auc", "f1", "accuracy", "sensitivity", "precision")
_RESULT_COLUMNS = (
    "model_instance",
    "prediction_column",
    "scope",
    "dataset",
    "statistic",
    "ci_level",
    "n_bootstrap",
    *(column for metric in _CLASSIFICATION_METRICS for column in (metric, f"{metric}_ci_lower", f"{metric}_ci_upper")),
)


def recompute_classification_metrics(
    tables: Mapping[str, pd.DataFrame],
    *,
    n_bootstrap: int = 10_000,
    random_state: int | None = config.seed,
) -> pd.DataFrame:
    """Recompute the pipeline's binary metrics from held-out prediction tables.

    A fresh random generator is created for every model and then consumed in
    MIMIC/TUDD order, matching the final evaluation performed during training.
    """

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")

    validated = {dataset: _validated_table(tables, dataset) for dataset in PREDICTION_DATASETS}
    prediction_columns = tuple(column for column in validated["mimic"].columns if is_prediction_column(column))
    for dataset in PREDICTION_DATASETS[1:]:
        columns = tuple(column for column in validated[dataset].columns if is_prediction_column(column))
        if columns != prediction_columns:
            raise ValueError("MIMIC and TUDD prediction tables must contain the same model columns in the same order")

    rows: list[dict[str, object]] = []
    for column in prediction_columns:
        rng = np.random.default_rng(random_state)
        by_dataset: dict[str, BootstrapClassificationMetrics] = {}
        for dataset in PREDICTION_DATASETS:
            table = validated[dataset]
            positive_probability = table[column].to_numpy(dtype=float)
            probabilities = np.column_stack((1.0 - positive_probability, positive_probability))
            evaluated = _evaluate_bootstrap_classification(
                predictions=probabilities,
                y_true=table[Y_TRUE_COLUMN].to_numpy(dtype=int),
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
            by_dataset[dataset] = evaluated
            rows.append(_test_row(column, dataset, evaluated))

        difference = calculate_metric_diff(
            by_dataset["mimic"].metrics,
            by_dataset["tudd"].metrics,
        )
        rows.append(_difference_row(column, difference, n_bootstrap))

    return pd.DataFrame(rows, columns=_RESULT_COLUMNS)


def _validated_table(tables: Mapping[str, pd.DataFrame], dataset: str) -> pd.DataFrame:
    if dataset not in tables:
        raise ValueError(f"Missing prediction table for dataset {dataset!r}")
    table = tables[dataset]
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"Prediction table for {dataset!r} must be a pandas DataFrame")

    missing = [column for column in (TEST_SET_ID_COLUMN, Y_TRUE_COLUMN) if column not in table.columns]
    if missing:
        raise ValueError(f"Prediction table for {dataset!r} is missing columns: {', '.join(missing)}")
    unexpected = [
        str(column)
        for column in table.columns
        if column not in (TEST_SET_ID_COLUMN, Y_TRUE_COLUMN) and not is_prediction_column(column)
    ]
    if unexpected:
        raise ValueError(f"Prediction table for {dataset!r} has unsupported columns: {', '.join(unexpected)}")

    prediction_columns = [column for column in table.columns if is_prediction_column(column)]
    if not prediction_columns:
        raise ValueError(f"Prediction table for {dataset!r} does not contain any model columns")
    checked = table.loc[:, [TEST_SET_ID_COLUMN, Y_TRUE_COLUMN, *prediction_columns]].copy()
    for column in prediction_columns:
        BinaryTestPredictions(
            test_set_id=checked[TEST_SET_ID_COLUMN].to_numpy(),
            y_true=checked[Y_TRUE_COLUMN].to_numpy(),
            positive_class_probability=checked[column].to_numpy(),
        )
    return checked


def _test_row(
    prediction_column_name: str,
    dataset: str,
    evaluated: BootstrapClassificationMetrics,
) -> dict[str, object]:
    row = _base_row(
        prediction_column_name,
        scope="test",
        dataset=dataset,
        statistic="point",
        ci_level=0.95,
        n_bootstrap=evaluated.n_bootstrap,
    )
    row.update(evaluated.scores)
    for metric, (lower, upper) in evaluated.confidence_intervals.items():
        row[f"{metric}_ci_lower"] = lower
        row[f"{metric}_ci_upper"] = upper
    return row


def _difference_row(
    prediction_column_name: str,
    difference: ClassificationMetrics,
    n_bootstrap: int,
) -> dict[str, object]:
    row = _base_row(
        prediction_column_name,
        scope="test_delta",
        dataset="mimic_minus_tudd",
        statistic="difference",
        ci_level=None,
        n_bootstrap=n_bootstrap,
    )
    row.update(difference.scores)
    return row


def _base_row(
    prediction_column_name: str,
    *,
    scope: str,
    dataset: str,
    statistic: str,
    ci_level: float | None,
    n_bootstrap: int,
) -> dict[str, object]:
    row: dict[str, object] = {column: None for column in _RESULT_COLUMNS}
    row.update(
        {
            "model_instance": prediction_column_name.removeprefix(PREDICTION_COLUMN_PREFIX),
            "prediction_column": prediction_column_name,
            "scope": scope,
            "dataset": dataset,
            "statistic": statistic,
            "ci_level": ci_level,
            "n_bootstrap": n_bootstrap,
        }
    )
    return row
