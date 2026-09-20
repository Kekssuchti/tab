"""Assign pipeline runs to experimental settings.

A "setting" is whatever the experiment varies between runs: the training row
count, an estimator budget, an ablation level. The two pairwise scripts need the
same derivation, so it lives here rather than in each of them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

SETTING_SOURCES = ("training_size", "run_name")


def assign_settings(
    metrics: pd.DataFrame,
    run_ids: Sequence[str],
    *,
    source: str = "training_size",
    pattern: str | None = None,
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Map every selected run to a setting and return the settings in order.

    ``source="training_size"`` reads the training row count of each run, which is
    what a sample-size experiment varies. ``source="run_name"`` extracts the
    setting from the pipeline run name, for experiments whose setting is not a
    column of the result table.
    """
    if source not in SETTING_SOURCES:
        raise ValueError(f"Unknown setting source {source!r}; expected one of {SETTING_SOURCES}")

    selected = metrics.loc[metrics["pipeline_mlflow_run_id"].astype(str).isin(set(run_ids))]
    runs = selected[["pipeline_mlflow_run_id", "pipeline_run_name", "training_size"]].drop_duplicates(
        "pipeline_mlflow_run_id"
    )
    if source == "training_size":
        sizes = pd.to_numeric(runs["training_size"], errors="coerce")
        if sizes.isna().any():
            raise ValueError("training_size must be numeric to group runs by training size")
        runs = runs.assign(setting=sizes.astype(int).astype(str)).sort_values(
            "setting", key=lambda column: column.astype(int), kind="stable"
        )
    else:
        if not pattern:
            raise ValueError("A run-name setting source needs a pattern with one capture group")
        runs = runs.assign(setting=runs["pipeline_run_name"].astype(str).str.extract(pattern, expand=False))
        if runs["setting"].isna().any():
            invalid = runs.loc[runs["setting"].isna(), "pipeline_run_name"].tolist()
            raise ValueError(f"Could not extract a setting from run names: {invalid}")
        numeric = pd.to_numeric(runs["setting"], errors="coerce")
        # Numeric settings read as numbers, not as text: 16 belongs after 8, not after 1.
        order_key = numeric if numeric.notna().all() else runs["setting"]
        runs = runs.assign(_order=order_key).sort_values("_order", kind="stable").drop(columns="_order")

    setting_by_run = dict(zip(runs["pipeline_mlflow_run_id"].astype(str), runs["setting"].astype(str), strict=True))
    return setting_by_run, tuple(runs["setting"].astype(str).drop_duplicates())


def setting_display(setting: str, source: str) -> str:
    """Format one setting value for an axis tick or a table header."""
    if source != "training_size":
        return setting
    try:
        return f"{int(float(setting)):,}"
    except ValueError:
        return setting


def settings_by_run(setting_by_run: Mapping[str, str]) -> dict[str, tuple[str, ...]]:
    """Group run IDs by their setting, keeping the order they were given in."""
    grouped: dict[str, list[str]] = {}
    for run_id, setting in setting_by_run.items():
        grouped.setdefault(str(setting), []).append(str(run_id))
    return {setting: tuple(run_ids) for setting, run_ids in grouped.items()}
