"""Load and select canonical evaluation artifacts for figure scripts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from src.mlflow.evaluation_data import DEFAULT_TRACKING_URI, load_bootstrap_data, load_evaluation_data


@dataclass(frozen=True)
class PlotArtifacts:
    """Metric and bootstrap artifacts selected for one figure."""

    metrics: pd.DataFrame
    bootstrap_scores: pd.DataFrame
    experiment_name: str
    run_ids: tuple[str, ...]


def load_plot_artifacts(
    experiment_name: str,
    *,
    pipeline_runs: str | Sequence[str] | None = None,
    models: str | Sequence[str] | None = None,
    full_training_only: bool = False,
    include_bootstrap: bool = True,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> PlotArtifacts:
    """Load plotting artifacts and resolve their concrete pipeline run IDs.

    When ``full_training_only`` is true and no explicit run selector is given,
    all runs at the largest observed training size are retained. This naturally
    keeps repeated full-data runs together for downstream averaging.
    """
    if pipeline_runs is not None and full_training_only:
        raise ValueError("pipeline_runs and full_training_only cannot be combined")

    metrics = load_evaluation_data(
        experiment_name,
        pipeline_runs=pipeline_runs,
        models=models,
        tracking_uri=tracking_uri,
    )
    if metrics.empty:
        raise ValueError(f"No evaluation data found for experiment {experiment_name!r}")

    if full_training_only:
        run_ids = select_full_training_run_ids(metrics)
        metrics = metrics.loc[metrics["pipeline_mlflow_run_id"].astype(str).isin(run_ids)].copy()
    else:
        run_ids = tuple(metrics["pipeline_mlflow_run_id"].astype(str).drop_duplicates())

    if not include_bootstrap:
        return PlotArtifacts(
            metrics=metrics,
            bootstrap_scores=pd.DataFrame(),
            experiment_name=experiment_name,
            run_ids=run_ids,
        )

    bootstrap_scores = load_bootstrap_data(
        experiment_name,
        pipeline_runs=run_ids,
        models=models,
        tracking_uri=tracking_uri,
    )
    if bootstrap_scores.empty:
        raise ValueError("No bootstrap_metrics.csv artifacts found for the selected pipeline runs")
    bootstrap_run_ids = set(bootstrap_scores["pipeline_mlflow_run_id"].astype(str))
    if bootstrap_run_ids != set(run_ids):
        raise ValueError(
            "Metric and bootstrap artifacts contain different pipeline runs: "
            f"metrics={list(run_ids)}, bootstrap={sorted(bootstrap_run_ids)}"
        )

    return PlotArtifacts(
        metrics=metrics,
        bootstrap_scores=bootstrap_scores,
        experiment_name=experiment_name,
        run_ids=run_ids,
    )


def select_full_training_run_ids(metrics: pd.DataFrame) -> tuple[str, ...]:
    """Select all repeated runs at the largest observed training size."""
    required = {"pipeline_mlflow_run_id", "scope", "statistic", "target", "trained_on", "training_size"}
    _require_columns(metrics, required, "evaluation metrics")
    rows = metrics.loc[metrics["scope"].eq("test") & metrics["statistic"].eq("point")].copy()
    if rows.empty:
        raise ValueError("No scope='test', statistic='point' rows are available")

    for column in ("target", "trained_on"):
        values = rows[column].dropna().astype(str).unique().tolist()
        if len(values) != 1:
            raise ValueError(
                f"Automatic full-data selection requires exactly one {column}; found {values}. "
                "Select pipeline runs explicitly."
            )

    rows["training_size"] = pd.to_numeric(rows["training_size"], errors="coerce")
    if rows["training_size"].isna().any():
        raise ValueError("training_size must be numeric for automatic full-data selection")
    sizes_per_run = rows.groupby("pipeline_mlflow_run_id", sort=False)["training_size"].nunique()
    if sizes_per_run.ne(1).any():
        bad = sizes_per_run[sizes_per_run.ne(1)].index.astype(str).tolist()
        raise ValueError("Pipeline runs contain multiple training sizes: " + ", ".join(bad))

    run_sizes = rows.groupby("pipeline_mlflow_run_id", sort=False)["training_size"].first()
    largest = float(run_sizes.max())
    return tuple(run_sizes.index[run_sizes.eq(largest)].astype(str))


def _require_columns(frame: pd.DataFrame, required: set[str], description: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing {description} columns: {', '.join(missing)}")
