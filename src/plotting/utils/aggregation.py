"""Validate and aggregate repeated evaluation runs for plotting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.plotting.defaults import ordered_models
from src.plotting.utils.artifacts import PlotArtifacts


@dataclass(frozen=True)
class AggregatedEvaluation:
    """Run-averaged point estimates and aligned bootstrap scores."""

    performance: pd.DataFrame
    bootstrap_scores: pd.DataFrame
    model_metadata: pd.DataFrame
    model_instances: tuple[str, ...]
    datasets: tuple[str, ...]
    metrics: tuple[str, ...]
    trained_on: str
    target: str
    run_ids: tuple[str, ...]
    bootstrap_count: int
    ci_level: float

    @property
    def run_count(self) -> int:
        return len(self.run_ids)

    def scores(self, dataset: str, metric: str) -> pd.DataFrame:
        """Return model columns indexed by aligned bootstrap ID."""
        return self.bootstrap_scores.xs((dataset, metric), level=("dataset", "metric"))


def aggregate_evaluation_runs(
    artifacts: PlotArtifacts,
    *,
    metrics: Sequence[str],
    ci_level: float = 0.95,
) -> AggregatedEvaluation:
    """Average selected runs and recompute intervals from bootstrap scores.

    Point estimates are arithmetic means across pipeline runs. Bootstrap scores
    are aligned by dataset, metric, bootstrap ID, and model instance before they
    are averaged across runs. Percentile intervals are then calculated from the
    run-averaged bootstrap distribution; per-run interval endpoints are never
    averaged.
    """
    selected_metrics = tuple(dict.fromkeys(metrics))
    if not selected_metrics:
        raise ValueError("metrics must contain at least one metric")
    if not 0 < ci_level < 1:
        raise ValueError("ci_level must lie strictly between zero and one")

    metric_required = {
        "pipeline_mlflow_run_id",
        "model_name",
        "model_instance",
        "scope",
        "statistic",
        "dataset",
        "trained_on",
        "target",
        *selected_metrics,
    }
    bootstrap_required = {
        "pipeline_mlflow_run_id",
        "model_name",
        "model_instance",
        "dataset",
        "metric",
        "bootstrap_id",
        "score",
        "trained_on",
        "target",
    }
    _require_columns(artifacts.metrics, metric_required, "evaluation metrics")
    _require_columns(artifacts.bootstrap_scores, bootstrap_required, "bootstrap scores")

    points = artifacts.metrics.loc[
        artifacts.metrics["scope"].eq("test") & artifacts.metrics["statistic"].eq("point")
    ].copy()
    bootstraps = artifacts.bootstrap_scores.loc[
        artifacts.bootstrap_scores["metric"].isin(selected_metrics)
    ].copy()
    if points.empty or bootstraps.empty:
        raise ValueError("Selected artifacts do not contain both point metrics and bootstrap scores")

    trained_on = _single_string(points, "trained_on")
    target = _single_string(points, "target")
    if _single_string(bootstraps, "trained_on") != trained_on:
        raise ValueError("Metric and bootstrap artifacts disagree on trained_on")
    if _single_string(bootstraps, "target") != target:
        raise ValueError("Metric and bootstrap artifacts disagree on target")

    run_ids = tuple(points["pipeline_mlflow_run_id"].astype(str).drop_duplicates())
    if set(run_ids) != set(artifacts.run_ids):
        raise ValueError("Plot artifact metadata and metric rows contain different pipeline runs")
    bootstrap_run_ids = set(bootstraps["pipeline_mlflow_run_id"].astype(str))
    if bootstrap_run_ids != set(run_ids):
        raise ValueError("Point and bootstrap artifacts contain different pipeline runs")

    for frame in (points, bootstraps):
        frame["pipeline_mlflow_run_id"] = frame["pipeline_mlflow_run_id"].astype(str)
        frame["model_instance"] = frame["model_instance"].astype(str)
        frame["model_name"] = frame["model_name"].astype(str)
        _validate_model_identity(frame)

    metadata = points[["model_name", "model_instance"]].drop_duplicates()
    names_by_instance = metadata.groupby("model_instance", sort=False)["model_name"].first()
    model_name_order = {name: index for index, name in enumerate(ordered_models(metadata["model_name"].tolist()))}
    metadata = metadata.assign(_order=metadata["model_name"].map(model_name_order)).sort_values(
        ["_order", "model_instance"], kind="stable"
    )
    metadata = metadata.drop(columns="_order").reset_index(drop=True)
    model_instances = tuple(metadata["model_instance"])
    datasets = tuple(points["dataset"].dropna().astype(str).drop_duplicates())
    if not datasets:
        raise ValueError("No test datasets are available")

    expected_points = pd.MultiIndex.from_product(
        [run_ids, model_instances, datasets],
        names=["pipeline_mlflow_run_id", "model_instance", "dataset"],
    )
    point_counts = points.groupby(list(expected_points.names), sort=False).size()
    missing_points = expected_points.difference(point_counts.index)
    non_unique_points = point_counts[point_counts.ne(1)]
    if len(missing_points) or not non_unique_points.empty:
        raise ValueError(
            "Every selected run must contain exactly one point row per model and test dataset; "
            f"missing={len(missing_points)}, non_unique={len(non_unique_points)}"
        )

    numeric_metrics = points.loc[:, selected_metrics].apply(pd.to_numeric, errors="coerce")
    if numeric_metrics.isna().any().any() or not np.isfinite(numeric_metrics.to_numpy()).all():
        raise ValueError("Selected point metrics must be finite numeric values")
    points.loc[:, selected_metrics] = numeric_metrics

    bootstraps["score"] = pd.to_numeric(bootstraps["score"], errors="coerce")
    bootstraps["bootstrap_id"] = pd.to_numeric(bootstraps["bootstrap_id"], errors="coerce")
    if bootstraps[["score", "bootstrap_id"]].isna().any().any():
        raise ValueError("Bootstrap scores and IDs must be numeric and non-missing")
    if not np.isfinite(bootstraps["score"].to_numpy()).all():
        raise ValueError("Bootstrap scores must be finite")

    key_columns = ["pipeline_mlflow_run_id", "dataset", "metric", "bootstrap_id", "model_instance"]
    if bootstraps.duplicated(key_columns).any():
        raise ValueError("Bootstrap artifacts contain duplicate run/dataset/metric/bootstrap/model rows")
    wide_bootstraps = bootstraps.pivot(index=key_columns[:-1], columns="model_instance", values="score")
    missing_instances = sorted(set(model_instances) - set(wide_bootstraps.columns.astype(str)))
    if missing_instances:
        raise ValueError("Bootstrap artifacts are missing model instances: " + ", ".join(missing_instances))
    wide_bootstraps = wide_bootstraps.loc[:, list(model_instances)]
    if wide_bootstraps.isna().any().any():
        raise ValueError("Bootstrap artifacts have incomplete model scores")

    expected_groups = pd.MultiIndex.from_product(
        [run_ids, datasets, selected_metrics],
        names=["pipeline_mlflow_run_id", "dataset", "metric"],
    )
    observed_groups = set(wide_bootstraps.index.droplevel("bootstrap_id").unique())
    missing_groups = [group for group in expected_groups if group not in observed_groups]
    if missing_groups:
        raise ValueError(f"Bootstrap artifacts are missing {len(missing_groups)} run/dataset/metric groups")

    reference_ids: pd.Index | None = None
    for group in expected_groups:
        ids = wide_bootstraps.xs(group, level=expected_groups.names).index
        if reference_ids is None:
            reference_ids = ids
        elif not ids.equals(reference_ids):
            raise ValueError("Bootstrap IDs differ between selected runs, datasets, or metrics")
    assert reference_ids is not None

    averaged_bootstraps = wide_bootstraps.groupby(["dataset", "metric", "bootstrap_id"], sort=False).mean()
    point_means = points.groupby(["model_instance", "dataset"], sort=False)[list(selected_metrics)].mean()
    alpha = (1.0 - ci_level) / 2.0
    performance_rows: list[dict[str, object]] = []
    for metric in selected_metrics:
        for dataset in datasets:
            scores = averaged_bootstraps.xs((dataset, metric), level=("dataset", "metric"))
            for instance in model_instances:
                lower, upper = scores[instance].quantile([alpha, 1.0 - alpha])
                performance_rows.append(
                    {
                        "model_name": str(names_by_instance.loc[instance]),
                        "model_instance": instance,
                        "dataset": dataset,
                        "metric": metric,
                        "estimate": float(point_means.loc[(instance, dataset), metric]),
                        "lower": float(lower),
                        "upper": float(upper),
                    }
                )

    return AggregatedEvaluation(
        performance=pd.DataFrame(performance_rows),
        bootstrap_scores=averaged_bootstraps,
        model_metadata=metadata,
        model_instances=model_instances,
        datasets=datasets,
        metrics=selected_metrics,
        trained_on=trained_on,
        target=target,
        run_ids=run_ids,
        bootstrap_count=len(reference_ids),
        ci_level=ci_level,
    )


def _single_string(frame: pd.DataFrame, column: str) -> str:
    values = frame[column].dropna().astype(str).unique().tolist()
    if len(values) != 1:
        raise ValueError(f"Expected exactly one {column}; found {values}")
    return values[0]


def _validate_model_identity(frame: pd.DataFrame) -> None:
    names = frame[["model_instance", "model_name"]].drop_duplicates()
    conflicts = names.groupby("model_instance", sort=False)["model_name"].nunique()
    if conflicts.ne(1).any():
        bad = conflicts[conflicts.ne(1)].index.astype(str).tolist()
        raise ValueError("Model instances map to multiple model names: " + ", ".join(bad))


def _require_columns(frame: pd.DataFrame, required: set[str], description: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing {description} columns: {', '.join(missing)}")
