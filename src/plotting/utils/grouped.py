"""Aggregate repeated evaluation runs by an explicit experimental setting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from src.plotting.defaults import ordered_models
from src.plotting.utils.artifacts import PlotArtifacts


@dataclass(frozen=True)
class GroupedEvaluation:
    """Point, interval, runtime, and bootstrap summaries by setting."""

    performance: pd.DataFrame
    runtimes: pd.DataFrame
    bootstrap_scores: pd.DataFrame
    model_metadata: pd.DataFrame
    model_instances: tuple[str, ...]
    settings: tuple[str, ...]
    datasets: tuple[str, ...]
    metrics: tuple[str, ...]
    run_counts: dict[str, int]
    bootstrap_count: int
    ci_level: float


def aggregate_runs_by_setting(
    artifacts: PlotArtifacts,
    setting_by_run: Mapping[str, str],
    *,
    setting_order: Sequence[str] | None = None,
    metrics: Sequence[str],
    runtime_columns: Sequence[str] = (),
    ci_level: float = 0.95,
) -> GroupedEvaluation:
    """Average repeated runs within settings and recompute bootstrap intervals."""
    selected_metrics = tuple(dict.fromkeys(metrics))
    runtime_columns = tuple(dict.fromkeys(runtime_columns))
    if not selected_metrics:
        raise ValueError("metrics must contain at least one metric")
    if not 0 < ci_level < 1:
        raise ValueError("ci_level must lie strictly between zero and one")

    run_map = {str(run_id): str(setting) for run_id, setting in setting_by_run.items()}
    artifact_runs = set(artifacts.run_ids)
    if set(run_map) != artifact_runs:
        raise ValueError(
            "setting_by_run must assign every selected pipeline run exactly once; "
            f"missing={sorted(artifact_runs - set(run_map))}, extra={sorted(set(run_map) - artifact_runs)}"
        )
    settings = tuple(dict.fromkeys(setting_order or run_map.values()))
    if set(settings) != set(run_map.values()):
        raise ValueError("setting_order must contain every mapped setting exactly once")

    required_points = {
        "pipeline_mlflow_run_id",
        "model_name",
        "model_instance",
        "scope",
        "statistic",
        "dataset",
        *selected_metrics,
        *runtime_columns,
    }
    required_bootstrap = {
        "pipeline_mlflow_run_id",
        "model_name",
        "model_instance",
        "dataset",
        "metric",
        "bootstrap_id",
        "score",
    }
    _require_columns(artifacts.metrics, required_points, "evaluation metrics")
    _require_columns(artifacts.bootstrap_scores, required_bootstrap, "bootstrap scores")

    points = artifacts.metrics.loc[
        artifacts.metrics["scope"].eq("test") & artifacts.metrics["statistic"].eq("point")
    ].copy()
    bootstraps = artifacts.bootstrap_scores.loc[artifacts.bootstrap_scores["metric"].isin(selected_metrics)].copy()
    for frame in (points, bootstraps):
        frame["pipeline_mlflow_run_id"] = frame["pipeline_mlflow_run_id"].astype(str)
        frame["model_instance"] = frame["model_instance"].astype(str)
        frame["model_name"] = frame["model_name"].astype(str)
        frame["setting"] = frame["pipeline_mlflow_run_id"].map(run_map)
        _validate_model_identity(frame)
    if points["setting"].isna().any() or bootstraps["setting"].isna().any():
        raise ValueError("Selected artifacts contain pipeline runs without a setting assignment")

    _validate_repeat_model_sets(points, settings)
    metadata = points[["model_name", "model_instance"]].drop_duplicates()
    model_order = {name: index for index, name in enumerate(ordered_models(metadata["model_name"].tolist()))}
    metadata = metadata.assign(_order=metadata["model_name"].map(model_order)).sort_values(
        ["_order", "model_instance"], kind="stable"
    )
    metadata = metadata.drop(columns="_order").reset_index(drop=True)
    model_instances = tuple(metadata["model_instance"])
    datasets = tuple(points["dataset"].dropna().astype(str).drop_duplicates())
    if not datasets:
        raise ValueError("No test datasets are available")
    _validate_point_cells(points, datasets)
    run_counts = {
        setting: len({run_id for run_id, mapped in run_map.items() if mapped == setting}) for setting in settings
    }

    point_columns = [*selected_metrics, *runtime_columns]
    points.loc[:, point_columns] = points.loc[:, point_columns].apply(pd.to_numeric, errors="coerce")
    if points.loc[:, point_columns].isna().any().any():
        raise ValueError("Selected metric and runtime values must be numeric and non-missing")
    point_means = points.groupby(["setting", "model_name", "model_instance", "dataset"], sort=False, as_index=False)[
        list(selected_metrics)
    ].mean()

    bootstraps["score"] = pd.to_numeric(bootstraps["score"], errors="coerce")
    bootstraps["bootstrap_id"] = pd.to_numeric(bootstraps["bootstrap_id"], errors="coerce")
    if bootstraps[["score", "bootstrap_id"]].isna().any().any():
        raise ValueError("Bootstrap scores and IDs must be numeric and non-missing")
    duplicate_keys = [
        "pipeline_mlflow_run_id",
        "dataset",
        "metric",
        "bootstrap_id",
        "model_instance",
    ]
    if bootstraps.duplicated(duplicate_keys).any():
        raise ValueError("Bootstrap artifacts contain duplicate run/dataset/metric/bootstrap/model rows")

    averaged_bootstraps = bootstraps.groupby(
        ["setting", "dataset", "metric", "bootstrap_id", "model_name", "model_instance"],
        sort=False,
        as_index=False,
    )["score"].mean()
    alpha = (1.0 - ci_level) / 2.0
    intervals = (
        averaged_bootstraps.groupby(
            ["setting", "dataset", "metric", "model_name", "model_instance"],
            sort=False,
        )["score"]
        .quantile([alpha, 1.0 - alpha])
        .unstack()
        .reset_index()
        .rename(columns={alpha: "lower", 1.0 - alpha: "upper"})
    )

    performance_rows = []
    for metric in selected_metrics:
        estimates = point_means[["setting", "model_name", "model_instance", "dataset", metric]].rename(
            columns={metric: "estimate"}
        )
        metric_intervals = intervals.loc[intervals["metric"].eq(metric)].drop(columns="metric")
        performance_rows.append(
            estimates.merge(
                metric_intervals,
                on=["setting", "model_name", "model_instance", "dataset"],
                how="left",
                validate="one_to_one",
            ).assign(metric=metric)
        )
    performance = pd.concat(performance_rows, ignore_index=True)

    if runtime_columns:
        per_run_runtime = points.drop_duplicates(["pipeline_mlflow_run_id", "model_instance"])
        runtimes = per_run_runtime.groupby(["setting", "model_name", "model_instance"], sort=False, as_index=False)[
            list(runtime_columns)
        ].mean()
    else:
        runtimes = pd.DataFrame(columns=["setting", "model_name", "model_instance"])

    return GroupedEvaluation(
        performance=performance,
        runtimes=runtimes,
        bootstrap_scores=averaged_bootstraps,
        model_metadata=metadata,
        model_instances=model_instances,
        settings=settings,
        datasets=datasets,
        metrics=selected_metrics,
        run_counts=run_counts,
        bootstrap_count=int(averaged_bootstraps["bootstrap_id"].nunique()),
        ci_level=ci_level,
    )


def _validate_model_identity(frame: pd.DataFrame) -> None:
    identity = frame[["model_instance", "model_name"]].drop_duplicates()
    conflicts = identity.groupby("model_instance", sort=False)["model_name"].nunique()
    if conflicts.ne(1).any():
        bad = conflicts[conflicts.ne(1)].index.astype(str).tolist()
        raise ValueError("Model instances map to multiple model names: " + ", ".join(bad))


def _validate_point_cells(points: pd.DataFrame, datasets: Sequence[str]) -> None:
    key_columns = ["pipeline_mlflow_run_id", "model_instance", "dataset"]
    counts = points.groupby(key_columns, sort=False).size()
    if counts.ne(1).any():
        raise ValueError("Every run/model/dataset point cell must contain exactly one row")
    for run_id, run_rows in points.groupby("pipeline_mlflow_run_id", sort=False):
        models = tuple(run_rows["model_instance"].drop_duplicates())
        expected = pd.MultiIndex.from_product((models, datasets), names=["model_instance", "dataset"])
        observed = pd.MultiIndex.from_frame(run_rows[["model_instance", "dataset"]])
        if len(expected.difference(observed)):
            raise ValueError(f"Pipeline run {run_id} has incomplete model/dataset point cells")


def _validate_repeat_model_sets(points: pd.DataFrame, settings: Sequence[str]) -> None:
    run_sets = points.groupby(["setting", "pipeline_mlflow_run_id"], sort=False)["model_instance"].agg(
        lambda values: set(values)
    )
    for setting in settings:
        sets = run_sets.xs(setting, level="setting").tolist()
        if any(model_set != sets[0] for model_set in sets[1:]):
            raise ValueError(f"Model sets differ between repeated runs for setting {setting!r}")


def _require_columns(frame: pd.DataFrame, required: set[str], description: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing {description} columns: {', '.join(missing)}")
