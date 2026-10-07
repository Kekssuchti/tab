"""Run-averaged evaluation results built from the shared per-setting aggregator."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from src.plotting.utils.artifacts import PlotArtifacts
from src.plotting.utils.grouped import aggregate_runs_by_setting
from src.plotting.utils.runs import require_columns, single_value


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
    ci_level: float | None

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
    ci_level: float | None = 0.95,
) -> AggregatedEvaluation:
    """Average selected runs as one setting, including their paired draws.

    Point estimates and bootstrap scores are averaged separately across runs;
    intervals come from the averaged draws, never averaged interval endpoints.
    The same implementation handles repeated runs inside each distinct setting.
    """
    require_columns(artifacts.metrics, {"scope", "statistic", "trained_on", "target"}, "evaluation metrics")
    require_columns(artifacts.bootstrap_scores, {"metric", "trained_on", "target"}, "bootstrap scores")
    points = artifacts.metrics.loc[artifacts.metrics["scope"].eq("test") & artifacts.metrics["statistic"].eq("point")]
    bootstraps = artifacts.bootstrap_scores.loc[artifacts.bootstrap_scores["metric"].isin(metrics)]
    if points.empty or bootstraps.empty:
        raise ValueError("Selected artifacts do not contain both point metrics and bootstrap scores")
    trained_on = single_value(points, "trained_on", "evaluation metrics")
    target = single_value(points, "target", "evaluation metrics")
    if single_value(bootstraps, "trained_on", "bootstrap scores") != trained_on:
        raise ValueError("Metric and bootstrap artifacts disagree on trained_on")
    if single_value(bootstraps, "target", "bootstrap scores") != target:
        raise ValueError("Metric and bootstrap artifacts disagree on target")

    run_ids = tuple(points["pipeline_mlflow_run_id"].astype(str).drop_duplicates())
    grouped = aggregate_runs_by_setting(
        artifacts,
        {str(run_id): "all" for run_id in artifacts.run_ids},
        metrics=metrics,
        ci_level=ci_level,
    )
    scores = grouped.bootstrap_scores.pivot(
        index=["dataset", "metric", "bootstrap_id"], columns="model_instance", values="score"
    ).loc[:, list(grouped.model_instances)]
    return AggregatedEvaluation(
        performance=grouped.performance.drop(columns="setting"),
        bootstrap_scores=scores,
        model_metadata=grouped.model_metadata,
        model_instances=grouped.model_instances,
        datasets=grouped.datasets,
        metrics=grouped.metrics,
        trained_on=trained_on,
        target=target,
        run_ids=run_ids,
        bootstrap_count=grouped.bootstrap_count,
        ci_level=ci_level,
    )
