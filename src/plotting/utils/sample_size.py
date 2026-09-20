"""Prepare sample-size experiment results for figure scripts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from src.plotting.utils.artifacts import PlotArtifacts
from src.plotting.utils.grouped import GroupedEvaluation, aggregate_runs_by_setting


@dataclass(frozen=True)
class SampleSizeEvaluation:
    """Repeated-run summaries ordered by integer training sample count."""

    grouped: GroupedEvaluation
    sample_sizes: tuple[int, ...]
    target: str
    trained_on: str

    @property
    def performance(self) -> pd.DataFrame:
        return self.grouped.performance

    @property
    def model_metadata(self) -> pd.DataFrame:
        return self.grouped.model_metadata

    @property
    def model_instances(self) -> tuple[str, ...]:
        return self.grouped.model_instances

    @property
    def datasets(self) -> tuple[str, ...]:
        return self.grouped.datasets

    @property
    def metrics(self) -> tuple[str, ...]:
        return self.grouped.metrics

    @property
    def run_counts(self) -> dict[str, int]:
        return self.grouped.run_counts

    @property
    def bootstrap_count(self) -> int:
        return self.grouped.bootstrap_count

    @property
    def ci_level(self) -> float:
        return self.grouped.ci_level


def prepare_sample_size_evaluation(
    artifacts: PlotArtifacts,
    *,
    metrics: Sequence[str],
    ci_level: float = 0.95,
) -> SampleSizeEvaluation:
    """Infer sample-size settings and aggregate repeated runs within each size."""
    required = {
        "pipeline_mlflow_run_id",
        "scope",
        "statistic",
        "training_size",
        "target",
        "trained_on",
    }
    missing = sorted(required - set(artifacts.metrics.columns))
    if missing:
        raise ValueError("Missing sample-size metric columns: " + ", ".join(missing))

    rows = artifacts.metrics.loc[
        artifacts.metrics["scope"].eq("test") & artifacts.metrics["statistic"].eq("point")
    ].copy()
    if rows.empty:
        raise ValueError("No scope='test', statistic='point' rows are available")
    target = _single_string(rows, "target")
    trained_on = _single_string(rows, "trained_on")
    rows["training_size"] = pd.to_numeric(rows["training_size"], errors="coerce")
    if rows["training_size"].isna().any() or rows["training_size"].le(0).any():
        raise ValueError("training_size must contain positive numeric sample counts")
    if rows["training_size"].mod(1).ne(0).any():
        raise ValueError("training_size must contain integer sample counts")

    size_counts = rows.groupby("pipeline_mlflow_run_id", sort=False)["training_size"].nunique()
    if size_counts.ne(1).any():
        bad = size_counts[size_counts.ne(1)].index.astype(str).tolist()
        raise ValueError("Pipeline runs contain multiple training sizes: " + ", ".join(bad))
    run_sizes = rows.groupby("pipeline_mlflow_run_id", sort=False)["training_size"].first().astype(int)
    setting_by_run = {str(run_id): str(size) for run_id, size in run_sizes.items()}
    sample_sizes = tuple(sorted(run_sizes.unique()))
    setting_order = tuple(str(size) for size in sample_sizes)

    grouped = aggregate_runs_by_setting(
        artifacts,
        setting_by_run,
        setting_order=setting_order,
        metrics=metrics,
        ci_level=ci_level,
    )
    return SampleSizeEvaluation(
        grouped=grouped,
        sample_sizes=sample_sizes,
        target=target,
        trained_on=trained_on,
    )


def _single_string(frame: pd.DataFrame, column: str) -> str:
    values = frame[column].dropna().astype(str).unique().tolist()
    if len(values) != 1:
        raise ValueError(f"Expected exactly one {column}; found {values}")
    return values[0]
