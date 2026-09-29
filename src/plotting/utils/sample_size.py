"""Prepare sample-size experiment results for figure scripts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.plotting.utils.aggregation import aggregate_evaluation_runs
from src.plotting.utils.artifacts import PlotArtifacts
from src.plotting.utils.grouped import GroupedEvaluation, aggregate_runs_by_setting
from src.schemas.training_schemas import scoring_is_lower_better


@dataclass(frozen=True)
class SampleSizeEvaluation:
    """Repeated-run summaries ordered by integer training sample count."""

    grouped: GroupedEvaluation
    sample_sizes: tuple[int, ...]
    target: str
    trained_on: str
    full_training_size: int | None
    full_training_run_count: int

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


@dataclass(frozen=True)
class FullDataBenchmark:
    """One fixed, run-averaged full-data winner for each reporting metric."""

    references: pd.DataFrame
    model_metadata: pd.DataFrame
    target: str
    trained_on: str
    evaluated_on: str
    training_size: int
    run_ids: tuple[str, ...]

    @property
    def run_count(self) -> int:
        return len(self.run_ids)

    def reference(self, metric: str) -> pd.Series:
        """Return the sole fixed winner row for ``metric``."""
        rows = self.references.loc[self.references["metric"].eq(metric)]
        if len(rows) != 1:
            raise ValueError(f"Expected one full-data reference for metric {metric!r}; found {len(rows)}")
        return rows.iloc[0]


@dataclass(frozen=True)
class XGBoostDifferenceEvaluation:
    """Same-size model-minus-XGBoost contrasts from one prepared sweep."""

    sample_size: SampleSizeEvaluation
    performance: pd.DataFrame
    bootstrap_differences: pd.DataFrame
    model_metadata: pd.DataFrame
    model_instances: tuple[str, ...]
    reference_model_instance: str

    @property
    def sample_sizes(self) -> tuple[int, ...]:
        return self.sample_size.sample_sizes

    @property
    def target(self) -> str:
        return self.sample_size.target

    @property
    def trained_on(self) -> str:
        return self.sample_size.trained_on

    @property
    def datasets(self) -> tuple[str, ...]:
        return self.sample_size.datasets

    @property
    def metrics(self) -> tuple[str, ...]:
        return self.sample_size.metrics

    @property
    def run_counts(self) -> dict[str, int]:
        return self.sample_size.run_counts

    @property
    def bootstrap_count(self) -> int:
        return self.sample_size.bootstrap_count

    @property
    def ci_level(self) -> float:
        return self.sample_size.ci_level


def prepare_sample_size_evaluation(
    artifacts: PlotArtifacts,
    *,
    metrics: Sequence[str],
    ci_level: float = 0.95,
    full_training_size: int | None = None,
    full_training_run_count: int = 0,
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
    if full_training_size is not None and full_training_size <= 0:
        raise ValueError("full_training_size must be positive when provided")
    if full_training_run_count < 0:
        raise ValueError("full_training_run_count cannot be negative")
    if (full_training_size is None) != (full_training_run_count == 0):
        raise ValueError("full_training_size and full_training_run_count must be provided together")
    return SampleSizeEvaluation(
        grouped=grouped,
        sample_sizes=sample_sizes,
        target=target,
        trained_on=trained_on,
        full_training_size=full_training_size,
        full_training_run_count=full_training_run_count,
    )


def prepare_xgboost_difference_evaluation(data: SampleSizeEvaluation) -> XGBoostDifferenceEvaluation:
    """Compute same-setting model-minus-XGBoost scores and paired intervals.

    The input has already averaged repeated pipeline runs within each training
    count. Point contrasts therefore compare run-averaged scores, while interval
    endpoints are quantiles of score differences paired on training count,
    evaluation dataset, metric, and bootstrap ID. Confidence-interval endpoints
    from the absolute-score summaries are never subtracted.
    """
    reference_name = "xgboost"
    reference_metadata = data.model_metadata.loc[data.model_metadata["model_name"].eq(reference_name)]
    if len(reference_metadata) != 1:
        instances = reference_metadata["model_instance"].astype(str).tolist()
        raise ValueError(
            "Same-size differences require exactly one XGBoost model instance; "
            f"found {len(reference_metadata)} ({instances})"
        )
    reference_instance = str(reference_metadata.iloc[0]["model_instance"])

    points = data.performance.copy()
    point_columns = {
        "setting",
        "dataset",
        "metric",
        "model_name",
        "model_instance",
        "estimate",
    }
    missing_point_columns = sorted(point_columns - set(points.columns))
    if missing_point_columns:
        raise ValueError("Missing sample-size point columns: " + ", ".join(missing_point_columns))
    points["model_name"] = points["model_name"].astype(str)
    points["model_instance"] = points["model_instance"].astype(str)
    point_keys = ["setting", "dataset", "metric", "model_name", "model_instance"]
    if points.duplicated(point_keys).any():
        raise ValueError("Sample-size performance contains duplicate setting/dataset/metric/model cells")
    if not np.isfinite(pd.to_numeric(points["estimate"], errors="coerce")).all():
        raise ValueError("Sample-size point estimates must be finite numeric values")

    reference_cell_keys = ["setting", "dataset", "metric"]
    expected_reference_cells = pd.MultiIndex.from_product(
        (data.grouped.settings, data.datasets, data.metrics),
        names=reference_cell_keys,
    )
    reference_points = points.loc[points["model_instance"].eq(reference_instance)].copy()
    reference_counts = reference_points.groupby(reference_cell_keys, sort=False).size()
    missing_reference_cells = expected_reference_cells.difference(reference_counts.index)
    duplicate_reference_cells = reference_counts[reference_counts.ne(1)]
    if len(missing_reference_cells) or not duplicate_reference_cells.empty:
        raise ValueError(
            "XGBoost point references must contain exactly one same-size cell for every "
            "training count/evaluation center/metric; "
            f"missing={_preview_index(missing_reference_cells)}, "
            f"non_unique={_preview_index(duplicate_reference_cells.index)}"
        )

    target_points = points.loc[~points["model_instance"].eq(reference_instance)].copy()
    if target_points.empty:
        raise ValueError("The sample-size experiment contains no non-XGBoost models to compare")
    point_reference = reference_points[reference_cell_keys + ["estimate"]].rename(
        columns={"estimate": "xgboost_estimate"}
    )
    point_differences = target_points.merge(
        point_reference,
        on=reference_cell_keys,
        how="left",
        validate="many_to_one",
    )
    if point_differences["xgboost_estimate"].isna().any():
        raise ValueError("A non-XGBoost point cell has no same-size XGBoost reference")
    point_differences["estimate"] = pd.to_numeric(point_differences["estimate"]) - pd.to_numeric(
        point_differences["xgboost_estimate"]
    )

    bootstraps = data.grouped.bootstrap_scores.copy()
    bootstrap_columns = {
        "setting",
        "dataset",
        "metric",
        "bootstrap_id",
        "model_name",
        "model_instance",
        "score",
    }
    missing_bootstrap_columns = sorted(bootstrap_columns - set(bootstraps.columns))
    if missing_bootstrap_columns:
        raise ValueError("Missing sample-size bootstrap columns: " + ", ".join(missing_bootstrap_columns))
    bootstraps["model_name"] = bootstraps["model_name"].astype(str)
    bootstraps["model_instance"] = bootstraps["model_instance"].astype(str)
    bootstrap_keys = [*reference_cell_keys, "bootstrap_id", "model_name", "model_instance"]
    if bootstraps.duplicated(bootstrap_keys).any():
        raise ValueError("Sample-size bootstrap scores contain duplicate setting/dataset/metric/draw/model rows")
    if not np.isfinite(pd.to_numeric(bootstraps["score"], errors="coerce")).all():
        raise ValueError("Sample-size bootstrap scores must be finite numeric values")

    reference_bootstraps = bootstraps.loc[bootstraps["model_instance"].eq(reference_instance)].copy()
    observed_reference_cells = pd.MultiIndex.from_frame(reference_bootstraps[reference_cell_keys].drop_duplicates())
    missing_bootstrap_references = expected_reference_cells.difference(observed_reference_cells)
    if len(missing_bootstrap_references):
        raise ValueError(
            "XGBoost bootstrap references are missing same-size training count/evaluation center/metric cells: "
            + _preview_index(missing_bootstrap_references)
        )

    target_bootstraps = bootstraps.loc[~bootstraps["model_instance"].eq(reference_instance)].copy()
    reference_draw_keys = [*reference_cell_keys, "bootstrap_id"]
    expected_target_draws = target_points[point_keys].merge(
        reference_bootstraps[reference_draw_keys],
        on=reference_cell_keys,
        how="left",
        validate="many_to_many",
    )
    expected_draw_index = pd.MultiIndex.from_frame(expected_target_draws[bootstrap_keys])
    observed_draw_index = pd.MultiIndex.from_frame(target_bootstraps[bootstrap_keys])
    missing_target_draws = expected_draw_index.difference(observed_draw_index)
    unmatched_target_draws = observed_draw_index.difference(expected_draw_index)
    if len(missing_target_draws) or len(unmatched_target_draws):
        raise ValueError(
            "Non-XGBoost and XGBoost bootstrap scores do not have an exact same-size paired match on "
            "setting/dataset/metric/bootstrap_id; "
            f"missing_model_draws={_preview_index(missing_target_draws)}, "
            f"unmatched_model_draws={_preview_index(unmatched_target_draws)}"
        )

    paired = target_bootstraps.merge(
        reference_bootstraps[reference_draw_keys + ["score"]].rename(columns={"score": "xgboost_score"}),
        on=reference_draw_keys,
        how="left",
        validate="many_to_one",
    )
    if paired["xgboost_score"].isna().any():
        raise ValueError("A non-XGBoost bootstrap score has no same-size XGBoost match")
    paired["difference"] = pd.to_numeric(paired["score"]) - pd.to_numeric(paired["xgboost_score"])

    alpha = (1.0 - data.ci_level) / 2.0
    intervals = (
        paired.groupby(point_keys, sort=False)["difference"]
        .quantile([alpha, 1.0 - alpha])
        .unstack()
        .reset_index()
        .rename(columns={alpha: "lower", 1.0 - alpha: "upper"})
    )
    performance = point_differences[point_keys + ["estimate"]].merge(
        intervals,
        on=point_keys,
        how="left",
        validate="one_to_one",
    )
    if performance[["lower", "upper"]].isna().any().any():
        raise ValueError("Paired XGBoost differences did not produce an interval for every model cell")

    model_metadata = data.model_metadata.loc[
        ~data.model_metadata["model_instance"].astype(str).eq(reference_instance)
    ].reset_index(drop=True)
    model_instances = tuple(instance for instance in data.model_instances if str(instance) != reference_instance)
    return XGBoostDifferenceEvaluation(
        sample_size=data,
        performance=performance,
        bootstrap_differences=paired,
        model_metadata=model_metadata,
        model_instances=model_instances,
        reference_model_instance=reference_instance,
    )


def prepare_full_data_benchmark(
    artifacts: PlotArtifacts,
    *,
    evaluated_on: str,
    metrics: Sequence[str],
) -> FullDataBenchmark:
    """Choose one run-averaged full-data winner per metric on one test center.

    ``artifacts`` must already have been selected by ``load_plot_artifacts`` with
    ``full_training_only=True``. Winner selection uses point estimates averaged
    over those complete-data runs; bootstrap draws never select the winner.
    """
    aggregated = aggregate_evaluation_runs(artifacts, metrics=metrics, ci_level=None)
    if evaluated_on not in aggregated.datasets:
        raise ValueError(
            f"Full-data artifacts trained on {aggregated.trained_on!r} have no {evaluated_on!r} test dataset; "
            f"available: {list(aggregated.datasets)}"
        )

    references = []
    for metric in aggregated.metrics:
        candidates = aggregated.performance.loc[
            aggregated.performance["dataset"].eq(evaluated_on) & aggregated.performance["metric"].eq(metric)
        ]
        if candidates.empty:
            raise ValueError(f"No full-data candidates are available for {evaluated_on!r}/{metric!r}")
        winner_index = (
            candidates["estimate"].idxmin() if scoring_is_lower_better(metric) else candidates["estimate"].idxmax()
        )
        references.append(candidates.loc[winner_index, ["model_name", "model_instance", "metric", "estimate"]])

    return FullDataBenchmark(
        references=pd.DataFrame(references).reset_index(drop=True),
        model_metadata=aggregated.model_metadata,
        target=aggregated.target,
        trained_on=aggregated.trained_on,
        evaluated_on=evaluated_on,
        training_size=full_training_count(artifacts),
        run_ids=aggregated.run_ids,
    )


def full_training_count(artifacts: PlotArtifacts) -> int:
    """Return the common training row count of explicitly selected full runs."""
    required = {"pipeline_mlflow_run_id", "scope", "statistic", "training_size"}
    missing = sorted(required - set(artifacts.metrics.columns))
    if missing:
        raise ValueError("Missing full-data count columns: " + ", ".join(missing))
    rows = artifacts.metrics.loc[
        artifacts.metrics["scope"].eq("test") & artifacts.metrics["statistic"].eq("point")
    ].copy()
    if rows.empty:
        raise ValueError("No scope='test', statistic='point' rows are available")
    rows["training_size"] = pd.to_numeric(rows["training_size"], errors="coerce")
    if rows["training_size"].isna().any() or rows["training_size"].le(0).any():
        raise ValueError("Full-data training_size must contain positive numeric sample counts")
    if rows["training_size"].mod(1).ne(0).any():
        raise ValueError("Full-data training_size must contain integer sample counts")
    sizes_by_run = rows.groupby("pipeline_mlflow_run_id", sort=False)["training_size"].nunique()
    if sizes_by_run.ne(1).any():
        bad = sizes_by_run[sizes_by_run.ne(1)].index.astype(str).tolist()
        raise ValueError("Full-data pipeline runs contain multiple training sizes: " + ", ".join(bad))
    training_sizes = rows.groupby("pipeline_mlflow_run_id", sort=False)["training_size"].first().unique()
    if len(training_sizes) != 1:
        raise ValueError(f"Explicit full-data runs disagree on training count: {sorted(training_sizes.tolist())}")
    return int(training_sizes[0])


def _single_string(frame: pd.DataFrame, column: str) -> str:
    values = frame[column].dropna().astype(str).unique().tolist()
    if len(values) != 1:
        raise ValueError(f"Expected exactly one {column}; found {values}")
    return values[0]


def _preview_index(index: pd.Index, maximum: int = 3) -> str:
    if len(index) == 0:
        return "none"
    values = [str(value) for value in index[:maximum]]
    suffix = f" (+{len(index) - maximum} more)" if len(index) > maximum else ""
    return ", ".join(values) + suffix
