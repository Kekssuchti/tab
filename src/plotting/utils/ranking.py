"""Average model ranks and Friedman-Nemenyi critical differences.

Ranks are computed inside evaluation blocks, where a block is one pipeline run
on one test cohort. Ranking per block and averaging afterwards is what makes the
summary robust: a block that a model fails on costs it rank positions instead of
deleting it from the comparison, and every block contributes exactly one
observation per model, which is the design the Friedman test assumes.

One metric at a time. AUROC and AUPRC are different scales whose orderings need
not agree, so pooling their ranks would average two questions into one answer
that matches neither.

The Nemenyi post-hoc test and the critical difference come from
``scikit-posthocs`` rather than from a local reimplementation, so the diagram and
the p-values are the published ones.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats

from src.schemas.training_schemas import LOWER_IS_BETTER_SCORING


@dataclass(frozen=True)
class RankSummary:
    """Block-level ranks for one metric, with per-setting and overall averages."""

    metric: str
    block_ranks: pd.DataFrame
    block_matrix: pd.DataFrame
    by_setting: pd.DataFrame
    overall: pd.DataFrame
    tests: pd.DataFrame
    model_metadata: pd.DataFrame
    model_instances: tuple[str, ...]
    settings: tuple[str, ...]
    datasets: tuple[str, ...]
    excluded_models: tuple[str, ...]
    alpha: float

    @property
    def block_count(self) -> int:
        return int(self.block_matrix.shape[0])

    @property
    def model_count(self) -> int:
        return int(self.block_matrix.shape[1])

    @property
    def run_count(self) -> int:
        return int(self.block_ranks["pipeline_mlflow_run_id"].nunique())

    def ranks_for(self, setting: str) -> pd.Series:
        """Return average ranks for one setting, indexed by model instance."""
        rows = self.by_setting.loc[self.by_setting["setting"].eq(setting)]
        return rows.set_index("model_instance")["mean_rank"]

    def overall_ranks(self) -> pd.Series:
        return self.overall.set_index("model_instance")["mean_rank"]

    def setting_ranks(self) -> pd.DataFrame:
        """Return average ranks as a settings x models table."""
        return (
            self.by_setting.pivot(index="setting", columns="model_instance", values="mean_rank")
            .reindex(list(self.settings))
            .reindex(columns=list(self.model_instances))
        )

    def labels(self) -> dict[str, str]:
        """Return the display label of every model instance."""
        return dict(zip(self.model_metadata["model_instance"], self.model_metadata["model_name"], strict=True))

    def matrix_for(self, setting: str | None = None) -> pd.DataFrame:
        """Return the block x model score matrix of one setting or the experiment."""
        if setting is None:
            return self.block_matrix
        blocks = self.block_ranks.loc[self.block_ranks["setting"].eq(setting), "block"].drop_duplicates()
        return self.block_matrix.loc[list(blocks)]

    def test_for(self, setting: str | None) -> pd.Series:
        """Return the Friedman-Nemenyi summary for a setting or the whole experiment."""
        scope = "experiment" if setting is None else "setting"
        rows = self.tests.loc[self.tests["scope"].eq(scope)]
        if setting is not None:
            rows = rows.loc[rows["setting"].eq(setting)]
        return rows.iloc[0]


def prepare_rank_summary(
    metrics: pd.DataFrame,
    *,
    metric: str,
    setting_by_run: Mapping[str, str],
    setting_order: Sequence[str] | None = None,
    alpha: float = 0.05,
) -> RankSummary:
    """Rank models inside every evaluation block and average the ranks.

    Ranks are averaged over repeated runs within a setting and then over settings
    within the experiment, so the Overall column answers "how consistently does
    this model rank where it ranks", not "how good was its best run".
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")

    required = {
        "pipeline_mlflow_run_id",
        "model_instance",
        "model_name",
        "scope",
        "statistic",
        "dataset",
        metric,
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError("Missing evaluation metric columns: " + ", ".join(missing))

    points = metrics.loc[metrics["scope"].eq("test") & metrics["statistic"].eq("point")].copy()
    if points.empty:
        raise ValueError("No scope='test', statistic='point' rows are available")
    points = points.assign(
        pipeline_mlflow_run_id=points["pipeline_mlflow_run_id"].astype(str),
        model_instance=points["model_instance"].astype(str),
        model_name=points["model_name"].astype(str),
        dataset=points["dataset"].astype(str),
    )

    run_map = {str(run_id): str(setting) for run_id, setting in setting_by_run.items()}
    run_ids = tuple(points["pipeline_mlflow_run_id"].drop_duplicates())
    unmapped = sorted(set(run_ids) - set(run_map))
    if unmapped:
        raise ValueError("setting_by_run does not assign every selected run: " + ", ".join(unmapped))
    settings = tuple(dict.fromkeys(setting_order or (run_map[run_id] for run_id in run_ids)))
    if set(settings) != {run_map[run_id] for run_id in run_ids}:
        raise ValueError("setting_order must contain every mapped setting exactly once")

    block_ranks = points[
        ["pipeline_mlflow_run_id", "model_instance", "model_name", "dataset", metric]
    ].rename(columns={metric: "score"})
    block_ranks["score"] = pd.to_numeric(block_ranks["score"], errors="coerce")
    if block_ranks["score"].isna().any() or not np.isfinite(block_ranks["score"].to_numpy(dtype=float)).all():
        raise ValueError(f"Metric {metric!r} contains non-finite values")
    block_ranks["setting"] = block_ranks["pipeline_mlflow_run_id"].map(run_map)
    block_ranks["block"] = block_ranks["pipeline_mlflow_run_id"] + "|" + block_ranks["dataset"]

    # Restricting happens before ranking: a rank is only meaningful relative to
    # the models that share the block, so dropping a model has to renumber the
    # rest rather than leave the survivors holding its positions.
    block_ranks, excluded_models = _restrict_to_common_models(block_ranks)
    block_matrix = _block_matrix(block_ranks, metric)
    stacked_ranks = (
        block_matrix.rank(axis=1, method="average")
        .stack()
        .rename("rank")
        .reset_index()
        .set_axis(["block", "model_instance", "rank"], axis="columns")
    )
    block_ranks = block_ranks.merge(stacked_ranks, on=["block", "model_instance"], how="left", validate="many_to_one")
    if block_ranks["rank"].isna().any():
        raise ValueError("Every block score must map to a rank")
    model_instances = tuple(block_matrix.columns.astype(str))
    model_metadata = block_ranks[["model_name", "model_instance"]].drop_duplicates().reset_index(drop=True)
    block_ranks = block_ranks.sort_values(["setting", "block", "rank"], kind="stable").reset_index(drop=True)

    by_setting = block_ranks.groupby(["setting", "model_instance", "model_name"], sort=False, as_index=False).agg(
        mean_rank=("rank", "mean"), sd_rank=("rank", "std"), block_count=("rank", "size")
    )
    by_setting["rank_position"] = by_setting.groupby("setting", sort=False)["mean_rank"].rank(method="average")
    by_setting["setting"] = pd.Categorical(by_setting["setting"], categories=list(settings), ordered=True)
    by_setting = by_setting.sort_values(["setting", "rank_position"], kind="stable").reset_index(drop=True)

    overall = block_ranks.groupby(["model_instance", "model_name"], sort=False, as_index=False).agg(
        mean_rank=("rank", "mean"), sd_rank=("rank", "std"), block_count=("rank", "size")
    )
    overall["rank_position"] = overall["mean_rank"].rank(method="average")
    overall = overall.sort_values("rank_position", kind="stable").reset_index(drop=True)

    blocks_by_setting = {
        setting: tuple(block_ranks.loc[block_ranks["setting"].eq(setting), "block"].drop_duplicates())
        for setting in settings
    }
    test_rows = [
        {
            "scope": "setting",
            "setting": setting,
            **_friedman_row(
                block_matrix.loc[list(blocks_by_setting[setting])],
                alpha,
            ),
        }
        for setting in settings
    ]
    test_rows.append({"scope": "experiment", "setting": None, **_friedman_row(block_matrix, alpha)})
    tests = pd.DataFrame(test_rows).assign(
        metric=metric,
        alpha=alpha,
        excluded_models=", ".join(excluded_models),
    )

    return RankSummary(
        metric=metric,
        block_ranks=block_ranks,
        block_matrix=block_matrix,
        by_setting=by_setting,
        overall=overall,
        tests=tests,
        model_metadata=model_metadata,
        model_instances=model_instances,
        settings=settings,
        datasets=tuple(block_ranks["dataset"].drop_duplicates().astype(str)),
        excluded_models=excluded_models,
        alpha=alpha,
    )


def friedman_test(block_matrix: pd.DataFrame) -> dict[str, float]:
    """Return the Friedman chi-square and the Iman-Davenport F correction.

    The uncorrected chi-square is optimistic for the small block counts these
    experiments produce, so the F statistic with ``(k-1, (k-1)(n-1))`` degrees of
    freedom is reported alongside it and used for the p-value printed by the
    figure scripts.
    """
    n_blocks, n_models = block_matrix.shape
    if n_models < 2:
        raise ValueError("The Friedman test needs at least two models")
    if n_blocks < 2:
        raise ValueError("The Friedman test needs at least two evaluation blocks")
    average = block_matrix.rank(axis=1, method="average").mean(axis=0)
    chi_square = 12.0 * n_blocks / (n_models * (n_models + 1)) * (
        float((average**2).sum()) - n_models * (n_models + 1) ** 2 / 4.0
    )
    chi_square = max(chi_square, 0.0)
    denominator = n_blocks * (n_models - 1) - chi_square
    if denominator <= 0:
        f_statistic, f_p_value = float("inf"), 0.0
    else:
        f_statistic = (n_blocks - 1) * chi_square / denominator
        f_p_value = float(stats.f.sf(f_statistic, n_models - 1, (n_models - 1) * (n_blocks - 1)))
    return {
        "chi_square": chi_square,
        "chi_square_p": float(stats.chi2.sf(chi_square, n_models - 1)),
        "f_statistic": f_statistic,
        "f_p_value": f_p_value,
    }


def nemenyi_critical_difference(n_blocks: int, n_models: int, alpha: float = 0.05) -> float:
    """Return the Nemenyi critical difference between two average ranks.

    The critical value is the exact upper quantile of the studentized range
    distribution divided by ``sqrt(2)``, which is the q_alpha that Demšar (2006)
    tabulates; using the distribution directly avoids a lookup table that stops at
    a fixed number of models.
    """
    if n_blocks < 2:
        raise ValueError("A critical difference needs at least two evaluation blocks")
    if n_models < 2:
        raise ValueError("A critical difference needs at least two models")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    q_alpha = stats.studentized_range.ppf(1.0 - alpha, n_models, np.inf) / sqrt(2.0)
    return float(q_alpha * sqrt(n_models * (n_models + 1) / (6.0 * n_blocks)))


def nemenyi_p_values(block_matrix: pd.DataFrame) -> pd.DataFrame:
    """Return the pairwise Nemenyi p-value matrix for one block design."""
    if block_matrix.isna().any().any():
        raise ValueError("Every evaluation block must contain exactly one score per model")
    return sp.posthoc_nemenyi_friedman(block_matrix)


def _block_matrix(block_ranks: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return the block x model score matrix, oriented so rank 1 is the best model."""
    wide = block_ranks.pivot_table(index="block", columns="model_instance", values="score", aggfunc="first")
    if wide.isna().any().any():
        raise ValueError("Every evaluation block must contain exactly one score per model")
    direction = -1.0 if metric in LOWER_IS_BETTER_SCORING else 1.0
    return direction * -wide


def _friedman_row(block_matrix: pd.DataFrame, alpha: float) -> dict[str, float | int]:
    n_blocks, n_models = block_matrix.shape
    statistics = friedman_test(block_matrix)
    return {
        "n_blocks": n_blocks,
        "n_models": n_models,
        "friedman_chi_square": statistics["chi_square"],
        "friedman_p": statistics["chi_square_p"],
        "iman_davenport_f": statistics["f_statistic"],
        "iman_davenport_p": statistics["f_p_value"],
        "critical_difference": nemenyi_critical_difference(n_blocks, n_models, alpha),
    }


def _restrict_to_common_models(block_ranks: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Keep only models that every evaluation block scored."""
    presence = block_ranks.groupby("block", sort=False)["model_instance"].agg(lambda values: set(values))
    common = set.intersection(*presence.tolist()) if len(presence) else set()
    if len(common) < 2:
        raise ValueError(
            "Fewer than two models appear in every evaluation block; ranks can only be compared on a common model set"
        )
    excluded = tuple(sorted(set(block_ranks["model_instance"]) - common))
    return block_ranks.loc[block_ranks["model_instance"].isin(common)].reset_index(drop=True), excluded
