"""Average model ranks and Friedman-Nemenyi critical differences.

Ranks are computed inside evaluation blocks, where a block is one pipeline run on
one test cohort. Ranking per block and averaging afterwards is what makes the
summary robust: a block that a model fails on costs it rank positions instead of
deleting it from the comparison, and every block contributes exactly one
observation per model, which is the design the Friedman test assumes.

One metric at a time. AUROC and AUPRC are different scales whose orderings need
not agree, so pooling their ranks would average two questions into one answer
that matches neither.

The Nemenyi post-hoc test comes from ``scikit-posthocs`` and the critical
difference from the exact studentized range quantile, so neither is re-derived
here.
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
    block_matrix: pd.DataFrame
    block_setting: pd.Series
    by_setting: pd.DataFrame
    overall: pd.DataFrame
    tests: pd.DataFrame
    model_metadata: pd.DataFrame
    model_instances: tuple[str, ...]
    settings: tuple[str, ...]
    excluded_models: tuple[str, ...]
    run_count: int
    alpha: float

    @property
    def block_count(self) -> int:
        return int(self.block_matrix.shape[0])

    @property
    def model_count(self) -> int:
        return int(self.block_matrix.shape[1])

    def labels(self) -> dict[str, str]:
        """Return the display label of every model instance."""
        return dict(zip(self.model_metadata["model_instance"], self.model_metadata["model_name"], strict=True))

    def ranks_for(self, setting: str) -> pd.Series:
        """Return average ranks for one setting, indexed by model instance, best first.

        Equal ranks keep the overall rank order, which is the order the rank diagram
        splits its labels on, so a tied model stays on the same label side in every
        panel of a per-setting figure.
        """
        return self.setting_ranks().loc[setting].sort_values(kind="stable")

    def overall_ranks(self) -> pd.Series:
        """Return average ranks over every block, indexed by model instance."""
        return self.overall.set_index("model_instance")["mean_rank"]

    def setting_ranks(self) -> pd.DataFrame:
        """Return average ranks as a settings x models table."""
        return self._setting_table("mean_rank")

    def setting_spread(self) -> pd.DataFrame:
        """Return the run-to-run rank spread as a settings x models table."""
        return self._setting_table("sd_rank")

    def matrix_for(self, setting: str | None = None) -> pd.DataFrame:
        """Return the block x model score matrix of one setting or the experiment."""
        if setting is None:
            return self.block_matrix
        blocks = self.block_setting.index[self.block_setting.astype(str).eq(setting)]
        return self.block_matrix.loc[list(blocks)]

    def test_for(self, setting: str | None) -> pd.Series:
        """Return the Friedman-Nemenyi summary for a setting or the whole experiment."""
        scope = "experiment" if setting is None else "setting"
        rows = self.tests.loc[self.tests["scope"].eq(scope)]
        if setting is not None:
            rows = rows.loc[rows["setting"].eq(setting)]
        return rows.iloc[0]

    def _setting_table(self, column: str) -> pd.DataFrame:
        return (
            self.by_setting.pivot(index="setting", columns="model_instance", values=column)
            .reindex(list(self.settings))
            .reindex(columns=list(self.model_instances))
        )


def prepare_rank_summary(
    metrics: pd.DataFrame,
    *,
    metric: str,
    setting_by_run: Mapping[str, str],
    setting_order: Sequence[str] | None = None,
    alpha: float = 0.05,
) -> RankSummary:
    """Rank models inside every evaluation block and average the ranks.

    Ranks are averaged over repeated runs within a setting and then over settings,
    so the overall average answers "how consistently does this model rank where it
    ranks", not "how good was its best run".
    """
    points = metrics.loc[metrics["scope"].eq("test") & metrics["statistic"].eq("point")].copy()
    if points.empty:
        raise ValueError("No scope='test', statistic='point' rows are available")
    points["run"] = points["pipeline_mlflow_run_id"].astype(str)
    points["model_instance"] = points["model_instance"].astype(str)
    points["block"] = points["run"] + "|" + points["dataset"].astype(str)
    points["setting"] = points["run"].map({str(run): str(setting) for run, setting in setting_by_run.items()})
    points["score"] = pd.to_numeric(points[metric], errors="coerce")
    if points["score"].isna().any():
        raise ValueError(f"Metric {metric!r} has missing values")

    # A rank is only meaningful relative to the models that share the block, so a
    # model missing from any block is dropped before ranking rather than left
    # holding positions that the survivors should have.
    coverage = points.groupby("model_instance")["block"].nunique()
    shared = coverage.index[coverage.eq(points["block"].nunique())]
    excluded = tuple(sorted(set(points["model_instance"]) - set(shared)))
    points = points.loc[points["model_instance"].isin(shared)]

    # The score matrix is oriented so that rank 1 is the best model, which is what
    # both the per-block ranks and the post-hoc test assume.
    block_matrix = points.pivot(index="block", columns="model_instance", values="score")
    if metric not in LOWER_IS_BETTER_SCORING:
        block_matrix = -block_matrix

    # Ranks are attached to the artifact rows and then sorted, so that equal ranks
    # read in model order instead of inheriting the artifact's row order.
    rank_table = block_matrix.rank(axis=1, method="average").stack().rename("rank").reset_index()
    rank_table.columns = ["block", "model_instance", "rank"]
    ranks = points[["setting", "block", "model_instance", "model_name"]].merge(
        rank_table, on=["block", "model_instance"], how="left"
    )
    ranks = ranks.sort_values(["setting", "block", "rank", "model_instance"], kind="stable").reset_index(drop=True)

    settings = _settings(ranks, setting_order)
    by_setting = ranks.groupby(["setting", "model_instance", "model_name"], as_index=False).agg(
        mean_rank=("rank", "mean"), sd_rank=("rank", "std")
    )
    by_setting["setting"] = pd.Categorical(by_setting["setting"], categories=list(settings), ordered=True)
    by_setting = by_setting.sort_values(["setting", "mean_rank", "model_instance"], kind="stable").reset_index(
        drop=True
    )
    overall = (
        ranks.groupby(["model_instance", "model_name"], as_index=False)
        .agg(mean_rank=("rank", "mean"), sd_rank=("rank", "std"))
        .sort_values(["mean_rank", "model_instance"], kind="stable")
        .reset_index(drop=True)
    )

    blocks_by_setting = {
        setting: list(ranks.loc[ranks["setting"].eq(setting), "block"].unique()) for setting in settings
    }
    tests = pd.DataFrame(
        [
            {"scope": "setting", "setting": setting, **_test_row(block_matrix.loc[blocks], alpha)}
            for setting, blocks in blocks_by_setting.items()
        ]
        + [{"scope": "experiment", "setting": None, **_test_row(block_matrix, alpha)}]
    )

    return RankSummary(
        metric=metric,
        block_matrix=block_matrix,
        block_setting=ranks.drop_duplicates("block").set_index("block")["setting"],
        by_setting=by_setting,
        overall=overall,
        tests=tests,
        model_metadata=points[["model_name", "model_instance"]].drop_duplicates().reset_index(drop=True),
        # Rank order, not alphabetical: it fixes the drawing order of the rank
        # figures, and it is the order a reader expects the labels in.
        model_instances=tuple(overall["model_instance"].astype(str)),
        settings=settings,
        excluded_models=excluded,
        run_count=int(points["run"].nunique()),
        alpha=alpha,
    )


def nemenyi_critical_difference(n_blocks: int, n_models: int, alpha: float = 0.05) -> float:
    """Return the Nemenyi critical difference between two average ranks.

    The critical value is the exact upper quantile of the studentized range
    distribution divided by ``sqrt(2)``, the q_alpha that Demšar (2006) tabulates.
    """
    q_alpha = stats.studentized_range.ppf(1.0 - alpha, n_models, np.inf) / sqrt(2.0)
    return float(q_alpha * sqrt(n_models * (n_models + 1) / (6.0 * n_blocks)))


def nemenyi_p_values(block_matrix: pd.DataFrame) -> pd.DataFrame:
    """Return the pairwise Nemenyi p-value matrix for one block design."""
    return sp.posthoc_nemenyi_friedman(block_matrix)


def _settings(ranks: pd.DataFrame, setting_order: Sequence[str] | None) -> tuple[str, ...]:
    """Return the settings in the requested order, appending any the caller missed."""
    present = list(dict.fromkeys(ranks["setting"].astype(str)))
    if setting_order is None:
        return tuple(present)
    ordered = [str(setting) for setting in setting_order if str(setting) in present]
    return tuple(ordered + [setting for setting in present if setting not in ordered])


def _test_row(block_matrix: pd.DataFrame, alpha: float) -> dict[str, float | int]:
    """Run the Friedman test on one block design and add its critical difference."""
    n_blocks, n_models = block_matrix.shape
    average = block_matrix.rank(axis=1, method="average").mean(axis=0)
    chi_square = (
        12.0
        * n_blocks
        / (n_models * (n_models + 1))
        * (float((average**2).sum()) - n_models * (n_models + 1) ** 2 / 4.0)
    )
    chi_square = max(chi_square, 0.0)
    denominator = n_blocks * (n_models - 1) - chi_square
    f_statistic = float("inf") if denominator <= 0 else (n_blocks - 1) * chi_square / denominator
    # The uncorrected chi-square is optimistic at these block counts, so the
    # Iman-Davenport F correction carries the reported p-value.
    f_p_value = (
        0.0
        if f_statistic == float("inf")
        else float(stats.f.sf(f_statistic, n_models - 1, (n_models - 1) * (n_blocks - 1)))
    )
    return {
        "n_blocks": n_blocks,
        "n_models": n_models,
        "friedman_p": float(stats.chi2.sf(chi_square, n_models - 1)),
        "iman_davenport_p": f_p_value,
        "critical_difference": nemenyi_critical_difference(n_blocks, n_models, alpha),
    }
