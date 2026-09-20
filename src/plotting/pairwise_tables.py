"""Pairwise comparison LaTeX tables.

Regenerate with:

    uv run python -m src.plotting.pairwise_tables

The tables are printed to the console rather than written to disk, matching how
the other table generators in this repository are used. The figures for the same
data, including their captions, live in ``src.plotting.pairwise_wins``.

Two tables per metric:

* a split-encoded pairwise matrix by test cohort, with win shares above and mean
  metric differences below the diagonal,
* average model ranks per setting and across the experiment, with the block count
  and the Nemenyi critical difference of every column.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from src.plotting.defaults import dataset_label, metric_label
from src.plotting.utils import PairwiseSummary, RankSummary, load_pairwise_inputs
from src.plotting.utils.pairwise import pairwise_matrix_to_latex, rank_table_to_latex
from src.plotting.utils.settings import setting_display


@dataclass(frozen=True)
class DataSettings:
    """Experiment selection, setting definition, and output location."""

    experiment_name: str = "sample_size_mimic_mortality"
    pipeline_runs: tuple[str, ...] | None = None
    full_training_only: bool = True
    rank_pipeline_runs: tuple[str, ...] | None = None
    setting_source: str = "training_size"
    setting_pattern: str = r"fraction-(\d+-\d+|\d+)"
    setting_label: str = "Training rows"


@dataclass(frozen=True)
class TableSettings:
    """All locally editable table choices."""

    metrics: tuple[str, ...] = ("roc_auc", "prc_auc")
    matrix_datasets: tuple[str, ...] | None = None
    ci_level: float = 0.95
    alpha: float = 0.05
    reference_model: str | None = "xgboost"
    exclude_models: tuple[str, ...] | None = None
    score_scale: float = 100.0
    win_digits: int = 1
    delta_digits: int = 2
    rank_digits: int = 2


DATA = DataSettings()
TABLES = TableSettings()


def matrix_table(
    summary: PairwiseSummary,
    dataset: str,
    metric: str,
    tables: TableSettings = TABLES,
) -> str:
    """Return the split-encoded pairwise matrix table of one cohort and metric."""
    return pairwise_matrix_to_latex(
        summary,
        dataset=dataset,
        metric=metric,
        headline=(
            rf"\textbf{{Pairwise {metric_label(metric)} comparison of every model pair on "
            rf"{dataset_label(dataset)}.}}"
        ),
        label=f"tab:pairwise-{summary.target}-{dataset}-{metric}",
        scale=tables.score_scale,
        win_digits=tables.win_digits,
        delta_digits=tables.delta_digits,
    )


def rank_table(ranks: RankSummary, setting_label: str, setting_source: str, tables: TableSettings = TABLES) -> str:
    """Return the average-rank table of one metric."""
    return rank_table_to_latex(
        ranks,
        headline=rf"\textbf{{Average model rank per configuration and across the experiment "
        rf"({metric_label(ranks.metric)}).}}",
        label=f"tab:pairwise-ranks-{ranks.metric}",
        setting_label=setting_label,
        setting_display={setting: setting_display(setting, setting_source) for setting in ranks.settings},
        digits=tables.rank_digits,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default=DATA.experiment_name)
    parser.add_argument(
        "--run-id",
        action="append",
        dest="run_ids",
        help="Explicit pipeline MLflow run ID for the matrix tables; repeat to average runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary, ranks = load_pairwise_inputs(
        args.experiment_name,
        metrics=TABLES.metrics,
        exclude_models=TABLES.exclude_models,
        ci_level=TABLES.ci_level,
        alpha=TABLES.alpha,
        reference_model=TABLES.reference_model,
        pipeline_runs=tuple(args.run_ids) if args.run_ids else DATA.pipeline_runs,
        rank_pipeline_runs=DATA.rank_pipeline_runs,
        full_training_only=DATA.full_training_only,
        setting_source=DATA.setting_source,
        setting_pattern=DATA.setting_pattern,
    )

    print("% Tables need \\usepackage{booktabs, graphicx, colortbl} and the \\definecolor lines printed below.")
    print("Selected pipeline runs: " + ", ".join(summary.run_ids))

    for metric in TABLES.metrics:
        for dataset in TABLES.matrix_datasets or summary.datasets:
            print(f"\n%% Split-encoded pairwise matrix: {dataset} / {metric}")
            print(matrix_table(summary, dataset, metric))

    for metric, metric_ranks in ranks.items():
        print(f"\n%% Average rank table: {metric} ({metric_ranks.block_count} blocks)")
        print(rank_table(metric_ranks, DATA.setting_label, DATA.setting_source))


if __name__ == "__main__":
    main()
