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
from dataclasses import dataclass, replace

from src.plotting.defaults import dataset_label, metric_label
from src.plotting.utils import (
    PairwiseSummary,
    RankSummary,
    aggregate_evaluation_runs,
    load_plot_artifacts,
    prepare_pairwise_summary,
    prepare_rank_summary,
)
from src.plotting.utils.pairwise import pairwise_matrix_to_latex, rank_table_to_latex
from src.plotting.utils.settings import assign_settings, setting_display

# ---------------------------------------------------------------------------
# EDIT THESE SETTINGS
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TABLES
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# LOCAL HELPERS
# ---------------------------------------------------------------------------


def table_requirements() -> str:
    """Return the LaTeX preamble a shaded matrix table needs."""
    return (
        "% Tables need \\usepackage{booktabs, graphicx, colortbl} and the \\definecolor lines printed with each table."
    )


# ---------------------------------------------------------------------------
# DATA SELECTION AND EXECUTION
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default=DATA.experiment_name)
    parser.add_argument(
        "--run-id",
        action="append",
        dest="run_ids",
        help="Explicit pipeline MLflow run ID for the matrix tables; repeat to average runs.",
    )
    parser.add_argument("--setting-source", choices=("training_size", "run_name"), default=DATA.setting_source)
    parser.add_argument("--setting-label", default=DATA.setting_label)
    parser.add_argument(
        "--setting-pattern",
        default=DATA.setting_pattern,
        help="Regex with one group, applied to pipeline run names when --setting-source is run_name.",
    )
    parser.add_argument(
        "--exclude-model",
        action="append",
        dest="exclude_models",
        help="Model name to leave out of every table; repeat for several models.",
    )
    parser.add_argument("--no-ci", action="store_true", help="Print no interval-based marks.")
    parser.add_argument(
        "--reference-model",
        default=TABLES.reference_model,
        help="Model name every paired difference is measured against; 'auto' anchors on the strongest external model.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.exclude_models or args.no_ci:
        global TABLES
        TABLES = replace(
            TABLES,
            exclude_models=tuple(args.exclude_models) if args.exclude_models else TABLES.exclude_models,
            ci_level=None if args.no_ci else TABLES.ci_level,
        )
    data = DataSettings(
        experiment_name=args.experiment_name,
        pipeline_runs=tuple(args.run_ids) if args.run_ids else DATA.pipeline_runs,
        full_training_only=DATA.full_training_only and not args.run_ids,
        rank_pipeline_runs=DATA.rank_pipeline_runs,
        setting_source=args.setting_source,
        setting_pattern=args.setting_pattern,
        setting_label=args.setting_label,
    )
    artifacts = load_plot_artifacts(
        data.experiment_name,
        pipeline_runs=data.pipeline_runs,
        exclude_models=TABLES.exclude_models,
        full_training_only=data.full_training_only,
    )
    aggregated = aggregate_evaluation_runs(artifacts, metrics=TABLES.metrics, ci_level=TABLES.ci_level)
    summary = prepare_pairwise_summary(
        aggregated,
        reference_model=None if args.reference_model in (None, "auto") else args.reference_model,
        alpha=TABLES.alpha,
    )

    rank_artifacts = load_plot_artifacts(
        data.experiment_name,
        pipeline_runs=data.rank_pipeline_runs,
        exclude_models=TABLES.exclude_models,
        include_bootstrap=False,
    )
    setting_by_run, setting_order = assign_settings(
        rank_artifacts.metrics,
        rank_artifacts.run_ids,
        source=data.setting_source,
        pattern=data.setting_pattern,
    )

    print(table_requirements())
    print("Selected pipeline runs: " + ", ".join(summary.run_ids))

    for metric in TABLES.metrics:
        for dataset in TABLES.matrix_datasets or summary.datasets:
            print(f"\n%% Split-encoded pairwise matrix: {dataset} / {metric}")
            print(matrix_table(summary, dataset, metric))

    for metric in TABLES.metrics:
        ranks = prepare_rank_summary(
            rank_artifacts.metrics,
            metric=metric,
            setting_by_run=setting_by_run,
            setting_order=setting_order,
            alpha=TABLES.alpha,
        )
        print(f"\n%% Average rank table: {metric} ({ranks.block_count} blocks)")
        print(rank_table(ranks, data.setting_label, data.setting_source))


if __name__ == "__main__":
    main()
