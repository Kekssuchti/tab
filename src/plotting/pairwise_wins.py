"""Pairwise model comparison figures.

Regenerate with:

    uv run python -m src.plotting.pairwise_wins

Edit ``DATA`` to select the experiment and ``VISUAL`` to change presentation.
The LaTeX tables for the same data live in ``src.plotting.pairwise_tables``.

Three questions, three figures:

* where does each model win, and does that survive the shift to another cohort
  (the cross-cohort matrix),
* by how much, against one reference model (the paired difference forest),
* how do the ranks move with the experimental setting (the rank figures).
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from matplotlib.figure import Figure

from src.config import config
from src.plotting.defaults import (
    PAIRWISE_CMAP,
    RANK_AXIS_LABEL,
    RANK_GROUP_COLOR,
    dataset_label,
    metric_label,
    model_label,
    set_plot_style,
)
from src.plotting.scientific_figstyle import BASELINE, WIDE, figure, figure_grid, label_ends, panel_labels, save
from src.plotting.utils import PairwiseSummary, RankSummary, load_pairwise_inputs
from src.plotting.utils.ranking import nemenyi_p_values
from src.plotting.utils.rendering import draw_model_forest, instance_plot_styles
from src.plotting.utils.settings import setting_display


@dataclass(frozen=True)
class DataSettings:
    """Experiment selection, setting definition, and output location."""

    experiment_name: str = "sample_size_mimic_mortality"
    pipeline_runs: tuple[str, ...] | None = None
    full_training_only: bool = True
    rank_pipeline_runs: tuple[str, ...] | None = None
    # "training_size" groups runs by their training row count; "run_name"
    # extracts a setting from the pipeline run name with `setting_pattern`.
    setting_source: str = "training_size"
    setting_pattern: str = r"fraction-(\d+-\d+|\d+)"
    setting_label: str = "Training rows"
    output_dir: Path = config.dir_plots / "pairwise"


@dataclass(frozen=True)
class VisualSettings:
    """All locally editable presentation choices for these figures."""

    comparison_metrics: tuple[str, ...] = ("roc_auc", "prc_auc")
    cross_metrics: tuple[str, ...] = ("roc_auc",)
    rank_metrics: tuple[str, ...] = ("roc_auc",)
    ci_level: float | None = 0.95
    alpha: float = 0.05
    reference_model: str | None = "xgboost"
    exclude_models: tuple[str, ...] | None = ("ebm", "tabicl-2")

    figure_width: float = WIDE
    score_scale: float = 100.0
    output_formats: tuple[str, ...] = ("pdf",)
    marker_size: float = 4.4
    line_width: float = 1.0
    paired_digits: int = 1

    cross_height: float = 0.92
    cross_cell_digits: int = 0
    rank_diagram_height: float = 1.6
    trajectory_height: float = 1.7
    movement_height: float = 1.6
    setting_panel_height: float = 1.25
    show_rank_movement: bool = True
    show_rank_spread: bool = False
    setting_panels: bool = True


DATA = DataSettings()
VISUAL = VisualSettings()

# Below this many blocks a post-hoc test cannot separate any pair, which the
# per-setting figures report rather than hide.
_MINIMUM_SETTING_BLOCKS = 4


# ---------------------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------------------


def make_cross_cohort_figure(summary: PairwiseSummary, metric: str, visual: VisualSettings = VISUAL) -> Figure:
    """Draw each pair's in-domain win share above and external share below the diagonal.

    The split keeps both cohorts in one cell instead of plotting their
    difference: a difference of zero is ambiguous, while two win shares either
    agree about the winner or they do not.
    """
    set_plot_style()
    cells = summary.cross_cells.loc[summary.cross_cells["metric"].eq(metric)]
    if cells.empty:
        raise ValueError(f"No cross-cohort pairs for metric {metric!r}")
    ordered = summary.order_for(summary.trained_on, metric)
    instances = list(ordered["model_instance"])
    styles = instance_plot_styles(summary.model_metadata)
    names = [styles[instance][1] for instance in instances]
    size = len(instances)
    position = {instance: index for index, instance in enumerate(instances)}

    values = np.full((size, size), np.nan)
    for row in cells.itertuples():
        top, bottom = position[row.row_instance], position[row.column_instance]
        # Both triangles read row over column: the upper is the in-domain cohort,
        # the lower is the external one, so the lower cell holds the share of the
        # model that the in-domain ranking puts second.
        values[top, bottom] = visual.score_scale * float(row.share_internal)
        values[bottom, top] = visual.score_scale * (1.0 - float(row.share_external))

    fig, ax = figure(width=visual.figure_width, ratio=visual.cross_height)
    color_map = plt.get_cmap(PAIRWISE_CMAP).copy()
    color_map.set_bad("#FFFFFF")
    image = ax.imshow(values, cmap=color_map, vmin=0.0, vmax=visual.score_scale, interpolation="nearest")
    ax.set_xticks(range(size), names, rotation=90, ha="left", va="center", rotation_mode="anchor")
    ax.set_yticks(range(size), names)
    # Column labels sit on top: rotated bottom labels collide with the row labels
    # in the bottom-left corner, which layout padding alone cannot resolve.
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    # ax.set_xlabel("Column model")
    # ax.set_ylabel("Row model")
    ax.tick_params(length=0)
    ax.grid(visible=False)
    # The house style hides the top and right spines; a matrix needs all four, or
    # the cells on those sides look like they run off the figure.
    for spine in ax.spines.values():
        spine.set_visible(False)
    #    spine.set_color(BASELINE)
    #    spine.set_linewidth(0.8)

    for (row_index, column_index), value in np.ndenumerate(values):
        if not np.isfinite(value):
            continue
        share = value / visual.score_scale
        ax.text(
            column_index,
            row_index,
            f"{value:.{visual.cross_cell_digits}f}",
            ha="center",
            va="center",
            fontsize=5.5,
            color="white" if abs(share - 0.5) > 0.3 else "black",
        )

    for index in range(1, size):
        ax.plot([-0.5, size - 0.5], [index - 0.5, index - 0.5], color="#FFFFFF", linewidth=0.8, zorder=3)
        ax.plot([index - 0.5, index - 0.5], [-0.5, size - 0.5], color="#FFFFFF", linewidth=0.8, zorder=3)

    colorbar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.045)
    colorbar.set_label(
        f"Win share ({metric_label(metric)}, %): {dataset_label(summary.trained_on)} above the diagonal, "
        f"{dataset_label(summary.external_dataset)} below"
    )
    return fig


def make_paired_forest_figure(summary: PairwiseSummary, visual: VisualSettings = VISUAL) -> Figure:
    """Draw each model's paired difference against the reference model."""
    set_plot_style()
    datasets = (summary.trained_on, summary.external_dataset)
    fig, axes = figure_grid(
        len(summary.metrics),
        len(datasets),
        width=visual.figure_width,
        ratio=0.34 * len(summary.metrics),
        sharey=True,
        squeeze=False,
    )
    styles = instance_plot_styles(summary.model_metadata)
    reference_label = styles[summary.reference_instance][1]
    instances = list(summary.model_instances)

    for metric_index, metric in enumerate(summary.metrics):
        rows = summary.forest.loc[summary.forest["metric"].eq(metric)]
        for dataset_index, dataset in enumerate(datasets):
            ax = axes[metric_index, dataset_index]
            panel = rows.loc[rows["dataset"].eq(dataset)].set_index("model_instance").loc[instances].reset_index()
            show_intervals = summary.ci_level is not None
            draw_model_forest(
                ax,
                panel,
                summary.model_instances,
                styles,
                scale=visual.score_scale,
                show_ci=show_intervals,
                show_model_labels=dataset_index == 0,
                marker_size=visual.marker_size,
            )
            ax.axvline(0, color=BASELINE, linewidth=0.8, linestyle="--", zorder=1)
            # Without intervals axis is sized from the point estimates only
            columns = ["lower", "upper"] if show_intervals else ["estimate"]
            spread = panel[columns].to_numpy(dtype=float)
            limit = max(float(np.nanmax(np.abs(spread))) * visual.score_scale * 1.35, 1.0)
            ax.set_xlim(-limit, limit)
            for position, row in enumerate(panel.itertuples(), start=0):
                text = (
                    "reference"
                    if row.is_reference
                    else f"{visual.score_scale * float(row.win_share):.{visual.paired_digits}f}%"
                )
                ax.text(limit * 0.985, position, text, ha="right", va="center", fontsize=5.5, color=BASELINE)
            ax.set_xlabel(f"{metric_label(metric)} vs {reference_label} on {dataset_label(dataset)}")
            ax.set_ylabel("Model" if dataset_index == 0 else "")

    panel_labels(axes)
    return fig


def make_rank_figure(ranks: RankSummary, setting_label: str, visual: VisualSettings = VISUAL) -> Figure:
    """Draw the experiment-level ranks, their path across settings, and their net movement."""
    set_plot_style()
    styles = instance_plot_styles(ranks.model_metadata)
    show_movement = visual.show_rank_movement and len(ranks.settings) > 1
    heights = [visual.rank_diagram_height, visual.trajectory_height]
    if show_movement:
        heights.append(visual.movement_height)
    fig, axes = figure_grid(
        len(heights),
        1,
        width=visual.figure_width,
        ratio=(sum(heights) + 0.4) / visual.figure_width,
        gridspec_kw={"height_ratios": heights},
        squeeze=False,
    )
    draw_rank_diagram(axes[0, 0], ranks, styles, setting=None)
    _draw_rank_trajectory(axes[1, 0], ranks, styles, setting_label, visual)
    if show_movement:
        _draw_rank_movement(axes[2, 0], ranks, styles, visual)
    panel_labels(axes)
    return fig


def make_setting_rank_figure(ranks: RankSummary, setting_source: str, visual: VisualSettings = VISUAL) -> Figure:
    """Draw one critical difference diagram per experimental setting."""
    set_plot_style()
    styles = instance_plot_styles(ranks.model_metadata)
    settings = list(ranks.settings)
    fig, axes = figure_grid(
        len(settings),
        1,
        width=visual.figure_width,
        ratio=(visual.setting_panel_height * len(settings) + 0.4) / visual.figure_width,
        squeeze=False,
    )
    labels = []
    for index, setting in enumerate(settings):
        draw_rank_diagram(axes[index, 0], ranks, styles, setting=setting)
        labels.append(f"({chr(ord('a') + index)}) {setting_display(setting, setting_source)}")
    panel_labels(axes, labels)
    return fig


def draw_rank_diagram(
    ax,
    ranks: RankSummary,
    styles: Mapping[str, tuple[object, str]],
    *,
    setting: str | None,
) -> None:
    """Draw one scikit-posthocs Nemenyi diagram from a rank summary.

    The post-hoc test, the maximal-clique crossbars, and the label layout all come
    from ``scikit_posthocs``. The critical difference is reported in the caption
    rather than drawn as a scale bar, which the library places on top of the rank
    numbers.
    """
    average = ranks.ranks_for(setting) if setting is not None else ranks.overall_ranks()
    display = ranks.labels()
    matrix = ranks.matrix_for(setting)
    label_by_instance = {instance: display[instance] for instance in average.index}
    p_values = nemenyi_p_values(matrix).rename(index=label_by_instance, columns=label_by_instance)

    sp.critical_difference_diagram(
        pd.Series({label_by_instance[instance]: float(average[instance]) for instance in average.index}),
        p_values,
        alpha=ranks.alpha,
        ax=ax,
        color_palette={label_by_instance[instance]: styles[instance][0].color for instance in average.index},
        label_fmt_left="{label} ({rank:.2g})",
        label_fmt_right="({rank:.2g}) {label}",
        crossbar_props={"color": RANK_GROUP_COLOR, "linewidth": 1.4},
        elbow_props={"linewidth": 0.8},
        marker_props={"s": 14},
    )
    ax.set_xlabel(RANK_AXIS_LABEL)
    _style_rank_axis(ax, ranks.model_count)


def _style_rank_axis(ax, model_count: int) -> None:
    """Finish the rank axis the library leaves half-drawn.

    Two separate gaps need closing. The library autoscales x to its elbow lines,
    which start left of the best rank, so the extreme ticks fall outside the view
    and get dropped; and it uses the top spine as the rank line, which the house
    style switches off. The upper quarter of the panel also has to stay empty,
    because the rank numbers sit above the axis line and the panel letter sits
    above the axes.
    """
    ax.set_xlim(0.5, model_count + 0.5)
    spine = ax.spines["top"]
    spine.set_position("zero")
    spine.set_visible(True)
    spine.set_color(RANK_GROUP_COLOR)
    spine.set_linewidth(0.8)
    ax.tick_params(axis="x", length=2.6, width=0.8, color=RANK_GROUP_COLOR, pad=1.2)
    ax.tick_params(axis="x", which="minor", length=0)
    bottom, _ = ax.get_ylim()
    if np.isfinite(bottom) and bottom < 0:
        ax.set_ylim(bottom, -bottom / 3.0)


def _draw_rank_trajectory(
    ax,
    ranks: RankSummary,
    styles: Mapping[str, tuple[object, str]],
    setting_label: str,
    visual: VisualSettings,
) -> None:
    """Draw each model's average rank across the experimental settings."""
    settings = list(ranks.settings)
    positions = np.arange(len(settings), dtype=float)
    table = ranks.setting_ranks()
    spread = ranks.setting_spread()
    lines = []
    for instance in ranks.model_instances:
        values = table[instance].to_numpy(dtype=float)
        style, _ = styles[instance]
        (line,) = ax.plot(
            positions,
            values,
            marker=style.marker,
            markersize=visual.marker_size,
            color=style.color,
            linewidth=visual.line_width,
            zorder=3,
        )
        lines.append(line)
        run_spread = spread[instance].to_numpy(dtype=float)
        if visual.show_rank_spread and np.isfinite(run_spread).any():
            ax.errorbar(
                positions,
                values,
                yerr=np.nan_to_num(run_spread),
                fmt="none",
                ecolor=style.color,
                elinewidth=0.7,
                capsize=1.6,
                alpha=0.55,
                zorder=2,
            )
    ax.set_xticks(positions, [setting_display(setting, DATA.setting_source) for setting in settings])
    ax.set_xlim(-0.4, len(settings) - 0.35)
    ax.invert_yaxis()
    ax.set_yticks(range(1, ranks.model_count + 1))
    ax.set_ylabel(RANK_AXIS_LABEL)
    ax.set_xlabel(setting_label)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    label_ends(ax, lines, [styles[instance][1] for instance in ranks.model_instances])


def _draw_rank_movement(
    ax,
    ranks: RankSummary,
    styles: Mapping[str, tuple[object, str]],
    visual: VisualSettings,
) -> None:
    """Draw how far each model moves between the smallest and the largest setting."""
    settings = list(ranks.settings)
    table = ranks.setting_ranks()
    start_setting, end_setting = settings[0], settings[-1]
    instances = list(table.loc[end_setting].sort_values(kind="stable").index)
    positions = np.arange(len(instances), dtype=float)

    for position, instance in zip(positions, instances, strict=True):
        style, _ = styles[instance]
        start, end = float(table.loc[start_setting, instance]), float(table.loc[end_setting, instance])
        ax.plot([start, end], [position, position], color=BASELINE, linewidth=visual.line_width, zorder=2)
        ax.plot(
            [start],
            [position],
            linestyle="none",
            marker="o",
            markersize=visual.marker_size,
            markerfacecolor="white",
            markeredgecolor=style.color,
            markeredgewidth=0.9,
            zorder=3,
        )
        ax.plot(
            [end],
            [position],
            linestyle="none",
            marker=style.marker,
            markersize=visual.marker_size,
            color=style.color,
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=4,
        )
        ax.text(
            ranks.model_count + 0.25,
            position,
            f"{start:.2f} → {end:.2f} ({start - end:+.2f})",
            ha="left",
            va="center",
            fontsize=5.5,
            color=style.color,
        )

    ax.set_xlim(0.5, ranks.model_count + 0.5)
    ax.set_ylim(len(instances) - 0.5, -0.5)
    ax.set_yticks(positions, [styles[instance][1] for instance in instances])
    ax.tick_params(axis="y", length=0)
    ax.set_xticks(range(1, ranks.model_count + 1))
    ax.set_xlabel(RANK_AXIS_LABEL)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    legend_labels = (
        f"{setting_display(start_setting, DATA.setting_source)} {DATA.setting_label.lower()}",
        f"{setting_display(end_setting, DATA.setting_source)} {DATA.setting_label.lower()}",
    )
    for index, label in enumerate(legend_labels):
        ax.plot(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=visual.marker_size,
            markerfacecolor=BASELINE if index else "white",
            markeredgecolor=BASELINE,
            label=label,
        )
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01))


# ---------------------------------------------------------------------------
# CAPTIONS
# ---------------------------------------------------------------------------


def cross_caption(summary: PairwiseSummary, metric: str) -> str:
    cells = summary.cross_cells.loc[summary.cross_cells["metric"].eq(metric)]
    reversed_count = int(cells["reversed"].sum())
    return (
        rf"\textbf{{{reversed_count} of {len(cells)} comparisons change hands between the in-domain and the "
        rf"external cohort.}} "
        f"Each cell holds two win shares for one model pair: the share of {summary.bootstrap_count:,} paired "
        f"cohort-bootstrap draws in which the row model beat the column model, above the diagonal on the held-out "
        f"{dataset_label(summary.trained_on)} cohort and below it on the held-out "
        f"{dataset_label(summary.external_dataset)} cohort. Rows and columns are ordered by the in-domain "
        f"{metric_label(metric)} estimate, and the white diagonal separates the two halves. Both triangles read row "
        f"over column, so a pair whose winner does not change has one warm cell and one cool cell, while a pair that "
        f"changes hands is warm in both. Cells near 50 are undecided, and a share counts resamples of these cohorts "
        f"rather than the chance that a model generalizes better."
    )


def forest_caption(summary: PairwiseSummary, visual: VisualSettings = VISUAL) -> str:
    del visual
    styles = instance_plot_styles(summary.model_metadata)
    reference_label = styles[summary.reference_instance][1]
    metrics = " and ".join(metric_label(metric) for metric in summary.metrics)
    intervals = (
        f"whiskers are {round(100 * summary.ci_level)}\\% percentile intervals from {summary.bootstrap_count:,} "
        f"aligned cohort-bootstrap draws"
        if summary.ci_level is not None
        else "no intervals are shown"
    )
    return (
        r"\textbf{Effect sizes against one reference model anchor the win shares.} "
        f"Points are paired {metrics} differences against {reference_label}, so a positive value means the model beat "
        f"{reference_label} on the held-out {dataset_label(summary.trained_on)} cohort and a negative value means it "
        f"lost to it, and the same reading holds on {dataset_label(summary.external_dataset)}. "
        f"{intervals.capitalize()}. The number at the right of each row is the share of those draws in which the model "
        f"outperformed {reference_label}. Circles denote classical baselines and triangles tabular foundation models."
    )


def rank_caption(ranks: RankSummary, setting_label: str, visual: VisualSettings = VISUAL) -> str:
    test = ranks.test_for(None)
    p_value = float(test["iman_davenport_p"])
    p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
    movement = (
        "Bottom panel: average rank in the smallest and the largest setting, with the net movement annotated. "
        if visual.show_rank_movement and len(ranks.settings) > 1
        else ""
    )
    spread = (
        "with the run-to-run spread where repeated runs exist and the model name at the right end"
        if visual.show_rank_spread
        else "with the model name at the right end"
    )
    return (
        f"{_rank_headline(ranks)} "
        f"Top panel: Nemenyi critical difference diagram of the average rank over {ranks.block_count} evaluation "
        f"blocks, one per configuration and held-out cohort, from {ranks.run_count} pipeline runs. Models joined by a "
        f"bar cannot be separated at $\\alpha$ = {ranks.alpha:g} (Friedman {p_text}), and two average ranks have to "
        f"differ by {float(test['critical_difference']):.2f} rank positions to count as separated. Middle panel: "
        f"average rank per setting, {spread}. {movement}{_excluded_note(ranks)}"
    )


def _rank_headline(ranks: RankSummary) -> str:
    """State the ranking result the figure shows, without hard-coding a claim."""
    leader = model_label(str(ranks.labels()[ranks.overall_ranks().index[0]]))
    metric = metric_label(ranks.metric)
    if len(ranks.settings) < 2:
        return rf"\textbf{{Average {metric} rank over {ranks.block_count} evaluation blocks: {leader} leads.}}"
    table = ranks.setting_ranks()
    movement = (table.iloc[0] - table.iloc[-1]).abs()
    movers = int(movement.ge(1.0).sum())
    claim = (
        "no model moves a full rank position"
        if movers == 0
        else f"{movers} of {ranks.model_count} models move at least a rank position"
    )
    return (
        rf"\textbf{{Average {metric} rank over {ranks.block_count} evaluation blocks: {leader} leads, and {claim} "
        rf"between the lowest and the highest setting.}}"
    )


def _excluded_note(ranks: RankSummary) -> str:
    if not ranks.excluded_models:
        return ""
    names = ", ".join(model_label(name) for name in ranks.excluded_models)
    return (
        f"Models that are absent from at least one setting are left out, because a rank comparison needs the same "
        f"models in every block: {names}."
    )


def setting_rank_caption(ranks: RankSummary) -> str:
    return (
        r"\textbf{Per-setting critical difference diagrams show where the overall ranking comes from.} "
        "One diagram per configuration level, each with its own Nemenyi critical difference and block "
        "count. A setting that contributes only a few blocks cannot separate any pair, which the scale bar makes "
        "explicit."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default=DATA.experiment_name)
    parser.add_argument(
        "--run-id",
        action="append",
        dest="run_ids",
        help="Explicit pipeline MLflow run ID for the pairwise views; repeat to average runs.",
    )
    parser.add_argument("--output-dir", type=Path, default=DATA.output_dir)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary, ranks = load_pairwise_inputs(
        args.experiment_name,
        metrics=VISUAL.comparison_metrics,
        rank_metrics=VISUAL.rank_metrics,
        exclude_models=VISUAL.exclude_models,
        ci_level=VISUAL.ci_level,
        alpha=VISUAL.alpha,
        reference_model=VISUAL.reference_model,
        pipeline_runs=tuple(args.run_ids) if args.run_ids else DATA.pipeline_runs,
        rank_pipeline_runs=DATA.rank_pipeline_runs,
        full_training_only=DATA.full_training_only,
        setting_source=DATA.setting_source,
        setting_pattern=DATA.setting_pattern,
    )

    print("Selected pipeline runs (pairwise view): " + ", ".join(summary.run_ids))
    for metric, metric_ranks in ranks.items():
        print(f"\nRank summary for {metric}: {metric_ranks.block_count} blocks, {metric_ranks.model_count} models")
        if metric_ranks.excluded_models:
            print("  Models dropped (missing from some blocks): " + ", ".join(metric_ranks.excluded_models))
        for test in metric_ranks.tests.itertuples():
            scope = "experiment" if test.scope == "experiment" else f"setting {test.setting}"
            print(
                f"  {scope}: N = {int(test.n_blocks)}, k = {int(test.n_models)}, "
                f"Friedman p = {_p_value(test.friedman_p)}, Iman-Davenport p = {_p_value(test.iman_davenport_p)}, "
                f"CD = {test.critical_difference:.2f}"
            )

    output_dir = args.output_dir / summary.target
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"pairwise_{summary.trained_on}_{summary.target}"
    stems: list[Path] = []

    for metric in VISUAL.cross_metrics:
        stem = output_dir / f"{prefix}_{metric}_cross_cohort"
        save(make_cross_cohort_figure(summary, metric), str(stem), formats=VISUAL.output_formats)
        stems.append(stem)
        print(f"\nCross-cohort caption ({metric}):")
        print(f"\\caption{{{cross_caption(summary, metric)}}}")

    forest = output_dir / f"{prefix}_paired_forest"
    save(make_paired_forest_figure(summary), str(forest), formats=VISUAL.output_formats)
    stems.append(forest)
    print("\nPaired difference forest caption:")
    print(f"\\caption{{{forest_caption(summary)}}}")

    for metric, metric_ranks in ranks.items():
        rank_figure = output_dir / f"{prefix}_{metric}_ranks"
        save(make_rank_figure(metric_ranks, DATA.setting_label), str(rank_figure), formats=VISUAL.output_formats)
        stems.append(rank_figure)
        print(f"\nRank figure caption ({metric}):")
        print(f"\\caption{{{rank_caption(metric_ranks, DATA.setting_label)}}}")

        if VISUAL.setting_panels and len(metric_ranks.settings) > 1:
            smallest = int(metric_ranks.tests.loc[metric_ranks.tests["scope"].eq("setting"), "n_blocks"].min())
            if smallest < _MINIMUM_SETTING_BLOCKS:
                print(
                    f"  Note: a setting contributes only {smallest} blocks, so no per-setting diagram can separate a "
                    f"pair; set VISUAL.setting_panels = False to skip that figure."
                )
            setting_figure = output_dir / f"{prefix}_{metric}_ranks_by_setting"
            save(
                make_setting_rank_figure(metric_ranks, DATA.setting_source),
                str(setting_figure),
                formats=VISUAL.output_formats,
            )
            stems.append(setting_figure)
            print(f"Per-setting rank figure caption ({metric}):")
            print(f"\\caption{{{setting_rank_caption(metric_ranks)}}}")

    print("\nFigures:")
    for stem in stems:
        for extension in VISUAL.output_formats:
            print(str(stem.with_suffix(f".{extension}")).removeprefix(f"{config.dir_root}/"))


def _p_value(value: float) -> str:
    """Format a p-value that can underflow to exactly zero."""
    if value <= 0.0:
        return "< 1e-300"
    return f"{value:.4g}"


if __name__ == "__main__":
    main()
