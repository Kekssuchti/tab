"""F7 and F8: training-data augmentation with external observations.

Regenerate with:

    uv run python -m src.plotting.augmentation

F7 fixes the complete external training pool and varies the number of local
target-center observations. F8 fixes a local budget and varies the number of
added external observations. Both are declared per prediction task and target
center in `src.plotting.experiments`; a declaration without a registered
experiment, or without its matched local-only reference, is reported and skipped
rather than replaced by another experiment.

Every panel evaluates the designated target center. The external source is the
other center, so the figure keeps its training-source identity in the file name
and the caption.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.config import config
from src.plotting.defaults import dataset_label, metric_label, set_plot_style, task_label
from src.plotting.experiments import DATA_SOURCES, MAIN_TARGETS, tasks_for, warn_skipped
from src.plotting.scientific_figstyle import BASELINE, PALETTE, WIDE, figure_grid, panel_labels, save
from src.plotting.utils import (
    FixedLocalAugmentation,
    FixedLocalBudgetView,
    FullExternalAugmentation,
    IncompleteExperimentError,
    MissingExperimentError,
    PlotArtifacts,
    load_plot_artifacts,
    prepare_fixed_local_augmentation,
    prepare_full_external_augmentation,
)
from src.plotting.utils.augmentation import COMBINED, EXTERNAL_ONLY, LOCAL_ONLY
from src.plotting.utils.rendering import (
    instance_plot_styles,
    interval_axis_limits,
    log_sample_ticks,
    short_count,
)


@dataclass(frozen=True)
class DataSettings:
    """Run/model selection and F7/F8 output location."""

    pipeline_runs: tuple[str, ...] | None = None
    models: tuple[str, ...] | None = None
    exclude_models: tuple[str, ...] | None = None
    output_dir: Path = config.dir_plots / "augmentation"


@dataclass(frozen=True)
class VisualSettings:
    """Locally editable presentation choices for F7 and F8."""

    metrics: tuple[str, ...] = ("roc_auc", "prc_auc")
    show_ci: bool = True
    ci_level: float = 0.95
    score_scale: float = 100.0
    figure_width: float = WIDE
    panel_height_ratio: float = 0.34
    max_columns: int = 3
    marker_size: float = 3.8
    line_width: float = 1.1
    ci_line_width: float = 0.7
    cap_size: float = 1.8
    ci_alpha: float = 0.7
    max_x_ticks: int = 6
    # The external-only boundary sits left of the logarithmic range: the zero
    # marker is a position in a narrow gutter, not a measured local count.
    zero_gap_factor: float = 6.0
    zero_marker_position: float = 1.35
    show_external_baseline: bool = True
    external_baseline_color: str = BASELINE
    external_baseline_linestyle: str = "--"
    external_baseline_alpha: float = 1.0
    axis_padding_fraction: float = 0.08
    local_color: str = PALETTE["blue"]
    combined_color: str = PALETTE["orange"]
    reference_color: str = BASELINE
    output_formats: tuple[str, ...] = ("pdf",)


DATA = DataSettings()
VISUAL = VisualSettings()

_CONDITION_STYLES = {
    LOCAL_ONLY: {"marker": "o", "linestyle": "-"},
    COMBINED: {"marker": "^", "linestyle": "--"},
}

EXTERNAL_BASELINE_LABEL = "External-only baseline"


def make_full_external_figure(
    data: FullExternalAugmentation,
    metric: str,
    visual: VisualSettings = VISUAL,
) -> Figure:
    """Draw local-only versus full-external-plus-local curves for one metric."""
    set_plot_style()
    rows = _metric_rows(data.performance, data.metrics, metric, data.local_center)
    fig, axes, column_count = _model_grid(len(data.model_instances), visual)
    styles = instance_plot_styles(data.model_metadata)
    limits = interval_axis_limits(
        rows,
        scale=visual.score_scale,
        show_ci=visual.show_ci,
        padding_fraction=visual.axis_padding_fraction,
    )
    measured = tuple(count for count in data.local_counts if count > 0)
    positions = _sample_axis(axes[0], measured, visual, measured_zero=data.external_only_measured)

    for ax, instance in zip(axes, data.model_instances, strict=True):
        instance_rows = rows.loc[rows["model_instance"].astype(str).eq(instance)]
        for condition, members in ((LOCAL_ONLY, (LOCAL_ONLY,)), (COMBINED, (COMBINED, EXTERNAL_ONLY))):
            cell = instance_rows.loc[instance_rows["training_condition"].isin(members)].sort_values("local_count")
            if cell.empty:
                raise ValueError(f"Model {instance!r} has no {condition!r} augmentation cell for {metric!r}")
            _draw_condition(ax, cell, condition, positions, data, visual)
        _draw_external_baseline(ax, instance_rows, instance, visual)
        ax.set_ylim(limits)
        ax.grid(axis="both")
        ax.annotate(
            styles[instance][1],
            xy=(1, 1),
            xycoords="axes fraction",
            xytext=(-2, -2),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize="small",
            fontweight="bold",
        )

    axes[0].set_ylabel(f"{metric_label(metric)} on {dataset_label(data.local_center)}")
    for ax in _bottom_row(axes, column_count):
        ax.set_xlabel(f"{dataset_label(data.local_center)} training observations")
    legend = _condition_legend(data, visual)
    fig.legend(*legend, loc="outside upper center", ncol=2)
    panel_labels(axes)
    return fig


def make_fixed_local_figure(
    view: FixedLocalBudgetView,
    metric: str,
    visual: VisualSettings = VISUAL,
) -> Figure:
    """Draw the added-external curve against its local-only reference."""
    set_plot_style()
    rows = _metric_rows(view.performance, view.metrics, metric, view.local_center)
    fig, axes, column_count = _model_grid(len(view.model_instances), visual)
    styles = instance_plot_styles(view.model_metadata)
    limits = interval_axis_limits(
        rows,
        scale=visual.score_scale,
        show_ci=visual.show_ci,
        padding_fraction=visual.axis_padding_fraction,
    )
    external_counts = view.external_counts
    if not external_counts:
        raise ValueError("Fixed-local augmentation requires at least one positive added-external count")
    if len(external_counts) > 1:
        axes[0].set_xscale("log", base=2)
        axes[0].set_xlim(external_counts[0] / 1.6, external_counts[-1] * 1.6)
    else:
        axes[0].set_xlim(0.0, external_counts[0] * 2.0)
    ticks = log_sample_ticks(external_counts, visual.max_x_ticks)
    axes[0].set_xticks(ticks, [short_count(value) for value in ticks])
    axes[0].minorticks_off()

    for ax, instance in zip(axes, view.model_instances, strict=True):
        instance_rows = rows.loc[rows["model_instance"].astype(str).eq(instance)]
        reference = instance_rows.loc[instance_rows["external_count"].eq(0)]
        if len(reference) != 1:
            raise ValueError(f"Model {instance!r} needs exactly one local-only reference; found {len(reference)}")
        value = visual.score_scale * float(reference.iloc[0]["estimate"])
        ax.axhline(
            value,
            color=visual.reference_color,
            linewidth=visual.line_width,
            linestyle="--",
            label=f"Local only ({view.local_count:,})",
            zorder=1,
        )
        cell = instance_rows.loc[instance_rows["external_count"].gt(0)].sort_values("external_count")
        if cell.empty:
            raise ValueError(f"Model {instance!r} has no combined-training cell for {metric!r}")
        estimates = visual.score_scale * cell["estimate"].to_numpy(dtype=float)
        lower = visual.score_scale * cell["lower"].to_numpy(dtype=float)
        upper = visual.score_scale * cell["upper"].to_numpy(dtype=float)
        ax.errorbar(
            cell["external_count"].to_numpy(dtype=float),
            estimates,
            yerr=np.vstack((estimates - lower, upper - estimates)) if visual.show_ci else None,
            color=visual.combined_color,
            marker=_CONDITION_STYLES[COMBINED]["marker"],
            linestyle=_CONDITION_STYLES[COMBINED]["linestyle"],
            markersize=visual.marker_size,
            linewidth=visual.line_width,
            elinewidth=visual.ci_line_width,
            capsize=visual.cap_size if visual.show_ci else 0,
            alpha=visual.ci_alpha if visual.show_ci else 1.0,
            label="Local budget + added external",
            zorder=2,
        )
        ax.set_ylim(limits)
        ax.grid(axis="both")
        ax.annotate(
            styles[instance][1],
            xy=(1, 1),
            xycoords="axes fraction",
            xytext=(-2, -2),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize="small",
            fontweight="bold",
        )

    axes[0].set_ylabel(f"Absolute {metric_label(metric)} on {dataset_label(view.local_center)} (%)")
    for ax in _bottom_row(axes, column_count):
        ax.set_xlabel(f"Added {dataset_label(view.external_source)} training observations")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2)
    panel_labels(axes)
    return fig


def _metric_rows(performance: pd.DataFrame, metrics: tuple[str, ...], metric: str, center: str) -> pd.DataFrame:
    if metric not in metrics:
        raise ValueError(f"Metric {metric!r} is unavailable; prepared metrics: {list(metrics)}")
    rows = performance.loc[performance["metric"].eq(metric) & performance["dataset"].eq(center)].copy()
    if rows.empty:
        raise ValueError(f"No {metric!r} rows are available for evaluation center {center!r}")
    return rows


def _draw_condition(
    ax,
    cell: pd.DataFrame,
    condition: str,
    positions: dict[int, float],
    data: FullExternalAugmentation,
    visual: VisualSettings,
) -> None:
    color = visual.local_color if condition == LOCAL_ONLY else visual.combined_color
    style = _CONDITION_STYLES[condition]
    measured = cell.loc[cell["local_count"].gt(0)].sort_values("local_count").copy()
    measured["position"] = measured["local_count"].astype(int).map(positions).astype(float)
    zero = cell.loc[cell["local_count"].eq(0)].copy()
    if len(zero) > 1:
        raise ValueError(f"Found {len(zero)} external-only rows for {condition!r}; expected at most one setting")
    if len(zero) == 1:
        zero["position"] = positions[0]
    curve = pd.concat((zero, measured), ignore_index=True).sort_values("position")
    _draw_curve(ax, curve["position"], curve, color, style, label=condition, visual=visual)


def _draw_external_baseline(
    ax,
    instance_rows: pd.DataFrame,
    instance: str,
    visual: VisualSettings,
) -> None:
    """Draw the externally trained model's score as the level local data must beat.

    The value is that model's measured external-only score on this evaluation
    center, not a fitted threshold, and it is the same number the combined curve
    starts from at zero local observations. Set the show_external_baseline field
    of VisualSettings to False to leave the line out.
    """
    if not visual.show_external_baseline:
        return
    baseline = instance_rows.loc[instance_rows["training_condition"].eq(EXTERNAL_ONLY)]
    if baseline.empty:
        return
    if len(baseline) > 1:
        raise ValueError(f"Model {instance!r} has {len(baseline)} external-only settings; expected at most one")
    ax.axhline(
        visual.score_scale * float(baseline.iloc[0]["estimate"]),
        color=visual.external_baseline_color,
        linestyle=visual.external_baseline_linestyle,
        linewidth=visual.line_width,
        alpha=visual.external_baseline_alpha,
        label=EXTERNAL_BASELINE_LABEL,
        zorder=1,
    )


def _draw_curve(
    ax,
    x: pd.Series,
    rows: pd.DataFrame,
    color: str,
    style: dict[str, str],
    *,
    label: str,
    visual: VisualSettings,
) -> None:
    estimates = visual.score_scale * rows["estimate"].to_numpy(dtype=float)
    lower = visual.score_scale * rows["lower"].to_numpy(dtype=float)
    upper = visual.score_scale * rows["upper"].to_numpy(dtype=float)
    ax.errorbar(
        x.to_numpy(dtype=float),
        estimates,
        yerr=np.vstack((estimates - lower, upper - estimates)) if visual.show_ci else None,
        color=color,
        marker=style["marker"],
        linestyle=style["linestyle"] if len(x) > 1 else "none",
        markersize=visual.marker_size,
        linewidth=visual.line_width,
        elinewidth=visual.ci_line_width,
        capsize=visual.cap_size if visual.show_ci else 0,
        alpha=visual.ci_alpha if visual.show_ci else 1.0,
        label=label,
    )


def _condition_legend(data: FullExternalAugmentation, visual: VisualSettings) -> tuple[list, list[str]]:
    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=_CONDITION_STYLES[condition]["marker"],
            linestyle=_CONDITION_STYLES[condition]["linestyle"],
            markersize=visual.marker_size,
            linewidth=visual.line_width,
            label=label,
        )
        for condition, color, label in (
            (LOCAL_ONLY, visual.local_color, f"Local {dataset_label(data.local_center)} only"),
            (
                COMBINED,
                visual.combined_color,
                f"Complete {dataset_label(data.external_source)} + local",
            ),
        )
    ]
    if visual.show_external_baseline and data.external_only_measured:
        handles.append(
            Line2D(
                [0],
                [0],
                color=visual.external_baseline_color,
                linestyle=visual.external_baseline_linestyle,
                linewidth=visual.line_width,
                label=EXTERNAL_BASELINE_LABEL,
            )
        )
    return handles, [handle.get_label() for handle in handles]


def _sample_axis(
    ax,
    local_counts: tuple[int, ...],
    visual: VisualSettings,
    *,
    measured_zero: bool,
) -> dict[int, float]:
    """Set the local-count axis and return the drawn position of each count."""
    if not local_counts:
        raise ValueError("Augmentation requires at least one positive local training count")
    smallest = min(local_counts)
    positions = {count: float(count) for count in local_counts}
    if measured_zero:
        ax.set_xscale("log", base=2)
        left = smallest / visual.zero_gap_factor
        positions[0] = left * visual.zero_marker_position
        ax.set_xlim(left, max(local_counts) * 1.25)
        ticks = [0, *log_sample_ticks(local_counts, visual.max_x_ticks)]
    else:
        ax.set_xscale("log", base=2)
        ax.set_xlim(smallest / 1.6, max(local_counts) * 1.6)
        ticks = list(log_sample_ticks(local_counts, visual.max_x_ticks))
    ax.set_xticks([positions[value] for value in ticks], [short_count(value) for value in ticks])
    ax.minorticks_off()
    return positions


def _model_grid(count: int, visual: VisualSettings) -> tuple[Figure, list, int]:
    if count < 1:
        raise ValueError("Cannot draw an augmentation figure without models")
    if visual.max_columns < 1:
        raise ValueError("max_columns must be positive")
    column_count = min(visual.max_columns, math.ceil(math.sqrt(count)))
    row_count = math.ceil(count / column_count)
    fig, axes = figure_grid(
        row_count,
        column_count,
        width=visual.figure_width,
        ratio=visual.panel_height_ratio * row_count,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    flat = list(axes.ravel())
    for ax in flat[count:]:
        ax.set_axis_off()
    return fig, flat[:count], column_count


def _bottom_row(axes: list, column_count: int) -> list:
    """Return the axes of the final grid row, which may be partially filled."""
    remainder = len(axes) % column_count
    return axes[len(axes) - (remainder or column_count) :]


def full_external_caption(
    data: FullExternalAugmentation,
    metric: str,
    visual: VisualSettings = VISUAL,
) -> str:
    """Return a self-contained F7 caption."""
    counts = ", ".join(f"{count:,}" for count in data.local_counts)
    boundary = (
        f"the leftmost combined marker is external-only training on the complete "
        f"{dataset_label(data.external_source)} pool of {data.external_count:,} observations at zero local "
        "observations, drawn at the labelled zero position left of the measured counts and connected to the "
        "curve because zero has no position on a logarithmic axis"
        if data.external_only_measured
        else "no external-only run was selected, so the zero-local boundary is unmeasured and omitted"
    )
    baseline = (
        " The grey dashed line repeats that model's externally trained score as the level its local data has to "
        "beat; it is a measured reference, not a fitted threshold, and it can be switched off with "
        "show_external_baseline in VisualSettings."
        if visual.show_external_baseline and data.external_only_measured
        else ""
    )
    ignored = (
        " Runs of the fixed-local design were present in the same experiment and are not drawn here: "
        + ", ".join(data.ignored_runs)
        + "."
        if data.ignored_runs
        else ""
    )
    provenance = (
        f" The local-only curve comes from that center's own single-source sweep ({data.local_only_experiment}), "
        "matched cell by cell on training_sample_seed and realized local count"
        + (
            f"; {len(data.dropped_runs)} reference run(s) repeated a size with seeds the augmentation sweep does "
            "not share and were left out rather than averaged in."
            if data.dropped_runs
            else "."
        )
        if data.local_only_experiment is not None
        else ""
    )
    return (
        rf"\textbf{{Adding local {dataset_label(data.local_center)} observations to the complete "
        f"{dataset_label(data.external_source)} training pool changes {task_label(data.target)} performance on the "
        f"target center.}} {metric_label(metric)} on the held-out {dataset_label(data.local_center)} cohort for "
        f"{len(data.model_instances)} models, each in its own panel. Measured local counts are {counts}; the fixed "
        f"external contribution is the complete {dataset_label(data.external_source)} pool of "
        f"{data.external_count:,} observations. The solid curve trains on local data only, the dashed curve trains "
        f"on the complete external pool plus the same local count, and {boundary}. Combined training deliberately "
        "has a larger total training count than local-only training. Runs of one cell share their "
        "training_sample_seed, so the local-only and combined conditions use the same local observations; "
        f"{_repeat_coverage(data.performance)}.{baseline}{provenance} {_uncertainty(data, visual)}{ignored}"
    )


def fixed_local_caption(
    view: FixedLocalBudgetView,
    metric: str,
    visual: VisualSettings = VISUAL,
) -> str:
    """Return a self-contained F8 caption."""
    counts = ", ".join(f"{count:,}" for count in view.external_counts)
    return (
        rf"\textbf{{Adding {dataset_label(view.external_source)} observations to a fixed local "
        f"{dataset_label(view.local_center)} training budget changes {task_label(view.target)} performance on the "
        f"target center.}} {metric_label(metric)} on the held-out {dataset_label(view.local_center)} cohort for "
        f"{len(view.model_instances)} models, each in its own panel. Every panel fixes {view.local_count:,} local "
        f"observations and increases the added {dataset_label(view.external_source)} count over {counts}; the "
        "dashed horizontal reference is that model's local-only score at the same budget, which is also the "
        "augmented condition at zero added observations. Runs of one budget share their training_sample_seed, so "
        "the local subset stays fixed while the external contribution grows; "
        f"{_repeat_coverage(view.performance)}."
        + (
            f" The local-only reference comes from that center's own single-source sweep ({view.local_only_experiment}), "
            "matched on training_sample_seed and realized local count."
            if view.local_only_experiment is not None
            else ""
        )
        + f" {_uncertainty(view, visual)}"
    )


def _repeat_coverage(performance: pd.DataFrame) -> str:
    counts = sorted(set(performance["run_count"].astype(int)))
    if len(counts) == 1:
        count = counts[0]
        text = "one pipeline run" if count == 1 else f"{count} repeated pipeline runs"
        return f"points summarize {text} per setting"
    return "repeat coverage is " + ", ".join(
        ("one pipeline run" if count == 1 else f"{count} repeated pipeline runs") for count in counts
    )


def _uncertainty(data, visual: VisualSettings) -> str:
    if not visual.show_ci:
        return "No uncertainty intervals are shown."
    return (
        rf"Whiskers are {round(100 * visual.ci_level)}\% percentile intervals from "
        f"{data.bootstrap_count:,} aligned test-cohort bootstrap draws after averaging repeats within each "
        "setting; they quantify held-out cohort resampling uncertainty, not training-repeat variation. "
        "Only measured settings are drawn and connected."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        choices=MAIN_TARGETS,
        help="Prediction target to rebuild; repeat for several. Defaults to all three main targets.",
    )
    parser.add_argument(
        "--evaluation-center",
        action="append",
        dest="evaluation_centers",
        choices=DATA_SOURCES,
        help="Designated target center; repeat for several. Defaults to both centers.",
    )
    parser.add_argument("--run-id", action="append", dest="run_ids")
    parser.add_argument("--output-dir", type=Path, default=DATA.output_dir)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_ids = tuple(args.run_ids) if args.run_ids else DATA.pipeline_runs
    for family in ("augmentation_full_external", "augmentation_fixed_local"):
        tasks = tasks_for(family, args.targets, evaluation_centers=args.evaluation_centers)
        if len(tasks) > 1 and run_ids:
            raise SystemExit("--run-id pins one augmentation input; select exactly one target and evaluation center")
        for task in tasks:
            print(f"\n=== {task.label} [{task.direction}] ({task.experiment_name or 'not registered'})")
            if task.experiment_name is None:
                warn_skipped(task, f"the intended {family} experiment has not been registered yet")
                continue
            try:
                artifacts = load_plot_artifacts(
                    task.experiment_name,
                    pipeline_runs=run_ids,
                    models=DATA.models,
                    exclude_models=DATA.exclude_models,
                    expected_target=task.target,
                )
            except MissingExperimentError as missing:
                warn_skipped(task, missing)
                continue

            visual = replace(VISUAL, metrics=task.metrics, score_scale=task.score_scale)
            local_center = str(task.evaluation_center)
            external_source = str(task.training_source)
            runner = _run_full_external if family == "augmentation_full_external" else _run_fixed_local
            try:
                runner(task, artifacts, local_center, external_source, visual, args.output_dir)
            except IncompleteExperimentError as incomplete:
                warn_skipped(task, incomplete)


def _load_single_source_reference(task, source: str) -> PlotArtifacts | None:
    """Load the declared single-source sweep of one center, when it is registered.

    That experiment also supplies the center's learning curves, so its runs carry
    the training-side sample seeds and realized counts an augmentation reference
    has to match; the preparation keeps only the matched cells.
    """
    declared = tasks_for("sample_size", [task.target], training_sources=[source])
    if len(declared) != 1 or declared[0].experiment_name is None:
        print(f"note: no single-source {dataset_label(source)} sweep is registered for this task", file=sys.stderr)
        return None
    try:
        artifacts = load_plot_artifacts(
            declared[0].experiment_name,
            models=DATA.models,
            exclude_models=DATA.exclude_models,
            expected_target=task.target,
            expected_training_source=source,
        )
    except MissingExperimentError as missing:
        print(f"warning: {declared[0].experiment_name!r} reference unavailable: {missing}", file=sys.stderr)
        return None
    print(f"Reference sweep {declared[0].experiment_name!r}: {len(artifacts.run_ids)} pipeline runs")
    return artifacts


def _common_roster(*items: PlotArtifacts) -> tuple[PlotArtifacts, ...]:
    """Restrict every input to the models all of them evaluated.

    A single-source sweep and an augmentation sweep may have been run with
    different model sets. Comparing them requires one shared roster: the models
    missing anywhere are reported rather than silently compared over unequal
    panels, and an explicitly pinned roster (DATA.models) must cover them all.
    """
    selected = [item for item in items if item is not None]
    rosters = [set(item.metrics["model_instance"].astype(str)) for item in selected]
    common = set.intersection(*rosters)
    if not common:
        raise ValueError("The augmentation inputs share no evaluated model")
    excluded = {item.experiment_name: sorted(roster - common) for item, roster in zip(selected, rosters, strict=True)}
    if DATA.models is not None and any(excluded.values()):
        raise ValueError(
            "The pinned DATA.models roster is not covered by every input; missing: "
            + str({name: models for name, models in excluded.items() if models})
        )
    for name, models in excluded.items():
        if models:
            print(f"  note: {models} are not evaluated in {name!r} and are left out of this figure")
    return tuple(
        PlotArtifacts(
            metrics=item.metrics.loc[item.metrics["model_instance"].astype(str).isin(common)].copy(),
            bootstrap_scores=item.bootstrap_scores.loc[
                item.bootstrap_scores["model_instance"].astype(str).isin(common)
            ].copy(),
            experiment_name=item.experiment_name,
            run_ids=item.run_ids,
        )
        for item in selected
    )


def _run_full_external(task, artifacts, local_center: str, external_source: str, visual, output_dir: Path) -> None:
    local_reference = _load_single_source_reference(task, local_center)
    external_reference = _load_single_source_reference(task, external_source)
    artifacts, local_reference, external_reference = _common_roster(artifacts, local_reference, external_reference)
    prepared = prepare_full_external_augmentation(
        artifacts,
        local_center=local_center,
        external_source=external_source,
        metrics=visual.metrics,
        ci_level=visual.ci_level,
        local_only_artifacts=local_reference,
        external_only_artifacts=external_reference,
    )
    print("Selected augmentation pipeline runs: " + ", ".join(artifacts.run_ids))
    if prepared.local_only_experiment is not None:
        print(f"Local-only reference experiment: {prepared.local_only_experiment!r}")
    if prepared.dropped_runs:
        print("Unmatched reference runs not drawn (different repeat seeds): " + ", ".join(prepared.dropped_runs))
    output_root = output_dir / prepared.target
    for metric in prepared.metrics:
        stem = output_root / f"{external_source}_to_{local_center}_full_external_{metric}"
        outputs = save(make_full_external_figure(prepared, metric, visual), str(stem), formats=visual.output_formats)
        print(f"LaTeX caption ({metric_label(metric)}):")
        print(f"\\caption{{{full_external_caption(prepared, metric, visual)}}}")
        for output in outputs:
            print("figure: " + str(Path(output).relative_to(config.dir_root)))


def _run_fixed_local(task, artifacts, local_center: str, external_source: str, visual, output_dir: Path) -> None:
    local_reference = _load_single_source_reference(task, local_center)
    artifacts, local_reference = _common_roster(artifacts, local_reference)
    prepared: FixedLocalAugmentation = prepare_fixed_local_augmentation(
        artifacts,
        local_center=local_center,
        external_source=external_source,
        metrics=visual.metrics,
        ci_level=visual.ci_level,
        local_only_artifacts=local_reference,
    )
    print("Selected augmentation pipeline runs: " + ", ".join(artifacts.run_ids))
    if prepared.local_only_experiment is not None:
        print(f"Local-only reference experiment: {prepared.local_only_experiment!r}")
    if prepared.dropped_runs:
        print("Unmatched reference runs not drawn (different repeat seeds): " + ", ".join(prepared.dropped_runs))
    output_root = output_dir / prepared.target
    for view in prepared.budget_views:
        for metric in prepared.metrics:
            stem = output_root / f"{external_source}_to_{local_center}_fixed_local-{view.local_count}_{metric}"
            outputs = save(
                make_fixed_local_figure(view, metric, visual),
                str(stem),
                formats=visual.output_formats,
            )
            print(f"LaTeX caption ({metric_label(metric)}, local={view.local_count:,}):")
            print(f"\\caption{{{fixed_local_caption(view, metric, visual)}}}")
            for output in outputs:
                print("figure: " + str(Path(output).relative_to(config.dir_root)))


if __name__ == "__main__":
    main()
