"""F7: training-data augmentation with external observations.

Regenerate with:

    uv run python -m src.plotting.augmentation

This figure fixes the complete external training pool and varies the number of
local target-center observations. It is declared per prediction task and target
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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from mlflow import MlflowClient
from src.config import config
from src.mlflow.evaluation_data import DEFAULT_TRACKING_URI
from src.plotting.defaults import MetricPanelSettings, dataset_label, metric_label, set_plot_style, task_label
from src.plotting.experiments import DATA_SOURCES, MAIN_TARGETS, tasks_for, warn_skipped
from src.plotting.scientific_figstyle import BASELINE, PALETTE, WIDE, figure_grid, panel_labels, save
from src.plotting.utils import (
    FullExternalAugmentation,
    IncompleteExperimentError,
    MissingExperimentError,
    PlotArtifacts,
    load_plot_artifacts,
    prepare_full_external_augmentation,
)
from src.plotting.utils.augmentation import COMBINED, EXTERNAL_ONLY, LOCAL_ONLY, reuse_full_pool_runs
from src.plotting.utils.rendering import (
    instance_plot_styles,
    interval_axis_limits,
    log_sample_ticks,
    short_count,
)
from src.plotting.utils.runs import read_training_sample_seed, select_artifact_runs


@dataclass(frozen=True)
class DataSettings:
    """Run/model selection and F7 output location."""

    pipeline_runs: tuple[str, ...] | None = None
    models: tuple[str, ...] | None = None
    exclude_models: tuple[str, ...] | None = None
    output_dir: Path = config.dir_plots / "augmentation"


@dataclass(frozen=True)
class VisualSettings(MetricPanelSettings):
    """Locally editable presentation choices for F7."""

    metrics: tuple[str, ...] = ("roc_auc", "prc_auc")
    show_ci: bool = True
    ci_level: float = 0.95
    figure_width: float = WIDE
    panel_height_ratio: float = 0.34
    max_columns: int = 3
    marker_size: float = 3.8
    line_width: float = 1.1
    ci_line_width: float = 0.7
    cap_size: float = 1.8
    ci_alpha: float = 0.7
    max_x_ticks: int = 6
    show_external_baseline: bool = True
    external_baseline_color: str = BASELINE
    external_baseline_linestyle: str = "--"
    external_baseline_alpha: float = 1.0
    axis_padding_fraction: float = 0.08
    local_color: str = PALETTE["blue"]
    combined_color: str = PALETTE["orange"]
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
    positions = _sample_axis(axes[0], data.local_counts, visual)

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
    fig.legend(*legend, loc="outside upper center", ncol=3)
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
    curve = cell.loc[cell["local_count"].gt(0)].sort_values("local_count").copy()
    curve["position"] = curve["local_count"].astype(int).map(positions).astype(float)
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


def _sample_axis(ax, local_counts: tuple[int, ...], visual: VisualSettings) -> dict[int, float]:
    """Set the local-count axis and return the drawn position of each count."""
    if not local_counts:
        raise ValueError("Augmentation requires at least one positive local training count")
    positions = {count: float(count) for count in local_counts}
    ax.set_xscale("log", base=2)
    ax.set_xlim(min(local_counts) / 1.6, max(local_counts) * 1.6)
    ticks = log_sample_ticks(local_counts, visual.max_x_ticks)
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
        row_height=visual.panel_height_ratio,
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
    baseline = (
        " The grey dashed line repeats that model's externally trained score as the level its local data has to "
        "beat; it is a measured reference, not a fitted threshold, and it can be switched off with "
        "show_external_baseline in VisualSettings."
        if visual.show_external_baseline and data.external_only_measured
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
        f"{data.external_count:,} observations. The solid curve trains on local data only, and the dashed curve "
        "trains on the complete external pool plus the same local count. Combined training deliberately has a "
        "larger total training count than local-only training. Runs of one cell share their "
        "training_sample_seed, so the local-only and combined conditions use the same local observations; "
        f"{_repeat_coverage(data.performance)}.{baseline}{provenance} {_uncertainty(data, visual)}"
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
    parser.add_argument("--seed", type=int, help="Keep augmentation runs with this training_sample_seed.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_ids = tuple(args.run_ids) if args.run_ids else DATA.pipeline_runs
    tasks = tasks_for("augmentation_full_external", args.targets, evaluation_centers=args.evaluation_centers)
    if len(tasks) > 1 and run_ids:
        raise SystemExit("--run-id pins one augmentation input; select exactly one target and evaluation center")
    for task in tasks:
        print(f"\n=== {task.label} [{task.direction}] ({task.experiment_name or 'not registered'})")
        if task.experiment_name is None:
            warn_skipped(task, "the intended augmentation experiment has not been registered yet")
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

        if args.seed is not None:
            client = MlflowClient(tracking_uri=DEFAULT_TRACKING_URI)
            selected_ids = [
                run_id
                for run_id in artifacts.run_ids
                if read_training_sample_seed(client, run_id) == args.seed
            ]
            if not selected_ids:
                warn_skipped(task, f"no augmentation runs record training_sample_seed={args.seed}")
                continue
            artifacts = select_artifact_runs(artifacts, selected_ids)

        reciprocal_tasks = tasks_for(
            "augmentation_full_external", [task.target], evaluation_centers=[str(task.training_source)]
        )
        reciprocal_name = reciprocal_tasks[0].experiment_name
        if reciprocal_name is not None:
            try:
                reciprocal = load_plot_artifacts(
                    reciprocal_name,
                    models=DATA.models,
                    exclude_models=DATA.exclude_models,
                    expected_target=task.target,
                )
            except MissingExperimentError as missing:
                print(f"note: reciprocal full-data endpoint unavailable: {missing}", file=sys.stderr)
            else:
                artifacts = reuse_full_pool_runs(artifacts, reciprocal)

        try:
            _run_full_external(
                task,
                artifacts,
                str(task.evaluation_center),
                str(task.training_source),
                VISUAL,
                DATA.output_dir,
            )
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


if __name__ == "__main__":
    main()
