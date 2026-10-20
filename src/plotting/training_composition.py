"""F6: two-center composition curves per budget and across budgets by model.

Regenerate with:

    uv run python -m src.plotting.training_composition

Each centrally declared target is skipped, without substitution, until its
composition MLflow experiment is registered and contains measured runs.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.config import config
from src.plotting.defaults import (
    DATASET_COLORS,
    DATASET_NAMES,
    MetricPanelSettings,
    dataset_label,
    metric_label,
    set_plot_style,
    task_label,
)
from src.plotting.experiments import MAIN_TARGETS, tasks_for, warn_skipped
from src.plotting.model_sets import PLOT_METRICS, plot_dir
from src.plotting.scientific_figstyle import WIDE, figure_grid, panel_labels, save
from src.plotting.utils import (
    CompositionBudgetView,
    CompositionEvaluation,
    MissingExperimentError,
    load_plot_artifacts,
    prepare_composition_evaluation,
)
from src.plotting.utils.rendering import instance_plot_styles, interval_axis_limits


@dataclass(frozen=True)
class DataSettings:
    """Run/model selection and F6 output location."""

    pipeline_runs: tuple[str, ...] | None = None
    models: tuple[str, ...] | None = None
    exclude_models: tuple[str, ...] | None = None
    output_dir: Path = plot_dir("composition")


@dataclass(frozen=True)
class VisualSettings(MetricPanelSettings):
    """Locally editable presentation choices for F6."""

    metrics: tuple[str, ...] = PLOT_METRICS
    show_ci: bool = True
    ci_level: float = 0.95
    figure_width: float = WIDE
    panel_height_ratio: float = 0.34
    budget_grid_row_height_ratio: float = 0.22
    max_columns: int = 3
    max_share_ticks: int = 5
    marker_size: float = 3.8
    line_width: float = 1.1
    ci_line_width: float = 0.7
    cap_size: float = 1.8
    ci_alpha: float = 0.7
    axis_padding_fraction: float = 0.08
    output_formats: tuple[str, ...] = ("pdf",)


DATA = DataSettings()
VISUAL = VisualSettings()

_EVALUATION_STYLES = {
    "mimic": {"color": DATASET_COLORS["mimic"], "marker": "o", "linestyle": "-"},
    "tudd": {"color": DATASET_COLORS["tudd"], "marker": "s", "linestyle": "--"},
}


def make_figure(
    data: CompositionBudgetView,
    metric: str,
    visual: VisualSettings = VISUAL,
) -> Figure:
    """Draw one model per panel with MIMIC-IV and EUH evaluation curves."""
    if metric not in data.metrics:
        raise ValueError(f"Metric {metric!r} is unavailable; prepared metrics: {list(data.metrics)}")
    if visual.max_columns < 1:
        raise ValueError("max_columns must be positive")
    if not data.model_instances:
        raise ValueError("Cannot draw a composition figure without models")

    set_plot_style()
    column_count = min(visual.max_columns, math.ceil(math.sqrt(len(data.model_instances))))
    row_count = math.ceil(len(data.model_instances) / column_count)
    fig, axes = figure_grid(
        row_count,
        column_count,
        width=visual.figure_width,
        row_height=visual.panel_height_ratio,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    flat_axes = axes.ravel().tolist()
    active_axes = flat_axes[: len(data.model_instances)]
    for ax in flat_axes[len(data.model_instances) :]:
        ax.set_axis_off()

    styles = instance_plot_styles(data.model_metadata)
    metric_rows = data.performance.loc[data.performance["metric"].eq(metric)].copy()
    y_limits = interval_axis_limits(
        metric_rows,
        scale=visual.score_scale,
        show_ci=visual.show_ci,
        padding_fraction=visual.axis_padding_fraction,
    )
    ticks, tick_labels = _share_ticks(data, visual.max_share_ticks)

    for ax, instance in zip(active_axes, data.model_instances, strict=True):
        instance_rows = metric_rows.loc[metric_rows["model_instance"].astype(str).eq(instance)]
        _draw_composition_curves(ax, data, instance_rows, instance, visual)
        ax.set_xlim(-5, 105)
        ax.set_ylim(y_limits)
        ax.set_xticks(ticks, tick_labels)
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

    row_starts = range(0, len(active_axes), column_count)
    for i in row_starts:
        active_axes[i].set_ylabel(f"{metric_label(metric)}")

    bottom_row_start = (row_count - 1) * column_count
    for ax in active_axes[bottom_row_start:]:
        ax.set_xlabel(f"{DATASET_NAMES['mimic']} share of training observations")
    handles, labels = active_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2)
    panel_labels(active_axes)
    return fig


def make_budget_grid(
    data: CompositionEvaluation,
    metric: str,
    visual: VisualSettings = VISUAL,
) -> Figure:
    """Draw one model per row and one measured total training size per column."""
    if not data.budget_views:
        raise ValueError("Cannot draw a composition grid without measured budgets")
    views = sorted(data.budget_views, key=lambda view: view.total_count)
    instances = views[0].model_instances
    if not instances:
        raise ValueError("Cannot draw a composition grid without models")
    for view in views:
        if metric not in view.metrics:
            raise ValueError(f"Metric {metric!r} is unavailable at N={view.total_count}")
        if set(view.model_instances) != set(instances):
            raise ValueError("Composition budgets must share one model roster")

    set_plot_style()
    fig, axes = figure_grid(
        len(instances),
        len(views),
        width=visual.figure_width,
        row_height=visual.budget_grid_row_height_ratio,
        sharex="col",
        sharey=True,
        squeeze=False,
    )
    rows_by_budget = [view.performance.loc[lambda rows: rows["metric"].eq(metric)] for view in views]
    y_limits = interval_axis_limits(
        pd.concat(rows_by_budget, ignore_index=True),
        scale=visual.score_scale,
        show_ci=visual.show_ci,
        padding_fraction=visual.axis_padding_fraction,
    )
    styles = instance_plot_styles(views[0].model_metadata)
    for column, (view, metric_rows) in enumerate(zip(views, rows_by_budget, strict=True)):
        ticks, tick_labels = _share_ticks(view, visual.max_share_ticks)
        axes[0, column].annotate(
            f"N = {view.total_count:,}",
            xy=(0.5, 1),
            xycoords="axes fraction",
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
        for row, instance in enumerate(instances):
            ax = axes[row, column]
            instance_rows = metric_rows.loc[metric_rows["model_instance"].astype(str).eq(instance)]
            _draw_composition_curves(ax, view, instance_rows, instance, visual)
            ax.set_xlim(-5, 105)
            ax.set_ylim(y_limits)
            ax.set_xticks(ticks, tick_labels)
            ax.grid(axis="both")
            if column == 0:
                ax.set_ylabel(f"{styles[instance][1]}\n{metric_label(metric)}")

    for ax in axes[-1, :]:
        ax.set_xlabel(f"{DATASET_NAMES['mimic']} training share")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2)
    panel_labels(
        axes,
        labels=[
            f"({row + 1}{chr(ord('a') + column)})"
            for row in range(len(instances))
            for column in range(len(views))
        ],
    )
    return fig


def _draw_composition_curves(
    ax: Axes,
    data: CompositionBudgetView,
    instance_rows: pd.DataFrame,
    instance: str,
    visual: VisualSettings,
) -> None:
    for center in data.evaluation_centers:
        rows = instance_rows.loc[instance_rows["dataset"].eq(center)].sort_values("mimic_share")
        if len(rows) != len(data.compositions):
            raise ValueError(
                f"Model {instance!r} has {len(rows)} {center!r} composition cells, "
                f"expected {len(data.compositions)}; refusing to connect incomparable coverage"
            )
        estimates = visual.score_scale * rows["estimate"].to_numpy(dtype=float)
        lower = visual.score_scale * rows["lower"].to_numpy(dtype=float)
        upper = visual.score_scale * rows["upper"].to_numpy(dtype=float)
        style = _EVALUATION_STYLES[center]
        ax.errorbar(
            rows["mimic_share"],
            estimates,
            yerr=np.vstack((estimates - lower, upper - estimates)) if visual.show_ci else None,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markersize=visual.marker_size,
            linewidth=visual.line_width,
            elinewidth=visual.ci_line_width,
            capsize=visual.cap_size if visual.show_ci else 0,
            alpha=visual.ci_alpha if visual.show_ci else 1.0,
            label=f"Evaluation on {dataset_label(center)}",
        )


def budget_grid_caption(
    data: CompositionEvaluation,
    metric: str,
    visual: VisualSettings = VISUAL,
) -> str:
    """Describe the cross-budget layout without pooling different training sizes."""
    coverage = " ".join(
        f"At N={view.total_count:,}, measured MIMIC-IV shares are "
        + ", ".join(_format_share(float(share)) for share in view.compositions["mimic_share"])
        + f"; points summarize {_repeat_coverage(view)}."
        for view in sorted(data.budget_views, key=lambda view: view.total_count)
    )
    uncertainty = (
        "Points are mean run estimates; whiskers span the minimum and maximum run estimates within each exact "
        "composition, not confidence intervals. A single run has zero-width whiskers."
        if visual.show_ci
        else "Points are mean run estimates; no uncertainty intervals are shown."
    )
    return (
        rf"\textbf{{Source-composition effects on {metric_label(metric)} across training sizes "
        f"for {task_label(data.target)}.}} Rows fix the model; columns increase the total number N of training "
        "observations. Each panel replaces EUH observations with MIMIC-IV observations at fixed N. "
        "Solid curves evaluate on held-out MIMIC-IV; dashed curves evaluate on held-out EUH. "
        f"All panels share the same performance scale. {coverage} {uncertainty} "
        "Different training sizes are never averaged together."
    )


def caption(
    data: CompositionBudgetView,
    metric: str,
    visual: VisualSettings = VISUAL,
) -> str:
    """Return a self-contained caption for one target/metric/budget figure."""
    measured = "; ".join(
        f"{_format_share(float(row.mimic_share))} MIMIC "
        f"({int(row.mimic_count):,} MIMIC-IV + {int(row.tudd_count):,} EUH)"
        for row in data.compositions.itertuples()
    )
    uncertainty = (
        "Points are mean run estimates; whiskers span the minimum and maximum run estimates within each exact "
        "composition, not confidence intervals. A single run has zero-width whiskers."
        if visual.show_ci
        else "No uncertainty intervals are shown."
    )
    return (
        rf"\textbf{{Fixed-budget source replacement changes {metric_label(metric)} differently across models and "
        f"evaluation centers for {task_label(data.target)}.}} Each panel fixes one model at N={data.total_count:,} "
        "training observations and shows exactly two absolute-performance curves: the solid MIMIC-IV curve and "
        "the dashed EUH curve identify the ordinary held-out evaluation centers. Measured training compositions "
        f"are {measured}. Points summarize {_repeat_coverage(data)}. {uncertainty} Both pure-source endpoints were measured. "
    )


def _share_ticks(data: CompositionBudgetView, maximum: int) -> tuple[list[float], list[str]]:
    if maximum < 2:
        raise ValueError("max_share_ticks must be at least two")
    measured = sorted({float(value) for value in data.compositions["mimic_share"] if 0.0 < float(value) < 100.0})
    available = maximum - 2
    if len(measured) > available:
        indices = np.linspace(0, len(measured) - 1, available).round().astype(int)
        measured = [measured[index] for index in np.unique(indices)]
    ticks = [0.0, *measured, 100.0]
    labels = [
        "0%",
        *[_format_share(value) for value in measured],
        "100%",
    ]
    return ticks, labels


def _format_share(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".") + "%"


def _repeat_coverage(data: CompositionBudgetView) -> str:
    counts_to_compositions: dict[int, list[str]] = {}
    for row in data.compositions.itertuples():
        identity = f"{_format_share(float(row.mimic_share))} ({int(row.mimic_count):,}+{int(row.tudd_count):,})"
        counts_to_compositions.setdefault(int(row.run_count), []).append(identity)
    if len(counts_to_compositions) == 1:
        repeat_count = next(iter(counts_to_compositions))
        return f"{_run_count_text(repeat_count)} at each of {len(data.compositions)} measured compositions"
    coverage = []
    for repeat_count, compositions in sorted(counts_to_compositions.items()):
        identities = ", ".join(compositions)
        coverage.append(f"{_run_count_text(repeat_count)} at {identities}")
    return "repeat coverage of " + "; ".join(coverage)


def _run_count_text(count: int) -> str:
    return "one pipeline run" if count == 1 else f"{count} repeated pipeline runs"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        choices=MAIN_TARGETS,
        help="Prediction target to rebuild; repeat for several. Defaults to all three main targets.",
    )
    parser.add_argument("--run-id", action="append", dest="run_ids")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tasks = tasks_for("composition", args.targets)
    run_ids = tuple(args.run_ids) if args.run_ids else DATA.pipeline_runs
    if len(tasks) > 1 and run_ids:
        raise SystemExit("--run-id pins one composition input; select exactly one --target")

    for task in tasks:
        experiment_name = task.experiment_name
        print(f"\n=== {task.label} [{task.direction}] ({experiment_name or 'not registered'})")
        if experiment_name is None:
            warn_skipped(task, "the intended fixed-budget composition experiment has not been registered yet")
            continue
        try:
            artifacts = load_plot_artifacts(
                experiment_name,
                pipeline_runs=run_ids,
                models=DATA.models,
                exclude_models=DATA.exclude_models,
                expected_target=task.target,
            )
        except MissingExperimentError as missing:
            warn_skipped(task, missing)
            continue

        visual = VISUAL
        prepared = prepare_composition_evaluation(
            artifacts,
            metrics=visual.metrics,
            ci_level=visual.ci_level,
        )
        print("Selected composition pipeline runs: " + ", ".join(artifacts.run_ids))
        for view in prepared.budget_views:
            output_dir = DATA.output_dir / view.target
            for metric in view.metrics:
                stem = output_dir / f"composition_budget-{view.total_count}"
                outputs = save(
                    make_figure(view, metric, visual),
                    str(stem),
                    formats=visual.output_formats,
                )
                print(f"LaTeX caption ({metric_label(metric)}, N={view.total_count:,}):")
                print(f"\\caption{{{caption(view, metric, visual)}}}")
                for output in outputs:
                    print("figure: " + str(Path(output).relative_to(config.dir_root)))
        for metric in prepared.metrics:
            stem = DATA.output_dir / prepared.target / "composition_by_model"
            outputs = save(
                make_budget_grid(prepared, metric, visual),
                str(stem),
                formats=tuple(dict.fromkeys((*visual.output_formats, "svg"))),
            )
            print(f"LaTeX caption ({metric_label(metric)}, model rows / size columns):")
            print(f"\\caption{{{budget_grid_caption(prepared, metric, visual)}}}")
            for output in outputs:
                print("figure: " + str(Path(output).relative_to(config.dir_root)))


if __name__ == "__main__":
    main()
