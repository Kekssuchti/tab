"""F10: retrieval performance over the selected budget with its unrestricted reference.

Regenerate with:

    uv run python -m src.plotting.retrieval_budget

Every measured target batch becomes its own figure, and every panel fixes one
model: the curves are the declared retrieval strategies over the number of
selected training observations, and the dashed reference is the same model
trained on the entire eligible candidate pool and evaluated on exactly that same
batch. F9 supplies the paired strategy effect against random; this figure supplies
absolute performance and the unrestricted comparison.

The unrestricted reference comes from its own registered experiment
(`retrieval_unrestricted` in `src.plotting.experiments`). Ordinary full-held-out
predictions are not equivalent to a matched-batch evaluation, so they are never
substituted when the reference input is missing.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.config import config
from src.plotting.defaults import dataset_label, metric_label, set_plot_style, task_label
from src.plotting.experiments import (
    DATA_SOURCES,
    MAIN_TARGETS,
    PlotExperiment,
    retrieval_unrestricted_input,
    tasks_for,
    warn_skipped,
)
from src.plotting.scientific_figstyle import BASELINE, WIDE, figure_grid, panel_labels, save
from src.plotting.utils import (
    IncompleteExperimentError,
    MissingExperimentError,
    PlotArtifacts,
    RetrievalBatchView,
    load_plot_artifacts,
    prepare_retrieval_budget_evaluation,
)
from src.plotting.utils.rendering import (
    instance_plot_styles,
    interval_axis_limits,
    log_sample_ticks,
    short_count,
)
from src.plotting.utils.retrieval import RANDOM_STRATEGY


@dataclass(frozen=True)
class DataSettings:
    """Run/model selection and F10 output location."""

    pipeline_runs: tuple[str, ...] | None = None
    unrestricted_runs: tuple[str, ...] | None = None
    models: tuple[str, ...] | None = None
    exclude_models: tuple[str, ...] | None = None
    output_dir: Path = config.dir_plots / "retrieval"


@dataclass(frozen=True)
class VisualSettings:
    """Locally editable presentation choices for F10."""

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
    axis_padding_fraction: float = 0.08
    reference_color: str = BASELINE
    reference_band_alpha: float = 0.12
    strategy_colors: tuple[str, ...] = ("#3B6FB6", "#D1701C", "#3F7D3F", "#7A4FA3", "#0F8B8D", "#8C564B")
    strategy_markers: tuple[str, ...] = ("o", "s", "^", "D", "v", "P")
    output_formats: tuple[str, ...] = ("pdf",)


DATA = DataSettings()
VISUAL = VisualSettings()


def make_figure(
    view: RetrievalBatchView,
    metric: str,
    visual: VisualSettings = VISUAL,
) -> Figure:
    """Draw strategy curves over the selected budget for one target batch."""
    set_plot_style()
    if metric not in view.metrics:
        raise ValueError(f"Metric {metric!r} is unavailable; prepared metrics: {list(view.metrics)}")
    rows = view.performance.loc[view.performance["metric"].eq(metric)].copy()
    if rows.empty:
        raise ValueError(f"No {metric!r} rows are available")
    styles = instance_plot_styles(view.model_metadata)
    strategy_styles = _strategy_styles(view.strategies, view.strategy_labels(), visual)

    limit_rows = rows
    if view.unrestricted is not None:
        reference_rows = view.unrestricted.loc[view.unrestricted["metric"].eq(metric)]
        limit_rows = pd.concat(
            (
                rows[["estimate", "lower", "upper"]],
                reference_rows[["estimate", "lower", "upper"]],
            ),
            ignore_index=True,
        )
    limits = interval_axis_limits(
        limit_rows,
        scale=visual.score_scale,
        show_ci=visual.show_ci,
        padding_fraction=visual.axis_padding_fraction,
    )

    fig, axes, column_count = _model_grid(len(view.model_instances), visual)
    counts = view.selected_counts
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlim(counts[0] / 1.6, counts[-1] * 1.6)
    ticks = log_sample_ticks(counts, visual.max_x_ticks)
    axes[0].set_xticks(ticks, [short_count(value) for value in ticks])
    axes[0].minorticks_off()

    for ax, instance in zip(axes, view.model_instances, strict=True):
        instance_rows = rows.loc[rows["model_instance"].astype(str).eq(instance)]
        for strategy in view.strategies:
            cell = instance_rows.loc[instance_rows["strategy"].eq(strategy)].sort_values("selected_count")
            if cell.empty:
                raise ValueError(f"Model {instance!r} has no {strategy!r} budget cells")
            color, marker, linestyle, label = strategy_styles[strategy]
            estimates = visual.score_scale * cell["estimate"].to_numpy(dtype=float)
            lower = visual.score_scale * cell["lower"].to_numpy(dtype=float)
            upper = visual.score_scale * cell["upper"].to_numpy(dtype=float)
            ax.errorbar(
                cell["selected_count"].to_numpy(dtype=float),
                estimates,
                yerr=np.vstack((estimates - lower, upper - estimates)) if visual.show_ci else None,
                color=color,
                marker=marker,
                linestyle=linestyle,
                markersize=visual.marker_size,
                linewidth=visual.line_width,
                elinewidth=visual.ci_line_width,
                capsize=visual.cap_size if visual.show_ci else 0,
                alpha=visual.ci_alpha if visual.show_ci else 1.0,
                label=label,
                zorder=2,
            )
        if view.unrestricted is not None:
            reference = view.unrestricted.loc[
                view.unrestricted["metric"].eq(metric) & view.unrestricted["model_instance"].astype(str).eq(instance)
            ]
            if len(reference) != 1:
                raise ValueError(f"The unrestricted reference needs one {metric!r} row for model {instance!r}")
            value = visual.score_scale * float(reference.iloc[0]["estimate"])
            if visual.show_ci:
                ax.axhspan(
                    visual.score_scale * float(reference.iloc[0]["lower"]),
                    visual.score_scale * float(reference.iloc[0]["upper"]),
                    color=visual.reference_color,
                    alpha=visual.reference_band_alpha,
                    zorder=0,
                )
            ax.axhline(
                value,
                color=visual.reference_color,
                linewidth=visual.line_width,
                linestyle="--",
                label=f"Unrestricted pool ({view.unrestricted_count:,})",
                zorder=1,
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

    axes[0].set_ylabel(f"Absolute {metric_label(metric)} on {dataset_label(view.batch_center)} target batch (%)")
    for ax in _bottom_row(axes, column_count):
        ax.set_xlabel("Selected training observations")
    handles, labels = axes[0].get_legend_handles_labels()
    if view.unrestricted is None:
        handles.append(
            Line2D(
                [0],
                [0],
                color=visual.reference_color,
                linestyle=":",
                linewidth=visual.line_width,
                label="Unrestricted reference unavailable",
            )
        )
        labels.append("Unrestricted reference unavailable")
    fig.legend(handles, labels, loc="outside upper center", ncol=min(len(handles), 3))
    panel_labels(axes)
    return fig


def _strategy_styles(
    strategies: tuple[str, ...],
    labels: dict[str, str],
    visual: VisualSettings,
) -> dict[str, tuple[str, str, str, str]]:
    """Return one stable style per retrieval strategy, random first."""
    styles = {}
    for index, strategy in enumerate(strategies):
        is_reference = strategy == RANDOM_STRATEGY
        color = BASELINE if is_reference else visual.strategy_colors[index % len(visual.strategy_colors)]
        marker = visual.strategy_markers[index % len(visual.strategy_markers)]
        linestyle = "-" if is_reference else "--"
        styles[strategy] = (color, marker, linestyle, labels[strategy])
    return styles


def _model_grid(count: int, visual: VisualSettings) -> tuple[Figure, list, int]:
    if count < 1:
        raise ValueError("Cannot draw a retrieval budget figure without models")
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


def caption(view: RetrievalBatchView, metric: str, visual: VisualSettings = VISUAL) -> str:
    """Return a self-contained F10 caption."""
    counts = ", ".join(f"{count:,}" for count in view.selected_counts)
    strategies = ", ".join(
        sorted(set(view.conditions["strategy_label"].astype(str)), key=lambda label: label)
    )
    repeats = sorted(set(view.conditions["run_count"].astype(int)))
    repeat_text = (
        "one pipeline run per setting"
        if repeats == [1]
        else "per setting " + ", ".join(f"{count} repeated pipeline runs" for count in repeats)
    )
    reference = (
        f"The dashed horizontal reference trains the same model on the unrestricted candidate pool of "
        f"{view.unrestricted_count:,} observations and evaluates it on this same batch"
        + (
            ", with its bootstrap interval shaded behind the line. "
            if visual.show_ci
            else ", without an interval band. "
        )
        + "It is not the best ordinary full-data model, and a selected-subset curve may legitimately lie above it. "
        if view.unrestricted is not None
        else "The unrestricted candidate-pool reference is not registered for this direction, so no reference line "
        "is shown; ordinary full-held-out results are not substituted because their evaluation population differs. "
    )
    uncertainty = (
        rf"Whiskers are {round(100 * view.ci_level)}\% percentile intervals from {view.bootstrap_count:,} "
        "bootstrap draws of this target batch after averaging training repeats within each setting; they describe "
        "resampling of these fixed batch observations and not variation across target batches or training repeats."
        if visual.show_ci
        else "No uncertainty intervals are shown."
    )
    return (
        rf"\textbf{{Retrieval performance over the selected training budget relative to unrestricted training on "
        f"the target batch for {task_label(view.target)}.}} {metric_label(metric)} of "
        f"{len(view.model_instances)} models on a {view.batch_size:,}-observation "
        f"{dataset_label(view.batch_center)} target batch (batch sample seed {view.test_sample_seed}), selecting "
        f"from the complete {dataset_label(view.candidate_source)} candidate pool. Each panel fixes one model; "
        f"curves are the declared strategies ({strategies}) over measured budgets {counts}, and every curve keeps "
        "its strategy configuration fixed. Measured repeat coverage: " + repeat_text + ". " + reference + uncertainty
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
        "--training-source",
        action="append",
        dest="training_sources",
        choices=DATA_SOURCES,
        help="Candidate-pool source; repeat for several. Defaults to both sources.",
    )
    parser.add_argument(
        "--evaluation-center",
        action="append",
        dest="evaluation_centers",
        choices=DATA_SOURCES,
        help="Target-batch center; repeat for several. Defaults to both centers.",
    )
    parser.add_argument("--run-id", action="append", dest="run_ids")
    parser.add_argument("--unrestricted-run-id", action="append", dest="unrestricted_run_ids")
    parser.add_argument("--output-dir", type=Path, default=DATA.output_dir)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tasks = tasks_for(
        "retrieval_budget",
        args.targets,
        training_sources=args.training_sources,
        evaluation_centers=args.evaluation_centers,
    )
    run_ids = tuple(args.run_ids) if args.run_ids else DATA.pipeline_runs
    unrestricted_ids = tuple(args.unrestricted_run_ids) if args.unrestricted_run_ids else DATA.unrestricted_runs
    if len(tasks) > 1 and (run_ids or unrestricted_ids):
        raise SystemExit(
            "--run-id/--unrestricted-run-id pins one retrieval input; select exactly one target, "
            "training source, and evaluation center"
        )

    for task in tasks:
        experiment_name = task.experiment_name
        print(f"\n=== {task.label} [{task.direction}] ({experiment_name or 'not registered'})")
        if experiment_name is None:
            warn_skipped(task, "the intended experiment has not been registered yet")
            continue
        try:
            budget_artifacts = load_plot_artifacts(
                experiment_name,
                pipeline_runs=run_ids,
                models=DATA.models,
                exclude_models=DATA.exclude_models,
                expected_target=task.target,
            )
        except MissingExperimentError as missing:
            warn_skipped(task, missing)
            continue

        visual = replace(VISUAL, metrics=task.metrics, score_scale=task.score_scale)
        unrestricted_artifacts = _load_unrestricted(task, unrestricted_ids, DATA)
        try:
            prepared = prepare_retrieval_budget_evaluation(
                budget_artifacts,
                unrestricted_artifacts=unrestricted_artifacts,
                metrics=visual.metrics,
                ci_level=visual.ci_level,
            )
        except IncompleteExperimentError as incomplete:
            warn_skipped(task, incomplete)
            continue
        print("Selected budget pipeline runs: " + ", ".join(budget_artifacts.run_ids))
        if prepared.unrestricted_run_ids:
            print("Selected unrestricted pipeline runs: " + ", ".join(prepared.unrestricted_run_ids))

        output_root = args.output_dir / prepared.target
        for view in prepared.batch_views:
            for metric in prepared.metrics:
                stem = output_root / (
                    f"{task.direction_slug}_batch-{view.batch_center}-{view.batch_size}"
                    f"_seed-{view.test_sample_seed}_{metric}"
                )
                outputs = save(make_figure(view, metric, visual), str(stem), formats=visual.output_formats)
                print(
                    f"LaTeX caption ({metric_label(metric)}, batch={view.batch_center}/{view.batch_size}/"
                    f"{view.test_sample_seed}):"
                )
                print(f"\\caption{{{caption(view, metric, visual)}}}")
                for output in outputs:
                    print("figure: " + str(Path(output).relative_to(config.dir_root)))


def _load_unrestricted(
    task: PlotExperiment,
    run_ids: tuple[str, ...],
    data: DataSettings,
) -> PlotArtifacts | None:
    """Load the registered matched-batch unrestricted reference, or report its absence."""
    registered = retrieval_unrestricted_input(task)
    if registered.experiment_name is None:
        print(
            f"warning: no unrestricted candidate-pool reference is registered for {task.label} "
            f"[{task.direction}]; the figure is drawn without it",
        )
        return None
    try:
        return load_plot_artifacts(
            registered.experiment_name,
            pipeline_runs=run_ids or None,
            models=data.models,
            exclude_models=data.exclude_models,
            expected_target=task.target,
        )
    except MissingExperimentError as missing:
        print(
            f"warning: the registered unrestricted reference {registered.experiment_name!r} is unavailable: {missing}",
        )
        return None


if __name__ == "__main__":
    main()
