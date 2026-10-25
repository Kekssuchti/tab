"""Performance progression and same-size XGBoost differences over training size.

Regenerate with:

    uv run python -m src.plotting.sample_size

Every prediction task declared for this family in ``src.plotting.experiments``
is rebuilt in turn, each into its own ``plots/sample_size/<auroc|auprc>/<all|main>/<target>/`` directory.
Each target also gets ``combined_performance.pdf`` with EUH training above MIMIC
training and EUH evaluation on the left. Use ``--combined-only`` to generate only
this addition without rewriting the single-direction figures.
Edit ``VISUAL`` for presentation; ``--target`` narrows the run to selected tasks.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.config import config
from src.plotting.defaults import MetricPanelSettings, dataset_label, metric_label, set_plot_style
from src.plotting.experiments import (
    DATA_SOURCES,
    PlotExperiment,
    reciprocal_single_source,
    tasks_for,
    warn_skipped,
)
from src.plotting.model_sets import MODEL_SET, PLOT_METRICS, plot_dir
from src.plotting.scientific_figstyle import BASELINE, WIDE, figure_grid, panel_labels, save
from src.plotting.utils import (
    FullDataBenchmark,
    MissingExperimentError,
    XGBoostDifferenceEvaluation,
    load_plot_artifacts,
    prepare_full_data_benchmark,
    prepare_sample_size_evaluation,
    prepare_xgboost_difference_evaluation,
)
from src.plotting.utils.rendering import (
    instance_plot_styles,
    interval_axis_limits,
    log_sample_ticks,
    sample_ticks,
    short_count,
)
from src.plotting.utils.sample_size import SampleSizeEvaluation


@dataclass(frozen=True)
class DataSettings:
    """Run/model selection and output location."""

    pipeline_runs: tuple[str, ...] | None = None
    models: tuple[str, ...] | None = None
    # Model names to leave out of every figure.
    exclude_models: tuple[str, ...] | None = None
    output_dir: Path = plot_dir("sample_size")


@dataclass(frozen=True)
class VisualSettings(MetricPanelSettings):
    """All locally editable presentation choices for this figure."""

    metrics: tuple[str, ...] = PLOT_METRICS
    datasets: tuple[str, ...] | None = None
    show_ci: bool = True
    ci_level: float = 0.95
    figure_width: float = WIDE
    row_height: float = 0.54
    log_sample_axis: bool = True
    max_sample_ticks: int = 6
    marker_size: float = 3.6
    line_width: float = 1.1
    ci_line_width: float = 0.7
    cap_size: float = 1.8
    ci_alpha: float = 0.65
    sample_axis_label: str = "Training sample count"
    metric_axis_template: str = "{metric} on {dataset}"
    legend_columns: int = 4 if MODEL_SET == "main" else 6
    axis_padding_fraction: float = 0.08
    benchmark_line_width: float = 0.9
    benchmark_linestyle: str = "--"
    difference_axis_template: str = "$\\Delta$ {metric} to XGBoost on {dataset}"
    # Combined figure only: off-diagonal train/test combinations.
    shade_external_panels: bool = True
    external_panel_color: str = "#F7F7F7"
    output_formats: tuple[str, ...] = ("pdf",)


DATA = DataSettings()
VISUAL = VisualSettings()


# ---------------------------------------------------------------------------
# FIGURE
# ---------------------------------------------------------------------------


def make_figure(
    data: SampleSizeEvaluation,
    visual: VisualSettings = VISUAL,
    *,
    benchmark: FullDataBenchmark | None = None,
) -> Figure:
    """Draw metric progression over training sample count for each cohort."""
    set_plot_style()
    datasets = _datasets(data, visual)
    if benchmark is not None:
        _validate_benchmark(data, benchmark)
    fig, axes = figure_grid(
        len(data.metrics),
        len(datasets),
        width=visual.figure_width,
        row_height=visual.row_height,
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    styles = instance_plot_styles(data.model_metadata)
    rows = data.performance.copy()
    rows["sample_size"] = pd.to_numeric(rows["setting"])

    for metric_index, metric in enumerate(data.metrics):
        metric_rows = rows.loc[rows["metric"].eq(metric)]
        limit_rows = metric_rows
        if benchmark is not None and data.trained_on in datasets:
            reference = benchmark.reference(metric)
            reference_limit = pd.DataFrame(
                {
                    "estimate": [reference["estimate"]],
                    "lower": [reference["estimate"]],
                    "upper": [reference["estimate"]],
                }
            )
            limit_rows = pd.concat((metric_rows, reference_limit), ignore_index=True)
        y_limits = interval_axis_limits(
            limit_rows,
            scale=visual.score_scale,
            show_ci=visual.show_ci,
            padding_fraction=visual.axis_padding_fraction,
        )
        for dataset_index, dataset in enumerate(datasets):
            ax = axes[metric_index, dataset_index]
            dataset_rows = metric_rows.loc[metric_rows["dataset"].eq(dataset)]
            for instance in data.model_instances:
                model_rows = dataset_rows.loc[dataset_rows["model_instance"].eq(instance)].sort_values("sample_size")
                if model_rows.empty:
                    continue
                _draw_performance_curve(ax, model_rows, styles[instance], visual)
            if benchmark is not None and dataset == data.trained_on:
                reference = benchmark.reference(metric)
                ax.axhline(
                    visual.score_scale * float(reference["estimate"]),
                    color=BASELINE,
                    linewidth=visual.benchmark_line_width,
                    linestyle=visual.benchmark_linestyle,
                    label=f"Best full {dataset_label(benchmark.trained_on)}-trained model",
                    zorder=1,
                )
            ax.set_ylim(y_limits)
            ax.set_ylabel(
                visual.metric_axis_template.format(
                    metric=metric_label(metric),
                    dataset=dataset_label(dataset),
                )
            )
            ax.set_xlim(left=85)
            if metric_index == len(data.metrics) - 1:
                ax.set_xlabel(visual.sample_axis_label)
            _format_sample_axis(ax, data.sample_sizes, visual)

    _add_legend(fig, axes[0, 0], visual.legend_columns)
    panel_labels(axes)
    return fig


def make_combined_figure(
    euh: SampleSizeEvaluation,
    mimic: SampleSizeEvaluation,
    visual: VisualSettings = VISUAL,
    *,
    euh_benchmark: FullDataBenchmark | None = None,
    mimic_benchmark: FullDataBenchmark | None = None,
) -> Figure:
    """Stack reciprocal sweeps: EUH training above, EUH evaluation at left.

    Each row has its own training counts, ticks and grid; all four panels use
    the same score limits. The dashed reference appears on the internal panels
    and always denotes the best full-data model from the other training center.
    """
    if (euh.trained_on, mimic.trained_on) != ("tudd", "mimic"):
        raise ValueError("Combined performance requires EUH and MIMIC training, in that order")
    if euh.target != mimic.target or euh.metrics != mimic.metrics or len(euh.metrics) != 1:
        raise ValueError("Combined performance requires the same target and one shared metric")
    datasets = ("tudd", "mimic")
    for data, benchmark in ((euh, euh_benchmark), (mimic, mimic_benchmark)):
        if not set(datasets).issubset(data.datasets):
            raise ValueError(f"The {data.trained_on} sweep must evaluate both EUH and MIMIC")
        if benchmark is not None:
            _validate_benchmark(data, benchmark)

    set_plot_style()
    fig, axes = _combined_grid(visual)
    metric = euh.metrics[0]
    limit_rows = pd.concat((euh.performance, mimic.performance), ignore_index=True)
    references = [
        {"estimate": value, "lower": value, "upper": value}
        for benchmark in (euh_benchmark, mimic_benchmark)
        if benchmark is not None
        for value in (float(benchmark.reference(metric)["estimate"]),)
    ]
    if references:
        limit_rows = pd.concat((limit_rows, pd.DataFrame(references)), ignore_index=True)
    y_limits = interval_axis_limits(
        limit_rows,
        scale=visual.score_scale,
        show_ci=visual.show_ci,
        padding_fraction=visual.axis_padding_fraction,
    )
    metadata = pd.concat((euh.model_metadata, mimic.model_metadata), ignore_index=True).drop_duplicates()
    if metadata["model_instance"].duplicated().any():
        raise ValueError("Reciprocal sweeps disagree on model-instance identities")
    styles = instance_plot_styles(metadata)
    for row_index, (data, benchmark) in enumerate(((euh, euh_benchmark), (mimic, mimic_benchmark))):
        rows = data.performance.copy()
        rows["sample_size"] = pd.to_numeric(rows["setting"])
        for column, dataset in enumerate(datasets):
            ax = axes[row_index, column]
            if visual.shade_external_panels and dataset != data.trained_on:
                ax.set_facecolor(visual.external_panel_color)
            dataset_rows = rows.loc[rows["dataset"].eq(dataset)]
            for instance in data.model_instances:
                model_rows = dataset_rows.loc[dataset_rows["model_instance"].eq(instance)].sort_values("sample_size")
                if not model_rows.empty:
                    _draw_performance_curve(ax, model_rows, styles[instance], visual)
            if benchmark is not None and dataset == data.trained_on:
                ax.axhline(
                    visual.score_scale * float(benchmark.reference(metric)["estimate"]),
                    color=BASELINE,
                    linewidth=visual.benchmark_line_width,
                    linestyle=visual.benchmark_linestyle,
                    label="Best full external model",
                    zorder=1,
                )
            ax.set_ylim(y_limits)
            if column == 0:
                ax.set_ylabel(metric_label(metric))
            ax.set_xlim(left=85)
            ax.set_xlabel(visual.sample_axis_label)
            _format_sample_axis(ax, data.sample_sizes, visual)

    _add_legend(fig, axes, visual.legend_columns)
    panel_labels(axes)
    return fig


def _combined_grid(visual: VisualSettings):
    """Reserve real layout cells for table-style train/test headings."""
    fig = plt.figure(
        figsize=(visual.figure_width, 2 * visual.figure_width * visual.row_height + 0.35),
        layout="constrained",
    )
    grid = fig.add_gridspec(
        4,
        4,
        width_ratios=(0.035, 0.05, 1, 1),
        height_ratios=(0.035, 0.05, 1, 1),
    )
    headings = (
        (grid[0, 2:], "Test set", 0, "bold"),
        (grid[1, 2], dataset_label("tudd"), 0, "bold"),
        (grid[1, 3], dataset_label("mimic"), 0, "bold"),
        (grid[2:, 0], "Train set", 90, "bold"),
        (grid[2, 1], dataset_label("tudd"), 90, "bold"),
        (grid[3, 1], dataset_label("mimic"), 90, "bold"),
    )
    for cell, text, rotation, weight in headings:
        heading = fig.add_subplot(cell)
        heading.set_axis_off()
        heading.text(
            0.5,
            0.5,
            text,
            transform=heading.transAxes,
            ha="center",
            va="center",
            rotation=rotation,
            fontweight=weight,
        )

    axes = np.empty((2, 2), dtype=object)
    for row in range(2):
        for column in range(2):
            axes[row, column] = fig.add_subplot(
                grid[row + 2, column + 2],
                sharex=axes[row, 0] if column else None,
                sharey=axes[0, 0] if row or column else None,
            )
            axes[row, column].tick_params(axis="y", labelleft=True)
    return fig, axes


def _draw_performance_curve(ax, rows: pd.DataFrame, model_style, visual: VisualSettings) -> None:
    """Keep single-direction and combined curves visually identical."""
    style, label = model_style
    estimates = visual.score_scale * rows["estimate"].to_numpy(dtype=float)
    lower = visual.score_scale * rows["lower"].to_numpy(dtype=float)
    upper = visual.score_scale * rows["upper"].to_numpy(dtype=float)
    yerr = np.vstack((estimates - lower, upper - estimates)) if visual.show_ci else None
    ax.errorbar(
        rows["sample_size"],
        estimates,
        yerr=yerr,
        color=style.color,
        marker=style.marker,
        linestyle=style.linestyle,
        markersize=visual.marker_size,
        linewidth=visual.line_width,
        elinewidth=visual.ci_line_width,
        capsize=visual.cap_size if visual.show_ci else 0,
        alpha=visual.ci_alpha if visual.show_ci else 1.0,
        label=label,
    )


def make_xgboost_difference_figure(
    data: XGBoostDifferenceEvaluation,
    visual: VisualSettings = VISUAL,
) -> Figure:
    """Draw same-size model-minus-XGBoost score differences for each cohort."""
    set_plot_style()
    datasets = _datasets(data.sample_size, visual)
    fig, axes = figure_grid(
        len(data.metrics),
        len(datasets),
        width=visual.figure_width,
        row_height=visual.row_height,
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    styles = instance_plot_styles(data.model_metadata)
    rows = data.performance.copy()
    rows["sample_size"] = pd.to_numeric(rows["setting"])

    for metric_index, metric in enumerate(data.metrics):
        metric_rows = rows.loc[rows["metric"].eq(metric)]
        y_limits = interval_axis_limits(
            metric_rows,
            scale=visual.score_scale,
            include_zero=True,
            show_ci=visual.show_ci,
            padding_fraction=visual.axis_padding_fraction,
        )
        for dataset_index, dataset in enumerate(datasets):
            ax = axes[metric_index, dataset_index]
            ax.axhline(
                0.0,
                color=BASELINE,
                linewidth=visual.benchmark_line_width,
                linestyle=visual.benchmark_linestyle,
                label="XGBoost",
                zorder=1,
            )
            dataset_rows = metric_rows.loc[metric_rows["dataset"].eq(dataset)]
            for instance in data.model_instances:
                model_rows = dataset_rows.loc[dataset_rows["model_instance"].eq(instance)].sort_values("sample_size")
                if model_rows.empty:
                    continue
                style, label = styles[instance]
                sample_sizes = model_rows["sample_size"].to_numpy(dtype=float)
                estimates = visual.score_scale * model_rows["estimate"].to_numpy(dtype=float)
                ax.plot(
                    sample_sizes,
                    estimates,
                    color=style.color,
                    marker=style.marker,
                    linestyle=style.linestyle,
                    markersize=visual.marker_size,
                    linewidth=visual.line_width,
                    alpha=visual.ci_alpha if visual.show_ci else 1.0,
                    label=label,
                    zorder=2,
                )
                if visual.show_ci:
                    lower = visual.score_scale * model_rows["lower"].to_numpy(dtype=float)
                    upper = visual.score_scale * model_rows["upper"].to_numpy(dtype=float)
                    interval_errors = np.vstack((estimates - lower, upper - estimates))
                    ax.errorbar(
                        sample_sizes,
                        estimates,
                        yerr=interval_errors,
                        fmt="none",
                        color=style.color,
                        elinewidth=visual.ci_line_width,
                        capsize=visual.cap_size,
                        alpha=visual.ci_alpha,
                        zorder=2,
                    )
            ax.set_ylim(y_limits)
            ax.set_ylabel(
                visual.difference_axis_template.format(
                    metric=metric_label(metric),
                    dataset=dataset_label(dataset),
                )
            )
            ax.set_xlim(left=85)
            if metric_index == len(data.metrics) - 1:
                ax.set_xlabel(visual.sample_axis_label)
            _format_sample_axis(ax, data.sample_sizes, visual)

    _add_legend(fig, axes[0, 0], visual.legend_columns)
    panel_labels(axes)
    return fig


def _add_legend(fig: Figure, axes, columns: int) -> None:
    # Collect the union: a model or reference may occur in only one direction.
    entries = {}
    for ax in np.asarray(axes, dtype=object).ravel():
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            entries.setdefault(label, handle)
    labels = list(entries)
    handles = list(entries.values())
    # Matplotlib fills columns; transpose the input to display entries row-wise.
    order = [index for column in range(columns) for index in range(column, len(handles), columns)]
    fig.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="outside upper center",
        ncol=columns,
    )


def _validate_benchmark(data: SampleSizeEvaluation, benchmark: FullDataBenchmark) -> None:
    if benchmark.target != data.target:
        raise ValueError(f"Benchmark target {benchmark.target!r} does not match learning-curve target {data.target!r}")
    if benchmark.evaluated_on != data.trained_on:
        raise ValueError(
            f"Benchmark is evaluated on {benchmark.evaluated_on!r}, not the local {data.trained_on!r} panel"
        )
    if benchmark.trained_on == data.trained_on:
        raise ValueError("Sample-size benchmark must come from the reciprocal training center")
    missing_metrics = sorted(set(data.metrics) - set(benchmark.references["metric"].astype(str)))
    if missing_metrics:
        raise ValueError("Benchmark is missing metrics: " + ", ".join(missing_metrics))


def _datasets(data: SampleSizeEvaluation, visual: VisualSettings) -> tuple[str, ...]:
    if visual.datasets is not None:
        missing = sorted(set(visual.datasets) - set(data.datasets))
        if missing:
            raise ValueError("Requested datasets are unavailable: " + ", ".join(missing))
        return visual.datasets
    external = [dataset for dataset in data.datasets if dataset != data.trained_on]
    return (data.trained_on, *external)


def _format_sample_axis(ax, sample_sizes: tuple[int, ...], visual: VisualSettings) -> None:
    ticks = sample_ticks(sample_sizes, visual.max_sample_ticks)
    if visual.log_sample_axis:
        ax.set_xscale("log", base=2)
        ticks = log_sample_ticks(sample_sizes, visual.max_sample_ticks)
    ax.set_xticks(ticks, [short_count(value) for value in ticks])
    ax.grid(which="both", axis="both")


# ---------------------------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        help=(
            "Prediction task to rebuild, for example mortality; repeat for several. "
            "Defaults to every task this family declares."
        ),
    )
    parser.add_argument("--run-id", action="append", dest="run_ids")
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Write only combined_performance; leave existing single-direction PDFs untouched.",
    )
    parser.add_argument(
        "--training-source",
        action="append",
        dest="training_sources",
        choices=DATA_SOURCES,
        help="Training source to rebuild; repeat for several. Defaults to both sources.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tasks = tasks_for("sample_size", args.targets, training_sources=args.training_sources)
    run_ids = tuple(args.run_ids) if args.run_ids else DATA.pipeline_runs
    if len(tasks) > 1 and run_ids:
        raise SystemExit("--run-id pins one experiment input; select exactly one --target and --training-source")

    pending: dict[str, dict[str, tuple[SampleSizeEvaluation, FullDataBenchmark | None]]] = {}
    for task in tasks:
        experiment_name = task.experiment_name
        print(f"\n=== {task.label} [{task.direction}] ({experiment_name or 'not registered'})")
        if experiment_name is None:
            warn_skipped(task, "the intended experiment has not been registered yet")
            continue
        try:
            artifacts = load_plot_artifacts(
                experiment_name,
                pipeline_runs=run_ids,
                models=DATA.models,
                exclude_models=DATA.exclude_models,
                expected_target=task.target,
                expected_training_source=task.training_source,
            )
        except MissingExperimentError as missing:
            warn_skipped(task, missing)
            continue

        visual = VISUAL
        prepared = prepare_sample_size_evaluation(
            artifacts,
            metrics=visual.metrics,
            ci_level=visual.ci_level,
        )
        del artifacts
        benchmark = _load_reciprocal_benchmark(task, visual)

        output_dir = DATA.output_dir / prepared.target
        output_dir.mkdir(parents=True, exist_ok=True)
        if not args.combined_only:
            stem = output_dir / f"{prepared.trained_on}_performance"
            outputs = save(
                make_figure(prepared, visual, benchmark=benchmark),
                str(stem),
                formats=visual.output_formats,
            )
            for output in outputs:
                print("figure: " + str(Path(output).relative_to(config.dir_root)))

            xgboost_differences = prepare_xgboost_difference_evaluation(prepared)
            difference_stem = output_dir / f"{prepared.trained_on}_xgboost_difference"
            difference_outputs = save(
                make_xgboost_difference_figure(xgboost_differences, visual),
                str(difference_stem),
                formats=visual.output_formats,
            )
            for output in difference_outputs:
                print("figure: " + str(Path(output).relative_to(config.dir_root)))
            del xgboost_differences

        directions = pending.setdefault(prepared.target, {})
        # The combined drawing only needs summaries; do not retain millions of
        # bootstrap/repeat rows while loading the opposite training direction.
        curves = replace(
            prepared,
            grouped=replace(prepared.grouped, bootstrap_scores=pd.DataFrame(), repeat_points=pd.DataFrame()),
        )
        directions[prepared.trained_on] = (curves, benchmark)
        if {"tudd", "mimic"}.issubset(directions):
            euh, euh_benchmark = directions["tudd"]
            mimic, mimic_benchmark = directions["mimic"]
            outputs = save(
                make_combined_figure(
                    euh,
                    mimic,
                    visual,
                    euh_benchmark=euh_benchmark,
                    mimic_benchmark=mimic_benchmark,
                ),
                str(output_dir / "combined_performance"),
                formats=visual.output_formats,
            )
            for output in outputs:
                print("figure: " + str(Path(output).relative_to(config.dir_root)))
            del pending[prepared.target]
            directions.clear()
            del euh, mimic
        del prepared, curves

    for target in pending:
        print(f"warning: skipping {target} combined performance: both training sources are required", file=sys.stderr)


def _load_reciprocal_benchmark(
    task: PlotExperiment,
    visual: VisualSettings,
) -> FullDataBenchmark | None:
    reciprocal = reciprocal_single_source(task)
    if reciprocal.experiment_name is None:
        reason = (
            f"no {reciprocal.label} single-source input is registered for "
            f"{dataset_label(str(reciprocal.training_source))} training"
        )
        _warn_benchmark_unavailable(task, reciprocal, reason)
        return None

    try:
        artifacts = load_plot_artifacts(
            reciprocal.experiment_name,
            models=DATA.models,
            exclude_models=DATA.exclude_models,
            full_training_only=True,
            expected_target=reciprocal.target,
            expected_training_source=reciprocal.training_source,
        )
    except MissingExperimentError as missing:
        _warn_benchmark_unavailable(task, reciprocal, missing)
        return None

    benchmark = prepare_full_data_benchmark(
        artifacts,
        evaluated_on=str(task.training_source),
        metrics=visual.metrics,
    )
    print("Reciprocal full-data pipeline runs: " + ", ".join(benchmark.run_ids))
    return benchmark


def _warn_benchmark_unavailable(task: PlotExperiment, reciprocal: PlotExperiment, reason: object) -> None:
    print(
        f"warning: generating {task.label} [{task.direction}] without the reciprocal full-data benchmark "
        f"from [{reciprocal.direction}]: {reason}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
