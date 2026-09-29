"""Performance progression and same-size XGBoost differences over training size.

Regenerate with:

    uv run python -m src.plotting.sample_size

Every prediction task declared for this family in ``src.plotting.experiments``
is rebuilt in turn, each into its own ``plots/sample_size/<target>/`` directory.
Edit ``VISUAL`` for presentation; ``--target`` narrows the run to selected tasks.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.config import config
from src.plotting.defaults import dataset_label, metric_label, set_plot_style, task_label
from src.plotting.experiments import (
    DATA_SOURCES,
    PlotExperiment,
    reciprocal_single_source,
    tasks_for,
    warn_skipped,
)
from src.plotting.scientific_figstyle import BASELINE, WIDE, figure_grid, panel_labels, save
from src.plotting.utils import (
    FullDataBenchmark,
    MissingExperimentError,
    XGBoostDifferenceEvaluation,
    full_training_count,
    load_plot_artifacts,
    prepare_full_data_benchmark,
    prepare_sample_size_evaluation,
    prepare_xgboost_difference_evaluation,
)
from src.plotting.utils.rendering import (
    instance_plot_styles,
    interval_axis_limits,
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
    output_dir: Path = config.dir_plots / "sample_size"


@dataclass(frozen=True)
class VisualSettings:
    """All locally editable presentation choices for this figure."""

    metrics: tuple[str, ...] = ("roc_auc", "prc_auc")
    datasets: tuple[str, ...] | None = None
    show_ci: bool = True
    ci_level: float = 0.95
    figure_width: float = WIDE
    figure_height_ratio: float = 1.08
    score_scale: float = 100.0
    log_sample_axis: bool = True
    max_sample_ticks: int = 6
    marker_size: float = 3.6
    line_width: float = 1.1
    ci_line_width: float = 0.7
    cap_size: float = 1.8
    ci_alpha: float = 0.65
    sample_axis_label: str = "Training sample count"
    metric_axis_template: str = "{metric} on {dataset} (%)"
    legend_columns: int = 3
    axis_padding_fraction: float = 0.08
    benchmark_line_width: float = 0.9
    benchmark_linestyle: str = "--"
    difference_axis_template: str = "{metric} difference from XGBoost on {dataset} (pp)"
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
        ratio=visual.figure_height_ratio,
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
                style, label = styles[instance]
                estimates = visual.score_scale * model_rows["estimate"].to_numpy(dtype=float)
                lower = visual.score_scale * model_rows["lower"].to_numpy(dtype=float)
                upper = visual.score_scale * model_rows["upper"].to_numpy(dtype=float)
                yerr = np.vstack((estimates - lower, upper - estimates)) if visual.show_ci else None
                ax.errorbar(
                    model_rows["sample_size"],
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

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=visual.legend_columns)
    panel_labels(axes)
    return fig


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
        ratio=visual.figure_height_ratio,
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
                label="XGBoost (zero)",
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

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=visual.legend_columns)
    panel_labels(axes)
    return fig


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
    if visual.log_sample_axis:
        ax.set_xscale("log", base=2)
    ticks = sample_ticks(sample_sizes, visual.max_sample_ticks)
    ax.set_xticks(ticks, [short_count(value) for value in ticks])
    ax.grid(which="both", axis="both")


# ---------------------------------------------------------------------------
# CAPTION AND EXECUTION
# ---------------------------------------------------------------------------


def caption(
    data: SampleSizeEvaluation,
    visual: VisualSettings = VISUAL,
    *,
    benchmark: FullDataBenchmark | None = None,
    benchmark_unavailable_reason: str | None = None,
) -> str:
    metrics = " and ".join(metric_label(metric) for metric in data.metrics)
    datasets = " and ".join(dataset_label(dataset) for dataset in _datasets(data, visual))
    if benchmark is not None and benchmark_unavailable_reason is not None:
        raise ValueError("A benchmark and an unavailable reason cannot both be supplied")
    run_text = _repeat_coverage(data)
    uncertainty = (
        rf"Whiskers are {round(100 * visual.ci_level)}\% percentile intervals from "
        f"{data.bootstrap_count:,} aligned test-cohort bootstrap draws after averaging repeats within each "
        "training count; they quantify held-out cohort resampling uncertainty, not training-repeat variation."
        if visual.show_ci
        else "No uncertainty intervals are shown."
    )
    if data.full_training_size is None:
        local_full = f"The explicit complete {dataset_label(data.trained_on)} training count is unavailable."
    else:
        local_full = (
            f"The complete {dataset_label(data.trained_on)} pool contains {data.full_training_size:,} observations "
            f"and is identified by {_run_count_text(data.full_training_run_count)} explicitly configured at "
            "fraction 1.0."
        )
    benchmark_text = _benchmark_caption(data, benchmark, benchmark_unavailable_reason)
    return (
        rf"\textbf{{Model performance changes with the amount of {dataset_label(data.trained_on)} training data.}} "
        f"{metrics} on the held-out {datasets} cohorts over increasing training sample counts. Points summarize "
        f"{run_text}. {local_full} {benchmark_text} {uncertainty}"
    )


def xgboost_difference_caption(
    data: XGBoostDifferenceEvaluation,
    visual: VisualSettings = VISUAL,
) -> str:
    """Return a self-contained caption for the same-size XGBoost contrasts."""
    metrics = " and ".join(metric_label(metric) for metric in data.metrics)
    datasets = " and ".join(dataset_label(dataset) for dataset in _datasets(data.sample_size, visual))
    uncertainty = (
        rf"Whiskers are {round(100 * data.ci_level)}\% percentile intervals of "
        f"{data.bootstrap_count:,} paired test-cohort bootstrap score differences, matched on training count, "
        "evaluation center, metric, and bootstrap ID after repeat aggregation; they quantify held-out cohort "
        "resampling uncertainty, not training-repeat variation."
        if visual.show_ci
        else "No uncertainty intervals are shown."
    )
    return (
        rf"\textbf{{Differences from same-size XGBoost show how model advantages change with the amount of "
        f"{dataset_label(data.trained_on)} training data for {task_label(data.target)}.}} "
        f"{metrics} differences on the held-out {datasets} evaluation centers are score(model) minus "
        "score(XGBoost) in percentage points, so positive values mean the model performs better. Each training "
        "count matches models with XGBoost from the same experimental setting after repeat aggregation. Repeat "
        f"coverage is {_repeat_coverage(data.sample_size)}. The grey zero line is XGBoost; its identically zero "
        "curve is omitted. "
        f"{uncertainty}"
    )


def _repeat_coverage(data: SampleSizeEvaluation) -> str:
    counts_to_sizes: dict[int, list[int]] = {}
    for setting, count in data.run_counts.items():
        counts_to_sizes.setdefault(count, []).append(int(setting))
    if len(counts_to_sizes) == 1:
        count = next(iter(counts_to_sizes))
        count_text = _run_count_text(count)
        return f"{count_text} at each of {len(data.sample_sizes)} measured training counts"

    coverage = []
    for count, sizes in sorted(counts_to_sizes.items()):
        size_text = ", ".join(f"{size:,}" for size in sorted(sizes))
        coverage.append(f"{_run_count_text(count)} at {size_text} observations")
    return "repeat coverage of " + "; ".join(coverage)


def _benchmark_caption(
    data: SampleSizeEvaluation,
    benchmark: FullDataBenchmark | None,
    unavailable_reason: str | None,
) -> str:
    if benchmark is None:
        reason = unavailable_reason or "the reciprocal full-data input was not supplied"
        return f"The reciprocal full-data benchmark is unavailable because {reason}; no substitute is shown."

    styles = instance_plot_styles(benchmark.model_metadata)
    winners = []
    for metric in data.metrics:
        reference = benchmark.reference(metric)
        winner = styles[str(reference["model_instance"])][1]
        winners.append(f"{metric_label(metric)}: {winner}")
    return (
        f"The grey Best full {dataset_label(benchmark.trained_on)}-trained model reference appears only on the "
        f"{dataset_label(data.trained_on)} evaluation panels. It uses the complete "
        f"{dataset_label(benchmark.trained_on)} pool of {benchmark.training_size:,} observations and the best "
        f"mean score {_benchmark_run_text(benchmark.run_count)} ({'; '.join(winners)}); the winning "
        "identity is fixed per metric rather than reselected in bootstrap draws."
    )


def _benchmark_run_text(count: int) -> str:
    if count == 1:
        return "from one pipeline run"
    return f"after averaging {count} repeated pipeline runs"


def _run_count_text(count: int) -> str:
    return "one pipeline run" if count == 1 else f"{count} repeated pipeline runs"


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
        "--training-source",
        action="append",
        dest="training_sources",
        choices=DATA_SOURCES,
        help="Training source to rebuild; repeat for several. Defaults to both sources.",
    )
    parser.add_argument("--output-dir", type=Path, default=DATA.output_dir)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tasks = tasks_for("sample_size", args.targets, training_sources=args.training_sources)
    run_ids = tuple(args.run_ids) if args.run_ids else DATA.pipeline_runs
    if len(tasks) > 1 and run_ids:
        raise SystemExit("--run-id pins one experiment input; select exactly one --target and --training-source")

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

        visual = replace(VISUAL, metrics=task.metrics, score_scale=task.score_scale)
        local_full_size, local_full_run_count = _load_local_full_metadata(task)
        prepared = prepare_sample_size_evaluation(
            artifacts,
            metrics=visual.metrics,
            ci_level=visual.ci_level,
            full_training_size=local_full_size,
            full_training_run_count=local_full_run_count,
        )
        benchmark, unavailable_reason = _load_reciprocal_benchmark(task, visual)

        output_dir = args.output_dir / prepared.target
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = output_dir / f"sample_size_{prepared.trained_on}_{prepared.target}_performance"
        outputs = save(
            make_figure(prepared, visual, benchmark=benchmark),
            str(stem),
            formats=visual.output_formats,
        )
        print("LaTeX caption:")
        print(
            f"\\caption{{{caption(prepared, visual, benchmark=benchmark, benchmark_unavailable_reason=unavailable_reason)}}}"
        )
        for output in outputs:
            print("figure: " + str(Path(output).relative_to(config.dir_root)))

        xgboost_differences = prepare_xgboost_difference_evaluation(prepared)
        difference_stem = output_dir / f"sample_size_{prepared.trained_on}_{prepared.target}_xgboost_difference"
        difference_outputs = save(
            make_xgboost_difference_figure(xgboost_differences, visual),
            str(difference_stem),
            formats=visual.output_formats,
        )
        print("LaTeX caption (XGBoost difference):")
        print(f"\\caption{{{xgboost_difference_caption(xgboost_differences, visual)}}}")
        for output in difference_outputs:
            print("figure: " + str(Path(output).relative_to(config.dir_root)))


def _load_local_full_metadata(task: PlotExperiment) -> tuple[int | None, int]:
    assert task.experiment_name is not None
    try:
        artifacts = load_plot_artifacts(
            task.experiment_name,
            models=DATA.models,
            exclude_models=DATA.exclude_models,
            full_training_only=True,
            include_bootstrap=False,
            expected_target=task.target,
            expected_training_source=task.training_source,
        )
    except MissingExperimentError as missing:
        print(
            f"warning: {task.label} [{task.direction}] complete local count unavailable: {missing}",
            file=sys.stderr,
        )
        return None, 0
    return full_training_count(artifacts), len(artifacts.run_ids)


def _load_reciprocal_benchmark(
    task: PlotExperiment,
    visual: VisualSettings,
) -> tuple[FullDataBenchmark | None, str | None]:
    reciprocal = reciprocal_single_source(task)
    if reciprocal.experiment_name is None:
        reason = (
            f"no {reciprocal.label} single-source input is registered for "
            f"{dataset_label(str(reciprocal.training_source))} training"
        )
        _warn_benchmark_unavailable(task, reciprocal, reason)
        return None, reason

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
        reason = (
            f"the registered {dataset_label(str(reciprocal.training_source))}-trained "
            f"{reciprocal.label} input has no available explicit fraction-1.0 result"
        )
        _warn_benchmark_unavailable(task, reciprocal, missing)
        return None, reason

    benchmark = prepare_full_data_benchmark(
        artifacts,
        evaluated_on=str(task.training_source),
        metrics=visual.metrics,
    )
    print("Reciprocal full-data pipeline runs: " + ", ".join(benchmark.run_ids))
    return benchmark, None


def _warn_benchmark_unavailable(task: PlotExperiment, reciprocal: PlotExperiment, reason: object) -> None:
    print(
        f"warning: generating {task.label} [{task.direction}] without the reciprocal full-data benchmark "
        f"from [{reciprocal.direction}]: {reason}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
