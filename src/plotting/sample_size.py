"""Performance progression over increasing training sample sizes.

Regenerate with:

    uv run python -m src.plotting.sample_size

Edit ``DATA`` for experiment/model selection and ``VISUAL`` for presentation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.config import config
from src.plotting.defaults import dataset_label, metric_label, set_plot_style
from src.plotting.scientific_figstyle import WIDE, figure_grid, panel_labels, save
from src.plotting.utils import load_plot_artifacts, prepare_sample_size_evaluation
from src.plotting.utils.rendering import instance_plot_styles, interval_axis_limits
from src.plotting.utils.sample_size import SampleSizeEvaluation


@dataclass(frozen=True)
class DataSettings:
    """Experiment/model selection and output location."""

    experiment_name: str = "sample_size_mimic_mortality"
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
    output_formats: tuple[str, ...] = ("pdf",)


DATA = DataSettings()
VISUAL = VisualSettings()


# ---------------------------------------------------------------------------
# FIGURE
# ---------------------------------------------------------------------------


def make_figure(data: SampleSizeEvaluation, visual: VisualSettings = VISUAL) -> Figure:
    """Draw metric progression over training sample count for each cohort."""
    set_plot_style()
    datasets = _datasets(data, visual)
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
            ax.set_ylim(y_limits)
            ax.set_ylabel(
                visual.metric_axis_template.format(
                    metric=metric_label(metric),
                    dataset=dataset_label(dataset),
                )
            )
            if metric_index == len(data.metrics) - 1:
                ax.set_xlabel(visual.sample_axis_label)
            _format_sample_axis(ax, data.sample_sizes, visual)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=visual.legend_columns)
    panel_labels(axes)
    return fig


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
    ticks = _sample_ticks(sample_sizes, visual.max_sample_ticks)
    ax.set_xticks(ticks, [_short_count(value) for value in ticks])
    ax.grid(which="both", axis="both")


def _sample_ticks(sample_sizes: tuple[int, ...], maximum: int) -> tuple[int, ...]:
    if maximum < 2:
        raise ValueError("max_sample_ticks must be at least two")
    if len(sample_sizes) <= maximum:
        return sample_sizes
    indices = np.linspace(0, len(sample_sizes) - 1, maximum).round().astype(int)
    return tuple(sample_sizes[index] for index in np.unique(indices))


def _short_count(value: int) -> str:
    if value < 1_000:
        return f"{value}"
    thousands = f"{value / 1_000:.1f}".rstrip("0").rstrip(".")
    return f"{thousands}k"


# ---------------------------------------------------------------------------
# CAPTION AND EXECUTION
# ---------------------------------------------------------------------------


def caption(data: SampleSizeEvaluation, visual: VisualSettings = VISUAL) -> str:
    metrics = " and ".join(metric_label(metric) for metric in data.metrics)
    datasets = " and ".join(dataset_label(dataset) for dataset in _datasets(data, visual))
    counts = set(data.run_counts.values())
    if counts == {1}:
        run_text = "one pipeline run at each sample size"
    elif len(counts) == 1:
        run_text = f"{next(iter(counts))} repeated pipeline runs at each sample size"
    else:
        run_text = "the available repeated pipeline runs at each sample size"
    uncertainty = (
        rf"Whiskers are {round(100 * visual.ci_level)}\% percentile intervals from "
        f"{data.bootstrap_count:,} aligned cohort-bootstrap draws."
        if visual.show_ci
        else "No uncertainty intervals are shown."
    )
    return (
        rf"\textbf{{Model performance changes with the amount of {dataset_label(data.trained_on)} training data.}} "
        f"{metrics} on the held-out {datasets} cohorts over increasing training sample counts. Points summarize "
        f"{run_text}. {uncertainty}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default=DATA.experiment_name)
    parser.add_argument("--run-id", action="append", dest="run_ids")
    parser.add_argument("--output-dir", type=Path, default=DATA.output_dir)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_ids = tuple(args.run_ids) if args.run_ids else DATA.pipeline_runs
    artifacts = load_plot_artifacts(
        args.experiment_name,
        pipeline_runs=run_ids,
        models=DATA.models,
        exclude_models=DATA.exclude_models,
    )
    prepared = prepare_sample_size_evaluation(
        artifacts,
        metrics=VISUAL.metrics,
        ci_level=VISUAL.ci_level,
    )

    output_dir = args.output_dir / prepared.target
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"sample_size_{prepared.trained_on}_{prepared.target}_performance"
    outputs = save(make_figure(prepared), str(stem), formats=VISUAL.output_formats)
    print("LaTeX caption:")
    print(f"\\caption{{{caption(prepared)}}}")
    for output in outputs:
        print("figure: " + str(Path(output).relative_to(config.dir_root)))


if __name__ == "__main__":
    main()
