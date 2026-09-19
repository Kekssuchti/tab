"""Full-data baseline performance and generalizability figures.

Regenerate with:

    uv run python -m src.plotting.baseline_transfer

Edit ``DATA`` to select the source experiment and ``VISUAL`` to change figure
presentation. CLI arguments only provide convenient run/output overrides.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from src.config import config
from src.plotting.defaults import dataset_label, metric_label, set_plot_style
from src.plotting.scientific_figstyle import BASELINE, WIDE, figure_grid, panel_labels, save
from src.plotting.utils import aggregate_evaluation_runs, load_plot_artifacts, prepare_transfer_summary
from src.plotting.utils.rendering import draw_model_forest, instance_plot_styles, interval_axis_limits
from src.plotting.utils.transfer import TransferSummary

# ---------------------------------------------------------------------------
# EDIT THESE SETTINGS
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataSettings:
    """Experiment selection and output location."""

    experiment_name: str = "sample_size_mimic_mortality"
    pipeline_runs: tuple[str, ...] | None = None
    full_training_only: bool = True
    output_dir: Path = config.dir_plots / "baseline"


@dataclass(frozen=True)
class VisualSettings:
    """All locally editable presentation choices for these figures."""

    metrics: tuple[str, ...] = ("roc_auc", "prc_auc")
    show_ci: bool = True
    ci_level: float = 0.95
    figure_width: float = WIDE
    performance_height_ratio: float = 1.08
    generalizability_height_ratio: float = 1.08
    score_scale: float = 100.0
    model_axis_label: str = "Model"
    performance_axis_template: str = "{metric} on {dataset} (%)"
    degradation_axis_template: str = "{metric} degradation (pp)"
    relative_axis_template: str = "{metric} relative external loss (pp)"
    shade_baselines: bool = True
    baseline_band_alpha: float = 0.13
    marker_size: float = 4.4
    ci_line_width: float = 1.0
    cap_size: float = 2.0
    axis_padding_fraction: float = 0.08
    output_formats: tuple[str, ...] = ("pdf",)


DATA = DataSettings()
VISUAL = VisualSettings()


# ---------------------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------------------


def make_performance_figure(data: TransferSummary, visual: VisualSettings = VISUAL) -> Figure:
    """Draw absolute performance on the in-domain and external cohorts."""
    set_plot_style()
    fig, axes = figure_grid(
        len(data.metrics),
        2,
        width=visual.figure_width,
        ratio=visual.performance_height_ratio,
        sharey=True,
        squeeze=False,
    )
    styles = instance_plot_styles(data.model_metadata)
    datasets = (data.trained_on, data.external_dataset)

    for metric_index, metric in enumerate(data.metrics):
        metric_rows = data.performance.loc[data.performance["metric"].eq(metric)]
        limits = interval_axis_limits(
            metric_rows,
            scale=visual.score_scale,
            show_ci=visual.show_ci,
            padding_fraction=visual.axis_padding_fraction,
        )
        for dataset_index, dataset in enumerate(datasets):
            ax = axes[metric_index, dataset_index]
            rows = metric_rows.loc[metric_rows["dataset"].eq(dataset)]
            _draw_panel(
                ax,
                rows,
                data,
                styles,
                visual,
                show_model_labels=dataset_index == 0,
            )
            ax.set_xlim(limits)
            ax.set_xlabel(
                visual.performance_axis_template.format(
                    metric=metric_label(metric),
                    dataset=dataset_label(dataset),
                )
            )
            if dataset_index == 0:
                ax.set_ylabel(visual.model_axis_label)

    panel_labels(axes)
    return fig


def make_generalizability_figure(data: TransferSummary, visual: VisualSettings = VISUAL) -> Figure:
    """Draw positive transfer degradation and relative external loss."""
    set_plot_style()
    fig, axes = figure_grid(
        len(data.metrics),
        2,
        width=visual.figure_width,
        ratio=visual.generalizability_height_ratio,
        sharey=False,
        squeeze=False,
    )
    styles = instance_plot_styles(data.model_metadata)

    for metric_index, metric in enumerate(data.metrics):
        degradation = data.degradation.loc[data.degradation["metric"].eq(metric)]
        relative = data.relative_loss.loc[data.relative_loss["metric"].eq(metric)]
        limits = interval_axis_limits(
            pd.concat((degradation, relative), ignore_index=True),
            scale=visual.score_scale,
            include_zero=True,
            show_ci=visual.show_ci,
            padding_fraction=visual.axis_padding_fraction,
        )
        panels = (
            (degradation, visual.degradation_axis_template),
            (relative, visual.relative_axis_template),
        )
        for contrast_index, (rows, axis_template) in enumerate(panels):
            ax = axes[metric_index, contrast_index]
            _draw_panel(
                ax,
                rows,
                data,
                styles,
                visual,
                show_model_labels=contrast_index == 0,
            )
            ax.axvline(0, color=BASELINE, linewidth=0.8, linestyle="--", zorder=1)
            ax.set_xlim(limits)
            ax.set_xlabel(axis_template.format(metric=metric_label(metric)))
            ax.set_ylabel(visual.model_axis_label)
            if contrast_index != 0:
                ax.yaxis.label.set_visible(False)

    panel_labels(axes)
    return fig


def _draw_panel(ax, rows, data, styles, visual, *, show_model_labels: bool) -> None:
    draw_model_forest(
        ax,
        rows,
        data.model_instances,
        styles,
        scale=visual.score_scale,
        show_ci=visual.show_ci,
        show_model_labels=show_model_labels,
        shade_baselines=visual.shade_baselines,
        baseline_band_alpha=visual.baseline_band_alpha,
        marker_size=visual.marker_size,
        ci_line_width=visual.ci_line_width,
        cap_size=visual.cap_size,
    )


# ---------------------------------------------------------------------------
# CAPTIONS AND EXECUTION
# ---------------------------------------------------------------------------


def performance_caption(data: TransferSummary, visual: VisualSettings = VISUAL) -> str:
    panel_groups = _metric_panel_groups(data.metrics)
    uncertainty = _uncertainty_caption(data, visual)
    return (
        rf"\textbf{{Full-data {dataset_label(data.trained_on)} models show dataset-dependent performance on the "
        rf"external {dataset_label(data.external_dataset)} cohort.}} "
        f"{panel_groups} for models trained on {dataset_label(data.trained_on)} and evaluated on held-out "
        f"{dataset_label(data.trained_on)} and {dataset_label(data.external_dataset)} cohorts. Points are "
        f"run-averaged estimates. {uncertainty} Circles denote classical baselines and triangles tabular foundation "
        r"models."
    )


def generalizability_caption(data: TransferSummary, visual: VisualSettings = VISUAL) -> str:
    references = []
    styles = instance_plot_styles(data.model_metadata)
    for metric in data.metrics:
        rows = data.relative_loss.loc[data.relative_loss["metric"].eq(metric)]
        reference_instance = str(rows["reference_model_instance"].iloc[0])
        references.append(f"{metric_label(metric)}: {styles[reference_instance][1]}")
    uncertainty = _uncertainty_caption(data, visual)
    prevalence_caveat = (
        " AUPRC degradation also reflects differences in outcome prevalence between cohorts."
        if "prc_auc" in data.metrics
        else ""
    )
    return (
        r"\textbf{Transfer degradation and rank loss expose different generalizability failures.} "
        f"Models were trained on {dataset_label(data.trained_on)} and transferred to "
        f"{dataset_label(data.external_dataset)}. Left panels show positive in-domain minus external performance "
        f"loss; right panels show loss relative to the strongest transferred model ({'; '.join(references)}), whose "
        f"loss is zero. Points are run-averaged estimates. {uncertainty}{prevalence_caveat}"
    )


def _metric_panel_groups(metrics: tuple[str, ...]) -> str:
    groups = []
    for index, metric in enumerate(metrics):
        first = chr(ord("a") + 2 * index)
        second = chr(ord("a") + 2 * index + 1)
        groups.append(f"{metric_label(metric)} ({first}, {second})")
    return " and ".join(groups)


def _uncertainty_caption(data: TransferSummary, visual: VisualSettings) -> str:
    run_text = "one pipeline run" if data.run_count == 1 else f"{data.run_count} repeated pipeline runs"
    if not visual.show_ci:
        return f"No uncertainty intervals are shown; estimates summarize {run_text}."
    confidence = round(100 * visual.ci_level)
    aggregation_text = (
        "for one pipeline run"
        if data.run_count == 1
        else f"after averaging {data.run_count} repeated pipeline runs"
    )
    return (
        rf"Whiskers are {confidence}\% percentile intervals from {data.bootstrap_count:,} aligned cohort-bootstrap "
        f"draws {aggregation_text}."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default=DATA.experiment_name)
    parser.add_argument(
        "--run-id",
        action="append",
        dest="run_ids",
        help="Explicit pipeline MLflow run ID; repeat to average runs. Defaults to full-training-size runs.",
    )
    parser.add_argument("--output-dir", type=Path, default=DATA.output_dir)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_ids = tuple(args.run_ids) if args.run_ids else DATA.pipeline_runs
    artifacts = load_plot_artifacts(
        args.experiment_name,
        pipeline_runs=run_ids,
        full_training_only=run_ids is None and DATA.full_training_only,
    )
    aggregated = aggregate_evaluation_runs(
        artifacts,
        metrics=VISUAL.metrics,
        ci_level=VISUAL.ci_level,
    )
    prepared = prepare_transfer_summary(aggregated)
    print("Selected pipeline runs: " + ", ".join(prepared.run_ids))

    output_dir = args.output_dir / prepared.target
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"baseline_{prepared.trained_on}_{prepared.target}"
    performance_path = output_dir / f"{prefix}_performance"
    generalizability_path = output_dir / f"{prefix}_generalizability"
    save(make_performance_figure(prepared), str(performance_path), formats=VISUAL.output_formats)
    save(make_generalizability_figure(prepared), str(generalizability_path), formats=VISUAL.output_formats)

    print("Performance figure caption:")
    print(f"\\caption{{{performance_caption(prepared)}}}")
    print("Generalizability figure caption:")
    print(f"\\caption{{{generalizability_caption(prepared)}}}")
    print("Figures:")
    for stem in (performance_path, generalizability_path):
        for extension in VISUAL.output_formats:
            print(stem.with_suffix(f".{extension}").relative_to(config.dir_root))


if __name__ == "__main__":
    main()
