"""Full-data baseline performance and generalizability figures.

Regenerate with:

    uv run python -m src.plotting.baseline_transfer

Every prediction task declared for this family in ``src.plotting.experiments``
is rebuilt in turn, each into its own ``plots/baseline/<target>/`` directory.
Edit ``VISUAL`` to change figure presentation; ``--target`` narrows the run to
selected tasks and ``--run-id`` pins explicit pipeline runs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.config import config
from src.plotting.defaults import (
    MetricPanelSettings,
    dataset_label,
    metric_label,
    set_plot_style,
)
from src.plotting.experiments import DATA_SOURCES, tasks_for, warn_skipped
from src.plotting.scientific_figstyle import BASELINE, WIDE, figure_grid, panel_labels, save
from src.plotting.utils import (
    MissingExperimentError,
    aggregate_evaluation_runs,
    load_plot_artifacts,
    prepare_transfer_summary,
)
from src.plotting.utils.artifacts import PlotArtifacts
from src.plotting.utils.rendering import draw_model_forest, instance_plot_styles, interval_axis_limits
from src.plotting.utils.transfer import TransferSummary


@dataclass(frozen=True)
class DataSettings:
    """Run selection and output location."""

    pipeline_runs: tuple[str, ...] | None = None
    full_training_only: bool = True
    # Model names to leave out of every figure, for example a model that a
    # reviewer asked to see separately.
    exclude_models: tuple[str, ...] | None = None
    output_dir: Path = config.dir_plots / "baseline"


@dataclass(frozen=True)
class VisualSettings(MetricPanelSettings):
    """All locally editable presentation choices for these figures."""

    metrics: tuple[str, ...] = ("roc_auc",)
    plot_type: Literal["current", "bootstrap", "min_max"] = "min_max"
    show_ci: bool = True
    ci_level: float = 0.95
    figure_width: float = WIDE
    # One metric is one grid row; the grid adds these heights, so dropping a
    # metric shortens the figure instead of stretching the remaining panels.
    performance_row_height: float = 0.54
    generalizability_row_height: float = 0.54
    repeat_row_height: float = 0.8
    run_marker_size: float = 2.8
    run_offset: float = 0.24
    model_axis_label: str = ""
    performance_axis_template: str = "{metric} on {dataset}"
    delta_spec_axis_template: str = r"$\Delta_{{\mathrm{{spec}}}}$"
    delta_comp_axis_template: str = r"$\Delta_{{\mathrm{{comp}}}}$"
    shade_baselines: bool = True
    baseline_band_alpha: float = 0.13
    marker_size: float = 4.4
    ci_line_width: float = 1.0
    cap_size: float = 2.0
    axis_padding_fraction: float = 0.08
    output_formats: tuple[str, ...] = ("pdf",)

    def __post_init__(self) -> None:
        if self.plot_type not in ("current", "bootstrap", "min_max"):
            raise ValueError(f"Unknown baseline plot_type: {self.plot_type!r}")


DATA = DataSettings()
VISUAL = VisualSettings()


# ---------------------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------------------


def make_performance_figure(
    data: TransferSummary, visual: VisualSettings = VISUAL, *, repeats: tuple[TransferSummary, ...] = ()
) -> Figure:
    """Draw absolute performance on the in-domain and external cohorts."""
    set_plot_style()
    fig, axes = figure_grid(
        len(data.metrics),
        2,
        width=visual.figure_width,
        row_height=visual.repeat_row_height if visual.plot_type == "bootstrap" else visual.performance_row_height,
        sharey=True,
        squeeze=False,
    )
    styles = instance_plot_styles(data.model_metadata)
    datasets = (data.trained_on, data.external_dataset)

    for metric_index, metric in enumerate(data.metrics):
        metric_rows = data.performance.loc[data.performance["metric"].eq(metric)]
        repeat_rows = _repeat_rows(repeats, "performance", metric, visual)
        limits = interval_axis_limits(
            _limit_rows(metric_rows, repeat_rows, visual),
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
                repeat_rows=repeat_rows.loc[repeat_rows["dataset"].eq(dataset)],
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


def make_generalizability_figure(
    data: TransferSummary, visual: VisualSettings = VISUAL, *, repeats: tuple[TransferSummary, ...] = ()
) -> Figure:
    """Draw signed model-specific and comparative generalizability changes."""
    set_plot_style()
    fig, axes = figure_grid(
        len(data.metrics),
        2,
        width=visual.figure_width,
        row_height=visual.repeat_row_height if visual.plot_type == "bootstrap" else visual.generalizability_row_height,
        sharey=False,
        squeeze=False,
    )
    styles = instance_plot_styles(data.model_metadata)

    for metric_index, metric in enumerate(data.metrics):
        delta_spec = data.delta_spec.loc[data.delta_spec["metric"].eq(metric)]
        delta_comp = data.delta_comp.loc[data.delta_comp["metric"].eq(metric)]
        spec_repeats = _repeat_rows(repeats, "delta_spec", metric, visual)
        comp_repeats = _repeat_rows(repeats, "delta_comp", metric, visual)
        limits = interval_axis_limits(
            _limit_rows(
                pd.concat((delta_spec, delta_comp), ignore_index=True),
                pd.concat((spec_repeats, comp_repeats), ignore_index=True),
                visual,
            ),
            scale=visual.score_scale,
            include_zero=True,
            show_ci=visual.show_ci,
            padding_fraction=visual.axis_padding_fraction,
        )
        panels = (
            (delta_spec, spec_repeats, visual.delta_spec_axis_template),
            (delta_comp, comp_repeats, visual.delta_comp_axis_template),
        )
        for contrast_index, (rows, repeat_rows, axis_template) in enumerate(panels):
            ax = axes[metric_index, contrast_index]
            _draw_panel(
                ax,
                rows,
                data,
                styles,
                visual,
                show_model_labels=contrast_index == 0,
                repeat_rows=repeat_rows,
            )
            ax.axvline(0, color=BASELINE, linewidth=0.8, linestyle="--", zorder=1)
            ax.set_xlim(limits)
            ax.set_xlabel(axis_template.format(metric=metric_label(metric)))
            ax.set_ylabel(visual.model_axis_label)
            if contrast_index != 0:
                ax.yaxis.label.set_visible(False)

    panel_labels(axes)
    return fig


def _repeat_rows(repeats, attribute, metric, visual) -> pd.DataFrame:
    if visual.plot_type == "current":
        return pd.DataFrame(columns=["model_instance", "dataset", "estimate", "lower", "upper"])
    if not repeats:
        raise ValueError(f"plot_type={visual.plot_type!r} requires individual run summaries")
    return pd.concat(
        [getattr(run, attribute).assign(repeat=index) for index, run in enumerate(repeats)],
        ignore_index=True,
    ).loc[lambda rows: rows["metric"].eq(metric)]


def _limit_rows(rows, repeat_rows, visual) -> pd.DataFrame:
    if visual.plot_type == "current":
        return rows
    if visual.plot_type == "min_max":
        return repeat_rows.assign(lower=repeat_rows["estimate"], upper=repeat_rows["estimate"])
    return pd.concat((rows.assign(lower=rows["estimate"], upper=rows["estimate"]), repeat_rows))


def _draw_panel(ax, rows, data, styles, visual, *, show_model_labels: bool, repeat_rows) -> None:
    if visual.plot_type == "min_max":
        ranges = repeat_rows.groupby("model_instance")["estimate"].agg(lower="min", upper="max")
        rows = rows.drop(columns=["lower", "upper"]).join(ranges, on="model_instance")
    draw_model_forest(
        ax,
        rows,
        data.model_instances,
        styles,
        scale=visual.score_scale,
        show_ci=visual.show_ci and visual.plot_type != "bootstrap",
        show_model_labels=show_model_labels,
        shade_baselines=visual.shade_baselines,
        baseline_band_alpha=visual.baseline_band_alpha,
        marker_size=visual.marker_size,
        ci_line_width=visual.ci_line_width,
        cap_size=visual.cap_size,
    )
    # The shared model labels belong to the left panel only.
    ax.yaxis.set_visible(show_model_labels)
    if visual.plot_type != "bootstrap":
        return
    offsets = np.linspace(-visual.run_offset, visual.run_offset, len(data.run_ids))
    if len(data.run_ids) % 2:
        offsets[len(data.run_ids) // 2] = visual.run_offset / 2
    for y, instance in enumerate(data.model_instances):
        style, _ = styles[instance]
        for run in repeat_rows.loc[repeat_rows["model_instance"].eq(instance)].itertuples():
            run_y = y + offsets[run.repeat]
            if visual.plot_type == "bootstrap" and visual.show_ci:
                # Draw endpoints directly: percentile bounds need not bracket the point estimate.
                ax.hlines(
                    run_y,
                    visual.score_scale * run.lower,
                    visual.score_scale * run.upper,
                    color=style.color,
                    linewidth=visual.ci_line_width,
                    alpha=0.65,
                    zorder=2,
                )
                ax.plot(
                    visual.score_scale * np.array([run.lower, run.upper]),
                    [run_y, run_y],
                    linestyle="none",
                    marker="|",
                    markersize=2 * visual.cap_size,
                    color=style.color,
                    alpha=0.65,
                    zorder=2,
                )
            ax.plot(
                visual.score_scale * run.estimate,
                run_y,
                marker=style.marker,
                markersize=visual.run_marker_size,
                markerfacecolor="white",
                markeredgecolor=style.color,
                markeredgewidth=0.8,
                linestyle="none",
                zorder=4,
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
        f"{dataset_label(data.trained_on)} and {dataset_label(data.external_dataset)} cohorts. "
        f"{uncertainty} Circles denote classical baselines and triangles tabular foundation "
        r"models."
    )


def generalizability_caption(data: TransferSummary, visual: VisualSettings = VISUAL) -> str:
    references = []
    styles = instance_plot_styles(data.model_metadata)
    for metric in data.metrics:
        rows = data.delta_comp.loc[data.delta_comp["metric"].eq(metric)]
        reference_instance = str(rows["reference_model_instance"].iloc[0])
        references.append(f"{metric_label(metric)}: {styles[reference_instance][1]}")
    uncertainty = _uncertainty_caption(data, visual)
    prevalence_caveat = (
        r" $\Delta_{\mathrm{spec}}$ for AUPRC also reflects differences in outcome prevalence between cohorts."
        if "prc_auc" in data.metrics
        else ""
    )
    return (
        r"\textbf{Signed generalizability changes separate model-specific transfer from comparative performance.} "
        f"Models were trained on {dataset_label(data.trained_on)} and transferred to "
        f"{dataset_label(data.external_dataset)}. Left panels show "
        r"$\Delta_{\mathrm{spec}}$ (target minus source performance); right panels show "
        r"$\Delta_{\mathrm{comp}}$ (target performance minus the strongest transferred model: "
        f"{'; '.join(references)}). Zero denotes no model-specific change or the comparative reference; more "
        f"negative values indicate worse generalizability. The comparative reference is selected from mean "
        f"external performance and held fixed across repeats. "
        f"{uncertainty}{prevalence_caveat}"
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
    if visual.plot_type == "min_max":
        points = f"Filled markers show mean estimates across {run_text}."
        if not visual.show_ci:
            return f"{points} No intervals are shown."
        return f"{points} Whiskers span the minimum and maximum run estimates, not confidence intervals."
    if visual.plot_type != "current":
        points = f"Large filled markers show the mean; small hollow markers show each of {run_text}."
        if not visual.show_ci:
            return f"{points} No intervals are shown."
        confidence = round(100 * visual.ci_level)
        return (
            rf"{points} Each run's whiskers show its {confidence}\% percentile interval from "
            f"{data.bootstrap_count:,} cohort-bootstrap draws; the mean has no whisker."
        )
    if not visual.show_ci:
        return f"Points are run-averaged estimates. No uncertainty intervals are shown; estimates summarize {run_text}."
    confidence = round(100 * visual.ci_level)
    aggregation_text = (
        "for one pipeline run" if data.run_count == 1 else f"after averaging {data.run_count} repeated pipeline runs"
    )
    return (
        rf"Points are run-averaged estimates. Whiskers are {confidence}\% percentile intervals from "
        f"{data.bootstrap_count:,} aligned cohort-bootstrap "
        f"draws {aggregation_text}."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override DATA.output_dir, for example plots/baseline/versions.",
    )
    parser.add_argument(
        "--plot-type",
        choices=("current", "bootstrap", "min_max", "all"),
        help="Override VISUAL.plot_type; all writes all three versions with distinct filenames.",
    )
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        help=(
            "Prediction task to rebuild, for example mortality; repeat for several. "
            "Defaults to every task this family declares."
        ),
    )
    parser.add_argument(
        "--run-id",
        action="append",
        dest="run_ids",
        help=(
            "Explicit pipeline MLflow run ID; repeat to average runs. Every explicit run must record one "
            "dataset.train_on entry with fraction 1.0."
        ),
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
    tasks = tasks_for("baseline", args.targets, training_sources=args.training_sources)
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
                exclude_models=DATA.exclude_models,
                full_training_only=DATA.full_training_only,
                expected_target=task.target,
                expected_training_source=task.training_source,
            )
        except MissingExperimentError as missing:
            warn_skipped(task, missing)
            continue

        visual = VISUAL
        aggregated = aggregate_evaluation_runs(artifacts, metrics=visual.metrics, ci_level=visual.ci_level)
        prepared = prepare_transfer_summary(aggregated)
        print("Selected pipeline runs: " + ", ".join(prepared.run_ids))
        plot_types = (
            ("current", "bootstrap", "min_max") if args.plot_type == "all" else (args.plot_type or visual.plot_type,)
        )
        repeats = ()
        if any(kind != "current" for kind in plot_types):
            references = prepared.delta_comp.set_index("metric")["reference_model_instance"].to_dict()
            repeat_summaries = []
            for run_id in artifacts.run_ids:
                single = PlotArtifacts(
                    metrics=artifacts.metrics.loc[artifacts.metrics["pipeline_mlflow_run_id"].eq(run_id)],
                    bootstrap_scores=artifacts.bootstrap_scores.loc[
                        artifacts.bootstrap_scores["pipeline_mlflow_run_id"].eq(run_id)
                    ],
                    experiment_name=artifacts.experiment_name,
                    run_ids=(run_id,),
                )
                repeat_summaries.append(
                    prepare_transfer_summary(
                        aggregate_evaluation_runs(single, metrics=visual.metrics, ci_level=visual.ci_level),
                        reference_by_metric=references,
                    )
                )
            repeats = tuple(repeat_summaries)

        output_dir = (args.output_dir or DATA.output_dir) / prepared.target
        output_dir.mkdir(parents=True, exist_ok=True)
        for kind in plot_types:
            selected_visual = replace(visual, plot_type=kind)
            suffix = ""  # if kind == "current" else f"_{kind}"
            prefix = prepared.trained_on
            paths = (
                output_dir / f"{prefix}_performance{suffix}",
                output_dir / f"{prefix}_generalizability{suffix}",
            )
            figures = (
                make_performance_figure(prepared, selected_visual, repeats=repeats),
                make_generalizability_figure(prepared, selected_visual, repeats=repeats),
            )
            for fig, path in zip(figures, paths, strict=True):
                save(fig, str(path), formats=selected_visual.output_formats)
            print(f"Performance figure caption ({kind}):")
            print(f"\\caption{{{performance_caption(prepared, selected_visual)}}}")
            print(f"Generalizability figure caption ({kind}):")
            print(f"\\caption{{{generalizability_caption(prepared, selected_visual)}}}")
            print("Figures:")
            for stem in paths:
                for extension in selected_visual.output_formats:
                    print(stem.with_suffix(f".{extension}"))


if __name__ == "__main__":
    main()
