"""Estimator-count ablation performance and runtime figures.

Regenerate with:

    uv run python -m src.plotting.ablation_estimators

Edit ``DATA`` for experiment selection and ``VISUAL`` for presentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.config import config
from src.plotting.defaults import dataset_label, metric_label, set_plot_style
from src.plotting.scientific_figstyle import WIDE, figure, save
from src.plotting.utils import aggregate_runs_by_setting, load_plot_artifacts
from src.plotting.utils.grouped import GroupedEvaluation
from src.plotting.utils.rendering import instance_plot_styles
from src.utils.prediction_metrics import pairwise_win_matrices

# ---------------------------------------------------------------------------
# EDIT THESE SETTINGS
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataSettings:
    """Experiment selection and output location."""

    experiment_name: str = "ablation_tudd_n_estimators"
    setting_pattern: str = r"grid-default-(\d+)-roc-auc"
    dataset: str = "mimic"
    metric: str = "roc_auc"
    runtime_metric: str = "total_time"
    # Model names to leave out of every figure.
    exclude_models: tuple[str, ...] | None = None
    output_dir: Path = config.dir_plots / "ablation_n_estimators"


@dataclass(frozen=True)
class VisualSettings:
    """All locally editable presentation choices for these figures."""

    show_ci: bool = True
    ci_level: float = 0.95
    figure_width: float = WIDE
    performance_height_ratio: float = 0.62
    runtime_height_ratio: float = 0.62
    score_scale: float = 100.0
    marker_size: float = 4.0
    line_width: float = 1.2
    ci_line_width: float = 0.8
    cap_size: float = 2.0
    log_estimator_axis: bool = True
    log_runtime_axis: bool = True
    performance_axis_label: str = "{metric} on {dataset} (%)"
    runtime_axis_label: str = "Total runtime (seconds)"
    estimator_axis_label: str = "Number of estimators"
    legend_columns: int = 3
    print_pairwise_wins: bool = True
    output_formats: tuple[str, ...] = ("pdf",)


DATA = DataSettings()
VISUAL = VisualSettings()


# ---------------------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------------------


def make_performance_figure(data: GroupedEvaluation, visual: VisualSettings = VISUAL) -> Figure:
    """Draw model performance over estimator count."""
    set_plot_style()
    fig, ax = figure(width=visual.figure_width, ratio=visual.performance_height_ratio)
    styles = instance_plot_styles(data.model_metadata)
    rows = data.performance.loc[
        data.performance["dataset"].eq(DATA.dataset) & data.performance["metric"].eq(DATA.metric)
    ].copy()
    rows["setting_value"] = pd.to_numeric(rows["setting"])

    for instance in data.model_instances:
        model_rows = rows.loc[rows["model_instance"].eq(instance)].sort_values("setting_value")
        if model_rows.empty:
            continue
        style, label = styles[instance]
        estimates = visual.score_scale * model_rows["estimate"].to_numpy(dtype=float)
        lower = visual.score_scale * model_rows["lower"].to_numpy(dtype=float)
        upper = visual.score_scale * model_rows["upper"].to_numpy(dtype=float)
        yerr = np.vstack((estimates - lower, upper - estimates)) if visual.show_ci else None
        ax.errorbar(
            model_rows["setting_value"],
            estimates,
            yerr=yerr,
            color=style.color,
            marker=style.marker,
            linestyle=style.linestyle,
            markersize=visual.marker_size,
            linewidth=visual.line_width,
            elinewidth=visual.ci_line_width,
            capsize=visual.cap_size if visual.show_ci else 0,
            label=label,
        )

    _format_estimator_axis(ax, data.settings, visual)
    ax.set_ylabel(
        visual.performance_axis_label.format(
            metric=metric_label(DATA.metric),
            dataset=dataset_label(DATA.dataset),
        )
    )
    ax.legend(ncol=visual.legend_columns, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    return fig


def make_runtime_figure(data: GroupedEvaluation, visual: VisualSettings = VISUAL) -> Figure:
    """Draw total runtime over estimator count."""
    set_plot_style()
    fig, ax = figure(width=visual.figure_width, ratio=visual.runtime_height_ratio)
    styles = instance_plot_styles(data.model_metadata)
    rows = data.runtimes.copy()
    rows["setting_value"] = pd.to_numeric(rows["setting"])

    for instance in data.model_instances:
        model_rows = rows.loc[rows["model_instance"].eq(instance)].sort_values("setting_value")
        if model_rows.empty:
            continue
        style, label = styles[instance]
        ax.plot(
            model_rows["setting_value"],
            model_rows[DATA.runtime_metric],
            color=style.color,
            marker=style.marker,
            linestyle=style.linestyle,
            markersize=visual.marker_size,
            linewidth=visual.line_width,
            label=label,
        )

    _format_estimator_axis(ax, data.settings, visual)
    if visual.log_runtime_axis:
        ax.set_yscale("log")
    ax.set_ylabel(visual.runtime_axis_label)
    ax.legend(ncol=visual.legend_columns, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    return fig


def _format_estimator_axis(ax, settings: tuple[str, ...], visual: VisualSettings) -> None:
    values = np.array([float(setting) for setting in settings])
    if visual.log_estimator_axis:
        ax.set_xscale("log", base=2)
    ax.set_xticks(values, [f"{value:g}" for value in values])
    ax.set_xlabel(visual.estimator_axis_label)
    ax.grid(which="both", axis="both")


# ---------------------------------------------------------------------------
# DATA SELECTION, CAPTIONS, AND EXECUTION
# ---------------------------------------------------------------------------


def _setting_map(metrics: pd.DataFrame) -> tuple[dict[str, str], tuple[str, ...]]:
    runs = metrics[["pipeline_mlflow_run_id", "pipeline_run_name"]].drop_duplicates().copy()
    runs["setting"] = runs["pipeline_run_name"].str.extract(DATA.setting_pattern, expand=False)
    if runs["setting"].isna().any():
        invalid = runs.loc[runs["setting"].isna(), "pipeline_run_name"].tolist()
        raise ValueError(f"Could not infer estimator count from run names: {invalid}")
    runs["setting_value"] = pd.to_numeric(runs["setting"])
    runs = runs.sort_values("setting_value", kind="stable")
    setting_by_run = dict(zip(runs["pipeline_mlflow_run_id"].astype(str), runs["setting"].astype(str), strict=True))
    setting_order = tuple(runs["setting"].astype(str).drop_duplicates())
    return setting_by_run, setting_order


def _print_pairwise_wins(data: GroupedEvaluation) -> None:
    if not VISUAL.print_pairwise_wins:
        return
    selected = data.bootstrap_scores.loc[
        data.bootstrap_scores["dataset"].eq(DATA.dataset) & data.bootstrap_scores["metric"].eq(DATA.metric)
    ]
    for instance in data.model_instances:
        rows = selected.loc[selected["model_instance"].eq(instance)]
        if rows.empty:
            continue
        scores = rows.pivot(index="bootstrap_id", columns="setting", values="score").reindex(columns=data.settings)
        matrix_input = scores.reset_index()
        matrix_input.insert(0, "metric", DATA.metric)
        matrix_input.insert(0, "dataset", DATA.dataset)
        matrix = pairwise_win_matrices(matrix_input)[f"{DATA.dataset}_{DATA.metric}"]
        matrix.index.name = "row n_estimators"
        matrix.columns.name = "column n_estimators"
        print(f"\n{styles_label(data, instance)} — paired bootstrap wins")
        print(matrix.to_string())


def styles_label(data: GroupedEvaluation, instance: str) -> str:
    return instance_plot_styles(data.model_metadata)[instance][1]


def performance_caption(data: GroupedEvaluation, visual: VisualSettings = VISUAL) -> str:
    uncertainty = (
        rf"whiskers are {round(100 * visual.ci_level)}\% percentile intervals from aligned cohort-bootstrap draws"
        if visual.show_ci
        else "no uncertainty intervals are shown"
    )
    return (
        r"\textbf{Estimator count changes model performance unevenly.} "
        f"{metric_label(DATA.metric)} on {dataset_label(DATA.dataset)} over the configured estimator counts; "
        f"points average repeated runs within each count and {uncertainty}."
    )


def runtime_caption() -> str:
    return (
        r"\textbf{Runtime rises with estimator count but differs substantially between models.} "
        r"Total measured pipeline runtime over the configured estimator counts; repeated runs are averaged."
    )


def main() -> None:
    artifacts = load_plot_artifacts(DATA.experiment_name, exclude_models=DATA.exclude_models)
    setting_by_run, setting_order = _setting_map(artifacts.metrics)
    prepared = aggregate_runs_by_setting(
        artifacts,
        setting_by_run,
        setting_order=setting_order,
        metrics=(DATA.metric,),
        runtime_columns=(DATA.runtime_metric,),
        ci_level=VISUAL.ci_level,
    )

    DATA.output_dir.mkdir(parents=True, exist_ok=True)
    performance_stem = DATA.output_dir / f"{DATA.dataset}_performance"
    runtime_stem = DATA.output_dir / f"{DATA.dataset}_runtime"
    save(make_performance_figure(prepared), str(performance_stem), formats=VISUAL.output_formats)
    save(make_runtime_figure(prepared), str(runtime_stem), formats=VISUAL.output_formats)
    _print_pairwise_wins(prepared)

    print("\nPerformance figure caption:")
    print(f"\\caption{{{performance_caption(prepared)}}}")
    print("Runtime figure caption:")
    print(f"\\caption{{{runtime_caption()}}}")
    print("Figures:")
    for stem in (performance_stem, runtime_stem):
        for extension in VISUAL.output_formats:
            print(stem.with_suffix(f".{extension}").relative_to(config.dir_root))


if __name__ == "__main__":
    main()
