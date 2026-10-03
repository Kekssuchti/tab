"""Paired comparison of custom training-set retrievers.

Regenerate with:

    uv run python -m src.plotting.retriever_comparison

The figure compares every targeted retriever with the random-subset reference
under the same test-sample and training/model seeds. Intended target and
source/target directions come from ``src.plotting.experiments``; unavailable
registered inputs are reported and skipped.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from mlflow import MlflowClient
from src.config import config
from src.mlflow.evaluation_data import DEFAULT_TRACKING_URI, load_evaluation_data
from src.plotting.defaults import MetricPanelSettings, metric_label, ordered_models, set_plot_style
from src.plotting.experiments import DATA_SOURCES, tasks_for, warn_skipped
from src.plotting.scientific_figstyle import BASELINE, DIVERGING, PALETTE, WIDE, figure_grid, panel_labels, save
from src.plotting.utils import MissingExperimentError
from src.plotting.utils.rendering import instance_plot_styles
from src.plotting.utils.retrieval import RetrieverDesign, read_retriever_design
from src.utils.prediction_tables import RETRIEVER_DATASET


@dataclass(frozen=True)
class DataSettings:
    """Experiment selection and expected fixed design dimensions."""

    experiment_name: str | None = None
    pipeline_runs: tuple[str, ...] | None = None
    # None compares every measured budget/batch size; each combination becomes its
    # own figure so different retrieval problems are never pooled into one mean.
    train_sizes: tuple[int, ...] | None = None
    test_sizes: tuple[int, ...] | None = None
    reference_strategy: str = "random"
    output_dir: Path = config.dir_plots / "retrieval"


@dataclass(frozen=True)
class VisualSettings(MetricPanelSettings):
    """All locally editable presentation choices for this figure."""

    metrics: tuple[str, ...] = ("roc_auc",)
    figure_width: float = WIDE
    # One metric is one grid row.
    row_height: float = 0.55
    heatmap_width_ratio: float = 3.6
    summary_width_ratio: float = 1.8
    heatmap_decimals: int = 1
    heatmap_font_size: float = 5.5
    cohort_colors: tuple[str, ...] = (PALETTE["blue"], PALETTE["orange"], PALETTE["green"])
    cohort_offset: float = 0.20
    cohort_marker_size: float = 3.5
    repeat_range_width: float = 0.9
    grand_mean_marker_size: float = 4.5
    output_formats: tuple[str, ...] = ("pdf",)


@dataclass(frozen=True)
class RetrieverComparison:
    """Paired effects and metadata needed by the figure."""

    effects: pd.DataFrame
    settings: tuple[str, ...]
    setting_labels: dict[str, str]
    model_instances: tuple[str, ...]
    model_labels: dict[str, str]
    test_seeds: tuple[int, ...]
    repeat_count: int
    run_count: int
    model_count: int
    train_size: int
    test_size: int


DATA = DataSettings()
VISUAL = VisualSettings()


# ---------------------------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------------------------


def load_retriever_comparisons(
    experiment_name: str,
    *,
    pipeline_runs: tuple[str, ...] | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    data_settings: DataSettings = DATA,
    visual: VisualSettings = VISUAL,
    expected_target: str | None = None,
    expected_training_source: str | None = None,
    expected_evaluation_center: str | None = None,
) -> tuple[RetrieverComparison, ...]:
    """Load point metrics and form seed-paired changes from random retrieval.

    Every measured selected budget and target-batch size becomes its own
    comparison: pooling them would mix different retrieval problems, and pooling
    several target batches into one mean would treat them as independent scores.
    """
    metrics = load_evaluation_data(
        experiment_name,
        pipeline_runs=pipeline_runs,
        tracking_uri=tracking_uri,
    )
    if metrics.empty:
        raise MissingExperimentError(f"No evaluation data found for experiment {experiment_name!r}")

    points = metrics.loc[
        metrics["scope"].eq("test") & metrics["statistic"].eq("point") & metrics["dataset"].eq(RETRIEVER_DATASET)
    ].copy()
    if points.empty:
        raise ValueError("No retriever point metrics are available")

    missing_metrics = sorted(set(visual.metrics) - set(points.columns))
    if missing_metrics:
        raise ValueError("Missing requested metrics: " + ", ".join(missing_metrics))

    client = MlflowClient(tracking_uri=tracking_uri)
    points["pipeline_mlflow_run_id"] = points["pipeline_mlflow_run_id"].astype(str)
    run_ids = tuple(points["pipeline_mlflow_run_id"].drop_duplicates())
    designs = tuple(read_retriever_design(client, run_id) for run_id in run_ids)
    selected = _select_designs(designs, data_settings)
    if not selected:
        raise ValueError(
            f"No retrieval runs match train_sizes={data_settings.train_sizes} and test_sizes={data_settings.test_sizes}"
        )

    selected_run_ids = {design.run_id for design in selected}
    points = points.loc[points["pipeline_mlflow_run_id"].isin(selected_run_ids)].copy()
    metadata = pd.DataFrame([_design_row(design) for design in selected])
    points = points.merge(metadata, on="pipeline_mlflow_run_id", how="inner", validate="many_to_one")
    _require_design_value(points, "target", expected_target, experiment_name)
    _require_design_value(points, "training_source", expected_training_source, experiment_name)
    _require_design_value(points, "evaluation_center", expected_evaluation_center, experiment_name)

    combinations = sorted({(design.selected_count, design.batch_size) for design in selected})
    return tuple(_prepare_comparison(points, combination, visual, data_settings) for combination in combinations)


def _prepare_comparison(
    points: pd.DataFrame,
    combination: tuple[int, int],
    visual: VisualSettings,
    data_settings: DataSettings,
) -> RetrieverComparison:
    """Pair every targeted retriever with the random reference at one fixed design."""
    train_size, test_size = combination
    rows = points.loc[
        points["retriever_train_size"].eq(train_size) & points["retriever_test_size"].eq(test_size)
    ].copy()
    if rows.empty:
        raise ValueError(f"No retrieval runs are available for train_size={train_size}, test_size={test_size}")
    _validate_design(rows, visual.metrics, data_settings.reference_strategy)

    model_metadata = rows[["model_name", "model_instance"]].drop_duplicates()
    styles = instance_plot_styles(model_metadata)
    model_names = ordered_models(model_metadata["model_name"].astype(str).tolist())
    model_instances = tuple(
        instance
        for model in model_names
        for instance in sorted(model_metadata.loc[model_metadata["model_name"].eq(model), "model_instance"].astype(str))
    )
    model_labels = {instance: _wrap_model_label(styles[instance][1]) for instance in model_instances}

    pair_keys = [
        "model_instance",
        "test_sample_seed",
        "training_sample_seed",
        "model_training_seed",
    ]
    reference = rows.loc[rows["selection_strategy"].eq(data_settings.reference_strategy)].copy()
    reference_columns = [*pair_keys, *visual.metrics]
    reference = reference[reference_columns].rename(
        columns={metric: f"{metric}_reference" for metric in visual.metrics}
    )
    if reference.duplicated(pair_keys).any():
        raise ValueError("Random reference contains duplicate model/seed cells")

    targeted = rows.loc[~rows["selection_strategy"].eq(data_settings.reference_strategy)].copy()
    paired = targeted.merge(reference, on=pair_keys, how="left", validate="many_to_one")
    reference_metric_columns = [f"{metric}_reference" for metric in visual.metrics]
    if paired[reference_metric_columns].isna().any().any():
        raise ValueError("At least one targeted retriever has no seed-matched random reference")

    effect_frames = []
    identity_columns = [
        "setting",
        "setting_label",
        "model_name",
        "model_instance",
        "test_sample_seed",
        "training_sample_seed",
        "model_training_seed",
    ]
    for metric in visual.metrics:
        effect = paired[identity_columns].copy()
        effect["metric"] = metric
        effect["delta"] = visual.score_scale * (
            pd.to_numeric(paired[metric]) - pd.to_numeric(paired[f"{metric}_reference"])
        )
        effect_frames.append(effect)
    effects = pd.concat(effect_frames, ignore_index=True)

    setting_rows = (
        targeted[["setting", "setting_label", "setting_order"]]
        .drop_duplicates()
        .sort_values("setting_order", kind="stable")
    )
    settings = tuple(setting_rows["setting"])
    setting_labels = dict(zip(setting_rows["setting"], setting_rows["setting_label"], strict=True))
    test_seeds = tuple(sorted(effects["test_sample_seed"].astype(int).unique()))
    repeat_cells = effects[["test_sample_seed", "training_sample_seed", "model_training_seed"]].drop_duplicates()
    repeats_per_test = repeat_cells.groupby("test_sample_seed", sort=False).size()
    if repeats_per_test.nunique() != 1:
        raise ValueError(f"Test samples have unequal repeat counts: {repeats_per_test.to_dict()}")

    return RetrieverComparison(
        effects=effects,
        settings=settings,
        setting_labels=setting_labels,
        model_instances=model_instances,
        model_labels=model_labels,
        test_seeds=test_seeds,
        repeat_count=int(repeats_per_test.iloc[0]),
        run_count=len(set(rows["pipeline_mlflow_run_id"].astype(str))),
        model_count=len(model_instances),
        train_size=train_size,
        test_size=test_size,
    )


def _design_row(design: RetrieverDesign) -> dict[str, object]:
    """Return the design metadata one comparison needs, keyed by run."""
    return {
        "pipeline_mlflow_run_id": design.run_id,
        "training_source": design.candidate_source,
        "evaluation_center": design.batch_center,
        "selection_strategy": design.strategy,
        "distance_metric": design.distance_metric,
        "diversity_clusters": design.diversity_clusters,
        "diversity_pool_multiplier": design.diversity_pool_multiplier,
        "retriever_train_size": design.selected_count,
        "retriever_test_size": design.batch_size,
        "test_sample_seed": design.test_sample_seed,
        "training_sample_seed": design.training_sample_seed,
        "model_training_seed": design.model_training_seed,
        "setting": design.setting,
        "setting_label": design.setting_label,
        "setting_order": design.setting_order,
    }


def _select_designs(designs: tuple[RetrieverDesign, ...], data_settings: DataSettings) -> tuple[RetrieverDesign, ...]:
    """Keep the runs whose selected budget and target-batch size were requested."""
    selected = designs
    if data_settings.train_sizes is not None:
        wanted = set(data_settings.train_sizes)
        available = {design.selected_count for design in designs}
        missing = sorted(wanted - available)
        if missing:
            raise ValueError(f"Requested selected budgets {missing} were not measured; available: {sorted(available)}")
        selected = tuple(design for design in selected if design.selected_count in wanted)
    if data_settings.test_sizes is not None:
        wanted = set(data_settings.test_sizes)
        available = {design.batch_size for design in designs}
        missing = sorted(wanted - available)
        if missing:
            raise ValueError(
                f"Requested target-batch sizes {missing} were not measured; available: {sorted(available)}"
            )
        selected = tuple(design for design in selected if design.batch_size in wanted)
    return selected


def _require_design_value(
    rows: pd.DataFrame,
    column: str,
    expected: str | None,
    experiment_name: str,
) -> None:
    if expected is None:
        return
    found = sorted(set(rows[column].dropna().astype(str)))
    if found != [expected]:
        raise ValueError(
            f"Experiment {experiment_name!r} contains {column} {found}, but the central plotting "
            f"registry declares {expected!r}"
        )


def _validate_design(points: pd.DataFrame, metrics: tuple[str, ...], reference_strategy: str) -> None:
    required = {
        "pipeline_mlflow_run_id",
        "model_instance",
        "model_name",
        "selection_strategy",
        "setting",
        "test_sample_seed",
        "training_sample_seed",
        "model_training_seed",
        *metrics,
    }
    missing = sorted(required - set(points.columns))
    if missing:
        raise ValueError("Missing retriever comparison columns: " + ", ".join(missing))
    if not points["selection_strategy"].eq(reference_strategy).any():
        raise ValueError(f"No {reference_strategy!r} reference runs are available")

    point_keys = ["pipeline_mlflow_run_id", "model_instance"]
    if points.duplicated(point_keys).any():
        raise ValueError("Every pipeline run/model must have exactly one retriever point row")

    seeds = ["test_sample_seed", "training_sample_seed", "model_training_seed"]
    expected_cells: set[tuple[object, ...]] | None = None
    expected_models: set[str] | None = None
    for setting, rows in points.groupby("setting", sort=False):
        cells = set(map(tuple, rows[seeds].drop_duplicates().itertuples(index=False, name=None)))
        models = set(rows["model_instance"].astype(str))
        if expected_cells is None:
            expected_cells = cells
            expected_models = models
            continue
        if cells != expected_cells:
            raise ValueError(f"Retriever setting {setting!r} does not share the complete seed grid")
        if models != expected_models:
            raise ValueError(f"Retriever setting {setting!r} does not share the complete model set")


def _wrap_model_label(label: str) -> str:
    replacements = {
        "TabPFNv3.5 Fast": "TabPFN\nFast",
        "TabPFNv3.5": "TabPFN\n3.5",
        "TabICLv2": "TabICL\n2",
        "tabdpt": "TabDPT",
    }
    return replacements.get(label, label)


# ---------------------------------------------------------------------------
# FIGURE
# ---------------------------------------------------------------------------


def make_figure(data: RetrieverComparison, visual: VisualSettings = VISUAL) -> Figure:
    """Draw model-specific mean effects beside the nested seed variation."""
    set_plot_style()
    fig, axes = figure_grid(
        len(visual.metrics),
        2,
        width=visual.figure_width,
        row_height=visual.row_height,
        squeeze=False,
        gridspec_kw={"width_ratios": [visual.heatmap_width_ratio, visual.summary_width_ratio]},
    )
    cohort_colors = _cohort_colors(data.test_seeds, visual)

    for metric_index, metric in enumerate(visual.metrics):
        heatmap_ax = axes[metric_index, 0]
        summary_ax = axes[metric_index, 1]
        summary_ax.sharey(heatmap_ax)

        matrix = _effect_matrix(data, metric)
        repeat_summary = _repeat_summary(data, metric)
        heatmap_limit = _symmetric_limit(matrix.to_numpy())
        summary_limit = _symmetric_limit(repeat_summary["delta"].to_numpy())
        norm = TwoSlopeNorm(vmin=-heatmap_limit, vcenter=0.0, vmax=heatmap_limit)

        image = heatmap_ax.imshow(
            matrix.to_numpy(dtype=float),
            aspect="auto",
            cmap=DIVERGING,
            norm=norm,
            interpolation="nearest",
        )
        _annotate_heatmap(heatmap_ax, matrix.to_numpy(dtype=float), heatmap_limit, visual)
        _format_heatmap(heatmap_ax, data, metric)
        colorbar = fig.colorbar(image, ax=heatmap_ax, fraction=0.045, pad=0.02)
        colorbar.set_label("Mean over seed pairs (pp)")

        _draw_repeat_summary(
            summary_ax,
            repeat_summary,
            data,
            cohort_colors,
            metric,
            summary_limit,
            visual,
        )

    panel_labels(axes)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            linewidth=visual.repeat_range_width,
            markersize=visual.cohort_marker_size,
            color=cohort_colors[seed],
            label=f"Test seed {seed}",
        )
        for seed in data.test_seeds
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markersize=visual.grand_mean_marker_size,
            color="black",
            label="Grand mean",
        )
    )
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles))
    return fig


def _effect_matrix(data: RetrieverComparison, metric: str) -> pd.DataFrame:
    rows = data.effects.loc[data.effects["metric"].eq(metric)]
    matrix = rows.pivot_table(
        index="setting",
        columns="model_instance",
        values="delta",
        aggfunc="mean",
    )
    return matrix.reindex(index=data.settings, columns=data.model_instances)


def _repeat_summary(data: RetrieverComparison, metric: str) -> pd.DataFrame:
    rows = data.effects.loc[data.effects["metric"].eq(metric)].copy()
    summary = rows.groupby(
        [
            "setting",
            "test_sample_seed",
            "training_sample_seed",
            "model_training_seed",
        ],
        sort=False,
        as_index=False,
    )["delta"].mean()
    model_counts = rows.groupby(
        [
            "setting",
            "test_sample_seed",
            "training_sample_seed",
            "model_training_seed",
        ],
        sort=False,
    )["model_instance"].nunique()
    if not model_counts.eq(data.model_count).all():
        raise ValueError("A repeat summary cell does not contain every model")
    return summary


def _symmetric_limit(values: np.ndarray) -> float:
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        raise ValueError("Cannot derive plot limits without finite retriever effects")
    maximum = float(finite.max())
    step = 1.0 if maximum <= 10 else 2.0
    return max(step, step * math.ceil(maximum / step))


def _annotate_heatmap(ax, values: np.ndarray, limit: float, visual: VisualSettings) -> None:
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = float(values[row, column])
            color = "white" if abs(value) >= 0.55 * limit else "black"
            ax.text(
                column,
                row,
                f"{value:+.{visual.heatmap_decimals}f}",
                ha="center",
                va="center",
                color=color,
                fontsize=visual.heatmap_font_size,
            )


def _format_heatmap(ax, data: RetrieverComparison, metric: str) -> None:
    ax.set_xticks(
        np.arange(len(data.model_instances)),
        [data.model_labels[instance] for instance in data.model_instances],
    )
    ax.set_yticks(
        np.arange(len(data.settings)),
        [data.setting_labels[setting] for setting in data.settings],
    )
    ax.set_xlabel("")
    ax.set_ylabel(f"Retriever setting ({metric_label(metric)})")
    ax.tick_params(axis="both", length=0)
    ax.grid(visible=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.4)
        spine.set_color("white")
    if len(data.settings) > 2:
        ax.axhline(1.5, color="white", linewidth=1.2)


def _draw_repeat_summary(
    ax,
    summary: pd.DataFrame,
    data: RetrieverComparison,
    cohort_colors: dict[int, str],
    metric: str,
    limit: float,
    visual: VisualSettings,
) -> None:
    offsets = np.linspace(-visual.cohort_offset, visual.cohort_offset, len(data.test_seeds))
    for row_index, setting in enumerate(data.settings):
        setting_rows = summary.loc[summary["setting"].eq(setting)]
        for offset, test_seed in zip(offsets, data.test_seeds, strict=True):
            values = setting_rows.loc[setting_rows["test_sample_seed"].eq(test_seed), "delta"].to_numpy(dtype=float)
            if len(values) != data.repeat_count:
                raise ValueError(
                    f"Setting {setting!r}, test seed {test_seed} has {len(values)} repeats; "
                    f"expected {data.repeat_count}"
                )
            y = row_index + offset
            color = cohort_colors[test_seed]
            ax.hlines(
                y,
                values.min(),
                values.max(),
                color=color,
                linewidth=visual.repeat_range_width,
                alpha=0.8,
                zorder=2,
            )
            ax.plot(
                values.mean(),
                y,
                marker="o",
                linestyle="none",
                color=color,
                markersize=visual.cohort_marker_size,
                markeredgecolor="white",
                markeredgewidth=0.35,
                zorder=3,
            )
        ax.plot(
            setting_rows["delta"].mean(),
            row_index,
            marker="D",
            linestyle="none",
            color="black",
            markersize=visual.grand_mean_marker_size,
            markeredgecolor="white",
            markeredgewidth=0.4,
            zorder=4,
        )

    ax.axvline(0, color=BASELINE, linestyle="--", linewidth=0.8, zorder=1)
    ax.set_xlim(-limit, limit)
    ax.set_xlabel(f"Mean paired {metric_label(metric)} change\nvs random (pp)")
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)


def _cohort_colors(test_seeds: tuple[int, ...], visual: VisualSettings) -> dict[int, str]:
    if len(test_seeds) > len(visual.cohort_colors):
        cmap = plt.get_cmap("viridis")
        colors = [cmap(value) for value in np.linspace(0.1, 0.9, len(test_seeds))]
    else:
        colors = list(visual.cohort_colors[: len(test_seeds)])
    return dict(zip(test_seeds, colors, strict=True))


# ---------------------------------------------------------------------------
# CAPTION AND EXECUTION
# ---------------------------------------------------------------------------


def figure_caption(data: RetrieverComparison, visual: VisualSettings = VISUAL) -> str:
    setting_means = data.effects.groupby(["metric", "setting"], sort=False)["delta"].mean()
    all_negative = bool(setting_means.lt(0).all())
    claim = (
        "Random training subsets outperform every targeted retriever on average."
        if all_negative
        else "Targeted retrieval changes predictive performance unevenly across models and cohorts."
    )
    metrics = " and ".join(metric_label(metric) for metric in visual.metrics)
    return (
        rf"\textbf{{{claim}}} "
        f"Paired {metrics} changes relative to random selection for {data.model_count} models. "
        f"Heatmap cells average {len(data.test_seeds)} test samples × {data.repeat_count} training/model repeats; "
        "positive values favor the targeted retriever. In the right panels, colored points average models and repeats "
        "within each test sample, thin lines span its repeats, and black diamonds are grand means. No inferential "
        "intervals are shown."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-name",
        default=DATA.experiment_name,
        help="Explicit override for one centrally selected retrieval input.",
    )
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        help="Prediction target to rebuild; repeat for several. Defaults to all intended targets.",
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
    parser.add_argument(
        "--run-id",
        action="append",
        dest="run_ids",
        help="Explicit pipeline MLflow run ID; repeat to select several runs.",
    )
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tasks = tasks_for(
        "retriever_comparison",
        args.targets,
        training_sources=args.training_sources,
        evaluation_centers=args.evaluation_centers,
    )
    run_ids = tuple(args.run_ids) if args.run_ids else DATA.pipeline_runs
    if len(tasks) > 1 and (args.experiment_name or run_ids):
        raise SystemExit(
            "--experiment-name/--run-id pins one retrieval input; select exactly one target, "
            "training source, and evaluation center"
        )

    for task in tasks:
        experiment_name = args.experiment_name or task.experiment_name
        print(f"\n=== {task.label} [{task.direction}] ({experiment_name or 'not registered'})")
        if experiment_name is None:
            warn_skipped(task, "the intended experiment has not been registered yet")
            continue
        try:
            comparisons = load_retriever_comparisons(
                experiment_name,
                pipeline_runs=run_ids,
                tracking_uri=args.tracking_uri,
                expected_target=task.target,
                expected_training_source=task.training_source,
                expected_evaluation_center=task.evaluation_center,
            )
        except MissingExperimentError as missing:
            warn_skipped(task, missing)
            continue

        output_dir = DATA.output_dir / task.target
        output_dir.mkdir(parents=True, exist_ok=True)
        for prepared in comparisons:
            output_stem = output_dir / (
                f"{task.direction_slug}_train-{prepared.train_size}_test-{prepared.test_size}_paired_effects"
            )
            save(make_figure(prepared), str(output_stem), formats=VISUAL.output_formats)
            print(
                f"Selected {prepared.run_count} pipeline runs at train-{prepared.train_size} test-"
                f"{prepared.test_size}: {len(prepared.settings) + 1} retrievers × "
                f"{len(prepared.test_seeds)} test samples × {prepared.repeat_count} repeats"
            )
            print("Figure caption:")
            print(f"\\caption{{{figure_caption(prepared)}}}")
            print("Figures:")
            for extension in VISUAL.output_formats:
                print(output_stem.with_suffix(f".{extension}").relative_to(config.dir_root))


if __name__ == "__main__":
    main()
