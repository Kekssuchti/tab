"""Paired comparison of custom training-set retrievers.

Regenerate with:

    uv run python -m src.plotting.retriever_comparison

The figure compares every targeted retriever with the random-subset reference
under the same test-sample and training/model seeds. Edit ``DATA`` for
experiment selection and ``VISUAL`` for presentation.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from mlflow import MlflowClient
from src.config import config
from src.mlflow.evaluation_data import DEFAULT_TRACKING_URI, load_evaluation_data
from src.mlflow.tracking_contract import ARTIFACT_CONFIG
from src.plotting.defaults import metric_label, ordered_models, set_plot_style
from src.plotting.scientific_figstyle import BASELINE, DIVERGING, PALETTE, WIDE, figure_grid, panel_labels, save
from src.plotting.utils.rendering import instance_plot_styles


@dataclass(frozen=True)
class DataSettings:
    """Experiment selection and expected fixed design dimensions."""

    experiment_name: str = "retriever_tudd_mortality"
    pipeline_runs: tuple[str, ...] | None = None
    train_size: int | None = 1600
    test_size: int | None = 100
    reference_strategy: str = "random"
    output_dir: Path = config.dir_plots / "retriever"


@dataclass(frozen=True)
class VisualSettings:
    """All locally editable presentation choices for this figure."""

    metrics: tuple[str, ...] = ("roc_auc", "prc_auc")
    figure_width: float = WIDE
    figure_height_ratio: float = 1.10
    heatmap_width_ratio: float = 3.6
    summary_width_ratio: float = 1.8
    score_scale: float = 100.0
    heatmap_decimals: int = 1
    heatmap_font_size: float = 5.5
    cohort_colors: tuple[str, ...] = (PALETTE["blue"], PALETTE["orange"], PALETTE["green"])
    cohort_offset: float = 0.20
    cohort_marker_size: float = 3.5
    repeat_range_width: float = 0.9
    grand_mean_marker_size: float = 4.5
    output_formats: tuple[str, ...] = ("pdf", "svg")


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


DATA = DataSettings()
VISUAL = VisualSettings()


# ---------------------------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------------------------


def load_retriever_comparison(
    experiment_name: str,
    *,
    pipeline_runs: tuple[str, ...] | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    data_settings: DataSettings = DATA,
    visual: VisualSettings = VISUAL,
) -> RetrieverComparison:
    """Load point metrics and form seed-paired changes from random retrieval."""
    metrics = load_evaluation_data(
        experiment_name,
        pipeline_runs=pipeline_runs,
        tracking_uri=tracking_uri,
    )
    if metrics.empty:
        raise ValueError(f"No evaluation data found for experiment {experiment_name!r}")

    points = metrics.loc[
        metrics["scope"].eq("test")
        & metrics["statistic"].eq("point")
        & metrics["dataset"].eq("retriever")
    ].copy()
    if points.empty:
        raise ValueError("No retriever point metrics are available")

    missing_metrics = sorted(set(visual.metrics) - set(points.columns))
    if missing_metrics:
        raise ValueError("Missing requested metrics: " + ", ".join(missing_metrics))

    client = MlflowClient(tracking_uri=tracking_uri)
    run_ids = tuple(points["pipeline_mlflow_run_id"].astype(str).drop_duplicates())
    run_metadata = pd.DataFrame([_read_run_design(client, run_id) for run_id in run_ids])
    run_metadata = _select_design(run_metadata, data_settings)
    selected_run_ids = set(run_metadata["pipeline_mlflow_run_id"])
    points = points.loc[points["pipeline_mlflow_run_id"].astype(str).isin(selected_run_ids)].copy()
    if points.empty:
        raise ValueError("No runs match the configured retriever train/test sizes")

    points["pipeline_mlflow_run_id"] = points["pipeline_mlflow_run_id"].astype(str)
    points = points.merge(
        run_metadata,
        on="pipeline_mlflow_run_id",
        how="inner",
        validate="many_to_one",
    )
    _validate_design(points, visual.metrics, data_settings.reference_strategy)

    model_metadata = points[["model_name", "model_instance"]].drop_duplicates()
    styles = instance_plot_styles(model_metadata)
    model_names = ordered_models(model_metadata["model_name"].astype(str).tolist())
    model_instances = tuple(
        instance
        for model in model_names
        for instance in sorted(
            model_metadata.loc[model_metadata["model_name"].eq(model), "model_instance"].astype(str)
        )
    )
    model_labels = {instance: _wrap_model_label(styles[instance][1]) for instance in model_instances}

    pair_keys = [
        "model_instance",
        "test_sample_seed",
        "training_sample_seed",
        "model_training_seed",
    ]
    reference = points.loc[points["selection_strategy"].eq(data_settings.reference_strategy)].copy()
    reference_columns = [*pair_keys, *visual.metrics]
    reference = reference[reference_columns].rename(
        columns={metric: f"{metric}_reference" for metric in visual.metrics}
    )
    if reference.duplicated(pair_keys).any():
        raise ValueError("Random reference contains duplicate model/seed cells")

    targeted = points.loc[~points["selection_strategy"].eq(data_settings.reference_strategy)].copy()
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
        run_count=len(selected_run_ids),
        model_count=len(model_instances),
    )


def _read_run_design(client: MlflowClient, run_id: str) -> dict[str, object]:
    artifact_uri = client.get_run(run_id).info.artifact_uri
    config_path = _local_artifact_path(artifact_uri, ARTIFACT_CONFIG)
    if config_path is None:
        config_path = Path(client.download_artifacts(run_id, ARTIFACT_CONFIG))
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    retriever = payload["dataset"]["custom_retriever"]
    random_states = payload["random_states"]
    test_on = retriever["test_on"]
    if len(test_on) != 1:
        raise ValueError(f"Run {run_id} must define exactly one retriever test cohort")

    strategy = str(retriever["selection_strategy"])
    metric = str(retriever.get("distance_metric", "euclidean"))
    clusters = int(retriever.get("diversity_clusters", 0))
    pool = float(retriever.get("diversity_pool_multiplier", 0.0))
    setting, setting_label, setting_order = _setting_identity(strategy, metric, clusters, pool)
    return {
        "pipeline_mlflow_run_id": run_id,
        "selection_strategy": strategy,
        "distance_metric": metric,
        "diversity_clusters": clusters,
        "diversity_pool_multiplier": pool,
        "retriever_train_size": int(retriever["train_size"]),
        "retriever_test_size": int(test_on[0]["fraction"]),
        "test_sample_seed": int(retriever["test_sample_seed"]),
        "training_sample_seed": int(random_states["training_sample_seed"]),
        "model_training_seed": int(random_states["model_training_seed"]),
        "setting": setting,
        "setting_label": setting_label,
        "setting_order": setting_order,
    }


def _local_artifact_path(artifact_uri: str, artifact_name: str) -> Path | None:
    parsed = urlparse(artifact_uri)
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path)) / artifact_name
    elif parsed.scheme == "":
        path = Path(artifact_uri) / artifact_name
    else:
        return None
    return path if path.is_file() else None


def _select_design(metadata: pd.DataFrame, data_settings: DataSettings) -> pd.DataFrame:
    selected = metadata
    if data_settings.train_size is not None:
        selected = selected.loc[selected["retriever_train_size"].eq(data_settings.train_size)]
    if data_settings.test_size is not None:
        selected = selected.loc[selected["retriever_test_size"].eq(data_settings.test_size)]
    if selected.empty:
        raise ValueError(
            "No run configs match "
            f"train_size={data_settings.train_size}, test_size={data_settings.test_size}"
        )
    return selected.copy()


def _setting_identity(
    strategy: str,
    distance_metric: str,
    clusters: int,
    pool_multiplier: float,
) -> tuple[str, str, tuple[object, ...]]:
    if strategy == "random":
        return "random", "Random", (-1,)
    if strategy == "knn":
        setting = f"knn:{distance_metric}"
        label = f"KNN: {distance_metric.title()}"
        metric_order = {"euclidean": 0, "manhattan": 1}.get(distance_metric, 99)
        return setting, label, (0, metric_order, distance_metric)
    if strategy == "knn-diverse":
        pool_label = f"{pool_multiplier:g}"
        setting = f"knn-diverse:{clusters}:{pool_label}"
        label = f"Diverse: k={clusters}, pool={pool_label}×"
        return setting, label, (1, clusters, pool_multiplier)
    raise ValueError(f"Unsupported retriever selection_strategy {strategy!r}")


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
        ratio=visual.figure_height_ratio,
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
    ax.set_xlabel("Model")
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
            values = setting_rows.loc[
                setting_rows["test_sample_seed"].eq(test_seed), "delta"
            ].to_numpy(dtype=float)
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
    parser.add_argument("--experiment-name", default=DATA.experiment_name)
    parser.add_argument(
        "--run-id",
        action="append",
        dest="run_ids",
        help="Explicit pipeline MLflow run ID; repeat to select several runs.",
    )
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--output-dir", type=Path, default=DATA.output_dir)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_ids = tuple(args.run_ids) if args.run_ids else DATA.pipeline_runs
    prepared = load_retriever_comparison(
        args.experiment_name,
        pipeline_runs=run_ids,
        tracking_uri=args.tracking_uri,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_dir / "retriever_paired_effects"
    save(make_figure(prepared), str(output_stem), formats=VISUAL.output_formats)

    print(
        f"Selected {prepared.run_count} pipeline runs: {len(prepared.settings) + 1} retrievers × "
        f"{len(prepared.test_seeds)} test samples × {prepared.repeat_count} repeats"
    )
    print("Figure caption:")
    print(f"\\caption{{{figure_caption(prepared)}}}")
    print("Figures:")
    for extension in VISUAL.output_formats:
        print(output_stem.with_suffix(f".{extension}").relative_to(config.dir_root))


if __name__ == "__main__":
    main()
