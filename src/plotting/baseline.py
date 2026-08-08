"""Reusable plots for baseline model evaluation and generalization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from src.plotting.defaults import metric_label, metric_lower_is_better
from src.plotting.plot_support import (
    draw_confidence_intervals,
    instance_plot_styles,
    runtime_label,
    test_point_scores,
    with_model_labels,
)

_MODEL_COLUMNS = [
    "pipeline_mlflow_run_id",
    "pipeline_run_name",
    "model_mlflow_run_id",
    "model_name",
    "model_instance",
    "target",
    "task_type",
    "trained_on",
]


def calculate_comparative_generalizability(
    results: pd.DataFrame,
    *,
    metric: str = "roc_auc",
    external_dataset: str | None = None,
) -> pd.DataFrame:
    """Calculate external-test loss and rank across the selected models."""
    scores = test_point_scores(results, metric)
    if external_dataset is None:
        external = scores.loc[scores["dataset"].ne(scores["trained_on"])].copy()
    else:
        external = scores.loc[scores["dataset"].eq(external_dataset)].copy()

    external = external.rename(
        columns={
            "dataset": "external_dataset",
            "value": "external_score",
            "ci_lower": "external_ci_lower",
            "ci_upper": "external_ci_upper",
        }
    )
    training = scores.loc[
        scores["dataset"].eq(scores["trained_on"]),
        ["model_mlflow_run_id", "value"],
    ].rename(columns={"value": "training_score"})
    external = external.merge(training, on="model_mlflow_run_id")
    lower_is_better = metric_lower_is_better(metric)
    groups = ["target", "task_type", "external_dataset"]
    if lower_is_better:
        external["generalizability_loss"] = external["training_score"] - external["external_score"]
        external["best_external_score"] = external.groupby(groups, dropna=False)["external_score"].transform("min")
        external["comparative_generalizability_loss"] = external["best_external_score"] - external["external_score"]
    else:
        external["generalizability_loss"] = external["external_score"] - external["training_score"]
        external["best_external_score"] = external.groupby(groups, dropna=False)["external_score"].transform("max")
        external["comparative_generalizability_loss"] = external["external_score"] - external["best_external_score"]
    external["generalization_rank"] = (
        external.groupby(groups, dropna=False)["external_score"]
        .rank(method="min", ascending=lower_is_better)
        .astype(int)
    )
    external["model_specific_generalization_rank"] = (
        external.groupby(groups, dropna=False)["generalizability_loss"].rank(method="min", ascending=False).astype(int)
    )
    return external[
        _MODEL_COLUMNS
        + [
            "external_dataset",
            "metric",
            "external_score",
            "external_ci_lower",
            "external_ci_upper",
            "training_score",
            "generalizability_loss",
            "best_external_score",
            "comparative_generalizability_loss",
            "generalization_rank",
            "model_specific_generalization_rank",
        ]
    ].sort_values(["target", "external_dataset", "generalization_rank", "model_instance"])


def plot_score_dumbbell(
    results: pd.DataFrame,
    *,
    metric: str = "roc_auc",
    datasets: tuple[str, str] = ("mimic", "tudd"),
    ignore_models: Sequence[str] | None = None,
    include_models: Sequence[str] | None = None,
    show_ci: bool = True,
    title: str | None = None,
    x_limits: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Compare one held-out metric between two test centers."""
    first, second = datasets
    scores = _filter_models(test_point_scores(results, metric), include_models, ignore_models)
    values = []
    for dataset in datasets:
        center = scores.loc[scores["dataset"].eq(dataset)].copy()
        center = center.rename(
            columns={
                "value": f"{dataset}_value",
                "ci_lower": f"{dataset}_lower",
                "ci_upper": f"{dataset}_upper",
            }
        )
        values.append(center[_MODEL_COLUMNS + [f"{dataset}_value", f"{dataset}_lower", f"{dataset}_upper"]])
    paired = values[0].merge(values[1], on=_MODEL_COLUMNS)
    paired["difference"] = paired[f"{first}_value"] - paired[f"{second}_value"]
    paired = with_model_labels(paired).sort_values("difference")

    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(4, len(paired) * 0.42)))
    positions = np.arange(len(paired))
    paired["_y"] = positions
    ax.hlines(
        positions,
        paired[f"{first}_value"],
        paired[f"{second}_value"],
        color="#A7A7A7",
        linewidth=2,
        zorder=1,
    )

    markers = ["o", "s", "^", "D"]
    marker_by_dataset = {dataset: markers[index % len(markers)] for index, dataset in enumerate(datasets)}
    styles = instance_plot_styles(paired)
    for dataset in datasets:
        for instance, (style, _) in styles.items():
            rows = paired.loc[paired["model_instance"].eq(instance)]
            if rows.empty:
                continue
            ax.scatter(
                rows[f"{dataset}_value"],
                rows["_y"],
                color=style.color,
                marker=marker_by_dataset[dataset],
                s=42,
                zorder=3,
            )
            if show_ci:
                draw_confidence_intervals(
                    ax,
                    rows["_y"],
                    rows[f"{dataset}_value"],
                    rows[f"{dataset}_lower"],
                    rows[f"{dataset}_upper"],
                    style.color,
                    horizontal=True,
                )

    label = metric_label(metric)
    if x_limits is None:
        score_bounds = pd.concat(
            [paired[f"{dataset}_{bound}"] for dataset in datasets for bound in ("value", "lower", "upper")]
        ).dropna()
        x_limits = (max(0.0, score_bounds.min() - 0.03), min(1.0, score_bounds.max() + 0.03))
    ax.set_yticks(positions, paired["model_label"])
    ax.invert_yaxis()
    ax.set(
        xlabel=label,
        ylabel="",
        title=title if title is not None else f"{label} by test center",
        xlim=x_limits,
    )
    ax.grid(axis="x", alpha=0.2)
    legend_handles = [
        Line2D(
            [],
            [],
            marker=marker_by_dataset[dataset],
            color="#555555",
            linestyle="none",
            markersize=7,
            label=dataset.upper(),
        )
        for dataset in datasets
    ]
    ax.legend(handles=legend_handles, title="Test center")
    return ax


def plot_roc_auc(
    results: pd.DataFrame,
    *,
    datasets: tuple[str, str] = ("mimic", "tudd"),
    ignore_models: Sequence[str] | None = None,
    include_models: Sequence[str] | None = None,
    show_ci: bool = True,
    title: str | None = None,
    x_limits: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot a ROC-AUC dumbbell comparison between two test datasets."""
    return plot_score_dumbbell(
        results,
        metric="roc_auc",
        datasets=datasets,
        ignore_models=ignore_models,
        include_models=include_models,
        show_ci=show_ci,
        title=title,
        x_limits=x_limits,
        ax=ax,
    )


def plot_generalization_gaps(
    results: pd.DataFrame,
    *,
    metric: str = "roc_auc",
    external_dataset: str | None = None,
    loss: Literal["model_specific", "comparative"] = "comparative",
    ignore_models: Sequence[str] | None = None,
    include_models: Sequence[str] | None = None,
    title: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot model-specific or comparative external-test loss."""
    selected = _filter_models(results, include_models, ignore_models)
    gaps = calculate_comparative_generalizability(selected, metric=metric, external_dataset=external_dataset)
    loss_column, rank_column, loss_label = {
        "model_specific": (
            "generalizability_loss",
            "model_specific_generalization_rank",
            "Model-specific generalizability loss",
        ),
        "comparative": (
            "comparative_generalizability_loss",
            "generalization_rank",
            "Comparative generalizability loss",
        ),
    }[loss]
    gaps = with_model_labels(gaps).sort_values([rank_column, "model_label"])

    if ax is None:
        _, ax = plt.subplots(figsize=(11, max(4, len(gaps) * 0.48)))
    positions = np.arange(len(gaps))
    styles = instance_plot_styles(gaps)
    bar_colors = [styles[instance][0].color for instance in gaps["model_instance"]]
    ax.barh(positions, gaps[loss_column], color=bar_colors)
    for position, row in zip(positions, gaps.itertuples(), strict=True):
        value = getattr(row, loss_column)
        ax.annotate(
            f"#{getattr(row, rank_column)}",
            (value, float(position)),
            xytext=(-5, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    label = metric_label(metric)
    centers = ", ".join(dataset.upper() for dataset in gaps["external_dataset"].unique())
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(positions, gaps["model_label"])
    ax.invert_yaxis()
    ax.set(
        xlabel=f"{label} difference (closer to zero is better)",
        ylabel="",
        title=title if title is not None else f"{label} {loss_label} (external: {centers})",
    )
    ax.grid(axis="x", alpha=0.2)
    return ax


def plot_performance_vs_runtime(
    data: pd.DataFrame,
    ignore_models: Sequence[str] | None = None,
    *,
    include_models: Sequence[str] | None = None,
    metric: str = "roc_auc",
    test_dataset: str | None = "tudd",
    runtime_scope: str = "model",
    runtime_metric: str = "total_time",
    log_x: bool = True,
    show_ci: bool = True,
    title: str | None = None,
    ax: Axes | None = None,
    aggregate: bool = False,
    invert_x: bool = False,
) -> Axes:
    """Plot selected test performance against runtime, with score ranks."""
    data = _filter_models(data, include_models, ignore_models)
    scores = test_point_scores(data, metric)
    scores = scores.loc[scores["dataset"].eq(test_dataset)].copy()
    scores = scores.rename(
        columns={
            "dataset": "external_dataset",
            "value": "external_score",
            "ci_lower": "external_ci_lower",
            "ci_upper": "external_ci_upper",
        }
    )
    rank_groups = [column for column in ("target", "task_type", "external_dataset") if column in scores]
    scores["generalization_rank"] = (
        scores.groupby(rank_groups, dropna=False)["external_score"]
        .rank(method="min", ascending=metric_lower_is_better(metric))
        .astype(int)
    )
    timings = (
        data.loc[
            data["scope"].eq("test") & data[runtime_metric].notna(),
            ["model_mlflow_run_id", runtime_metric],
        ]
        .drop_duplicates("model_mlflow_run_id")
        .rename(columns={runtime_metric: "runtime"})
    )
    plot_data = with_model_labels(scores.merge(timings, on="model_mlflow_run_id"))

    created_axis = ax is None
    if created_axis:
        _, ax = plt.subplots(figsize=(10, 6))
    styles = instance_plot_styles(plot_data)
    for instance, (style, label) in styles.items():
        rows = plot_data.loc[plot_data["model_instance"].eq(instance)]
        if rows.empty:
            continue
        ax.scatter(
            rows["runtime"],
            rows["external_score"],
            color=style.color,
            marker=style.marker,
            s=52,
            zorder=3,
            label=label,
        )
        if show_ci:
            draw_confidence_intervals(
                ax,
                rows["runtime"],
                rows["external_score"],
                rows["external_ci_lower"],
                rows["external_ci_upper"],
                style.color,
            )
    for rank, runtime, score in zip(
        plot_data["generalization_rank"],
        plot_data["runtime"],
        plot_data["external_score"],
        strict=True,
    ):
        ax.annotate(
            f"#{rank}",
            (runtime, score),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
        )

    label = metric_label(metric)
    centers = ", ".join(dataset.upper() for dataset in plot_data["external_dataset"].unique())
    if log_x:
        ax.set_xscale("log")
    ax.set(
        xlabel=runtime_label(runtime_metric, log_x=log_x, scope=runtime_scope),
        ylabel=f"External {label}",
        title=title if title is not None else f"External performance ({centers}) vs runtime ",
    )
    ax.grid(alpha=0.2)
    if created_axis:
        ax.figure.subplots_adjust(right=0.82)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False, ncol=2)
    return ax


def _filter_models(
    frame: pd.DataFrame,
    include_models: Sequence[str] | None,
    ignore_models: Sequence[str] | None,
) -> pd.DataFrame:
    if ignore_models:
        frame = frame.loc[~frame["model_name"].isin(ignore_models)]
    if include_models:
        frame = frame.loc[frame["model_name"].isin(include_models)]
    return frame
