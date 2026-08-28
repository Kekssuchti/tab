"""Reusable plots for sample-size experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.plotting.defaults import dataset_label, metric_label, metric_scale
from src.plotting.plot_support import instance_plot_styles


def plot_over_training_size(
    data: pd.DataFrame,
    ignore_models: Sequence[str] | None = None,
    include_models: Sequence[str] | None = None,
    *,
    metric: str = "roc_auc",
    datasets: Sequence[str] = ("mimic", "tudd"),
    run_aggregation: Literal["average"] | None = None,
    log_x: bool = True,
    show_ci: bool = True,
    show_title: bool = True,
    y_label: str | None = None,
) -> Figure:
    """Plot a metric against training sample size for each test dataset.

    Repeated pipeline runs remain separate by default. Set
    ``run_aggregation="average"`` to average scores and available confidence
    bounds by model instance, training size, and test dataset. Bounded
    classification scores and confidence bounds are displayed as points on a
    0--100 scale.
    """
    if run_aggregation not in {None, "average"}:
        raise ValueError("run_aggregation must be: average")

    ci_lower = f"{metric}_ci_lower"
    ci_upper = f"{metric}_ci_upper"
    has_ci = show_ci and ci_lower in data.columns and ci_upper in data.columns

    data = data.loc[(data["scope"] == "test") & (data["dataset"].isin(datasets)) & data[metric].notna()].copy()
    if ignore_models:
        data = data.loc[~data["model_name"].isin(ignore_models)]
    if include_models:
        data = data.loc[data["model_name"].isin(include_models)]

    data["training_size"] = pd.to_numeric(data["training_size"], errors="coerce")
    data = data.dropna(subset=["training_size"])

    if run_aggregation == "average":
        group_columns = ["model_name", "model_instance", "training_size", "dataset"]
        value_columns = [metric] + ([ci_lower, ci_upper] if has_ci else [])
        data = data.groupby(group_columns, sort=False, dropna=False)[value_columns].mean().reset_index()

    value_columns = [metric] + ([ci_lower, ci_upper] if has_ci else [])
    data.loc[:, value_columns] = data[value_columns] * metric_scale(metric)

    styles = instance_plot_styles(data)
    datasets = tuple(datasets)
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 5), squeeze=False)
    axes = axes[0]

    for ax, dataset in zip(axes, datasets, strict=True):
        sub = data.loc[data["dataset"].eq(dataset)]
        for instance, (style, label) in styles.items():
            rows = sub.loc[sub["model_instance"].eq(instance)].sort_values("training_size")
            if rows.empty:
                continue
            ax.plot(
                rows["training_size"],
                rows[metric],
                marker=style.marker,
                markersize=4,
                linewidth=1.6,
                linestyle=style.linestyle,
                color=style.color,
                label=label,
            )
            if has_ci:
                band = rows.loc[rows[ci_lower].notna() & rows[ci_upper].notna()]
                if not band.empty:
                    ax.fill_between(
                        band["training_size"],
                        band[ci_lower],
                        band[ci_upper],
                        color=style.color,
                        alpha=0.18,
                        linewidth=0,
                    )
        sizes = np.unique(data["training_size"].to_numpy())
        if log_x:
            ax.set_xscale("log")
            x_label_ext = " (log scale)"
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{int(size):,}" for size in sizes], rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Training sample count" + x_label_ext)
        ax.set_ylabel(metric_label(metric))
        if "time" in metric:
            ax.set_yscale("log")
            ax.set_ylabel(y_label)
        if show_title:
            ax.set_title(dataset_label(dataset))
        ax.grid(alpha=0.3, which="both")
        ax.margins(y=0.08)

    if len(datasets) == 1:
        if axes[0].get_legend_handles_labels()[0]:
            axes[0].legend(loc="best", frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.93),
                ncol=min(len(handles), 4),
                frameon=False,
                fontsize=9,
            )
        fig.tight_layout(rect=(0, 0, 1, 0.9))

    if show_title:
        fig.suptitle(f"{metric_label(metric)} vs training size", fontsize=13, y=0.99)
    return fig
