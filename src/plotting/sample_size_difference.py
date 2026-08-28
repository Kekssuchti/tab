"""Plot model differences across sample-size experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.plotting.defaults import dataset_label, metric_label, metric_scale, model_label
from src.plotting.plot_support import instance_plot_styles


def plot_difference_training_size(
    data: pd.DataFrame,
    baseline_model: str,
    compare_models: Sequence[str],
    *,
    metric: str = "roc_auc",
    datasets: Sequence[str] = ("mimic", "tudd"),
    run_aggregation: Literal["average"] | None = None,
    log_x: bool = True,
    show_title: bool = True,
) -> Figure:
    """Plot signed metric differences from a baseline over training size.

    Each difference is the comparison model's metric minus the baseline metric,
    so positive values indicate a higher score than the baseline. Differences
    are calculated within each test dataset and training size. By
    default, comparison and baseline rows are paired by pipeline run. Set
    run_aggregation to "average" to average each model instance across runs
    before calculating the differences. Bounded classification metric
    differences are displayed as points on a 0--100 scale.

    baseline_model and compare_models refer to model names, rather than
    model-instance IDs. The baseline must resolve to exactly one model instance.
    """
    if run_aggregation not in {None, "average"}:
        raise ValueError("run_aggregation must be: average")

    compare_models = tuple(dict.fromkeys(compare_models))
    if baseline_model in compare_models:
        raise ValueError("baseline_model must not be included in compare_models")
    if not compare_models:
        raise ValueError("compare_models must contain at least one model")

    datasets = tuple(dict.fromkeys(datasets))
    if not datasets:
        raise ValueError("datasets must contain at least one dataset")

    required = {"scope", "dataset", "model_name", "model_instance", "training_size", metric}
    if run_aggregation is None:
        required.add("pipeline_id")
    missing_columns = sorted(required.difference(data.columns))
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))

    selected_models = (baseline_model, *compare_models)
    frame = data.loc[
        data["scope"].eq("test") & data["dataset"].isin(datasets) & data["model_name"].isin(selected_models)
    ].copy()
    frame["training_size"] = pd.to_numeric(frame["training_size"], errors="coerce")
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce") * metric_scale(metric)
    frame = frame.dropna(subset=["training_size", metric])

    available_models = set(frame["model_name"])
    missing_models = [name for name in selected_models if name not in available_models]
    if missing_models:
        raise ValueError("No usable rows for models: " + ", ".join(missing_models))

    baseline_instances = frame.loc[frame["model_name"].eq(baseline_model), "model_instance"].unique()
    if len(baseline_instances) != 1:
        raise ValueError(
            f"baseline_model {baseline_model!r} must resolve to exactly one model instance; "
            f"found {len(baseline_instances)}"
        )

    pair_columns = ["training_size", "dataset"]
    if run_aggregation == "average":
        frame = (
            frame.groupby(["model_name", "model_instance", *pair_columns], sort=False, dropna=False)[metric]
            .mean()
            .reset_index()
        )
    else:
        pair_columns.insert(0, "pipeline_id")

    baseline = frame.loc[frame["model_name"].eq(baseline_model), [*pair_columns, metric]].rename(
        columns={metric: "baseline_metric"}
    )
    duplicate_baselines = baseline.duplicated(pair_columns, keep=False)
    if duplicate_baselines.any():
        raise ValueError("Multiple baseline rows exist for the same run, training size, and dataset")

    differences = frame.loc[frame["model_name"].isin(compare_models)].merge(
        baseline,
        on=pair_columns,
        how="inner",
        validate="many_to_one",
    )
    if differences.empty:
        raise ValueError("No comparison rows could be paired with baseline rows")
    differences["difference"] = differences[metric] - differences["baseline_metric"]

    styles = instance_plot_styles(differences)
    figure, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 5), squeeze=False)
    axes = axes[0]
    sizes = np.sort(differences["training_size"].unique())

    for axis, dataset in zip(axes, datasets, strict=True):
        dataset_rows = differences.loc[differences["dataset"].eq(dataset)]
        for instance, (style, label) in styles.items():
            rows = dataset_rows.loc[dataset_rows["model_instance"].astype(str).eq(instance)].sort_values(
                "training_size"
            )
            if rows.empty:
                continue
            axis.plot(
                rows["training_size"],
                rows["difference"],
                marker=style.marker,
                markersize=4,
                linewidth=1.6,
                linestyle=style.linestyle,
                color=style.color,
                label=label,
            )
        axis.axhline(0, color="#757575", linewidth=0.8, alpha=0.7)
        if log_x:
            axis.set_xscale("log")
            x_label_ext = " (log scale)"
        else:
            x_label_ext = ""
        axis.set_xticks(sizes)
        axis.set_xticklabels([f"{int(size):,}" for size in sizes], rotation=45, ha="right", fontsize=8)
        axis.set_xlabel("Training sample count" + x_label_ext)
        axis.set_ylabel(f"Δ {metric_label(metric)}")
        if show_title:
            axis.set_title(dataset_label(dataset))
        axis.grid(alpha=0.3, which="both")
        axis.margins(y=0.08)

    if len(datasets) == 1:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(loc="best", frameon=False, fontsize=9)
        figure.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            figure.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.93),
                ncol=min(len(handles), 4),
                frameon=False,
                fontsize=9,
            )
        figure.tight_layout(rect=(0, 0, 1, 0.9))

    if show_title:
        figure.suptitle(
            f"{metric_label(metric)} difference from {model_label(baseline_model)}",
            fontsize=13,
            y=0.99,
        )
    return figure
