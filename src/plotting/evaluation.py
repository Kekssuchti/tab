"""Top-level plotting functions for pipeline evaluation results.

Shared style defaults (colors, markers, labels) come from
:mod:`src.plotting.defaults`.
"""

from __future__ import annotations

from typing import Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.plotting.defaults import (
    ModelStyle,
    dataset_label,
    metric_label,
    metric_lower_is_better,
    model_styles,
    ordered_models,
)
from src.plotting.plot_utils import prepare_model_setting_plot_data, runtime_label

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
    """Calculate external-test loss and rank across the selected models.

    Comparative loss is the model's external score minus the best external
    score among models for the same target and external dataset. The best
    model therefore has loss zero and rank one.
    """

    scores = _test_scores(results, metric)
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
    ax: Axes | None = None,
) -> Axes:
    """Compare one held-out metric between two test centers.

    Each model gets one row whose two points are colored in the model's
    shared color and marked per test center, so a model's scores can be
    tracked across datasets.
    """
    first, second = datasets
    scores = _test_scores(results, metric)
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
    paired = _with_model_labels(paired).sort_values("difference")

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
    instance_styles = _instance_plot_styles(paired)
    for dataset in datasets:
        for instance, (style, _) in instance_styles.items():
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
            _plot_horizontal_ci(
                ax,
                rows[f"{dataset}_value"],
                rows["_y"],
                rows[f"{dataset}_lower"],
                rows[f"{dataset}_upper"],
                style.color,
            )

    metric_label_text = metric_label(metric)
    score_bounds = pd.concat(
        [paired[f"{dataset}_{bound}"] for dataset in datasets for bound in ("value", "lower", "upper")]
    ).dropna()
    lower_limit = max(0.0, score_bounds.min() - 0.03)
    upper_limit = min(1.0, score_bounds.max() + 0.03)
    ax.set_yticks(positions, paired["model_label"])
    ax.invert_yaxis()
    ax.set(
        xlabel=metric_label_text,
        ylabel="",
        title=f"{metric_label_text} by test center",
        xlim=(lower_limit, upper_limit),
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


def plot_roc_auc(results: pd.DataFrame, *, ax: Axes | None = None) -> Axes:
    """Plot a ROC-AUC dumbbell comparison between MIMIC and TUDD."""

    return plot_score_dumbbell(results, metric="roc_auc", ax=ax)


def plot_generalization_gaps(
    results: pd.DataFrame,
    *,
    metric: str = "roc_auc",
    external_dataset: str | None = None,
    loss: Literal["model_specific", "comparative"] = "comparative",
    ax: Axes | None = None,
) -> Axes:
    """Plot model-specific or comparative external-test loss.

    Bars are colored per model using the shared model colors.
    """
    gaps = calculate_comparative_generalizability(results, metric=metric, external_dataset=external_dataset)
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
    gaps = _with_model_labels(gaps).sort_values([rank_column, "model_label"])

    if ax is None:
        _, ax = plt.subplots(figsize=(11, max(4, len(gaps) * 0.48)))
    positions = np.arange(len(gaps))
    instance_styles = _instance_plot_styles(gaps)
    bar_colors = [instance_styles[instance][0].color for instance in gaps["model_instance"]]
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

    metric_label_text = metric_label(metric)
    centers = ", ".join(dataset.upper() for dataset in gaps["external_dataset"].unique())
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(positions, gaps["model_label"])
    ax.invert_yaxis()
    ax.set(
        xlabel=f"{metric_label_text} difference (closer to zero is better)",
        ylabel="",
        title=f"{metric_label_text} {loss_label} (external: {centers})",
    )
    ax.grid(axis="x", alpha=0.2)
    return ax


def plot_performance_vs_runtime(
    data: pd.DataFrame,
    ignore_models: list[str] | None = None,
    *,
    metric: str = "roc_auc",
    test_dataset: str | None = None,
    runtime_scope: str = "model",
    runtime_metric: str = "total_time",
    ax: Axes | None = None,
) -> Axes:
    """Plot test performance against model runtime.

    Points are colored and marked per model using the shared model styles;
    each model's rank by external score is annotated next to its point.
    """
    if ignore_models:
        data = data.loc[~data["model_name"].isin(ignore_models)]
    comparison = calculate_comparative_generalizability(data, metric=metric, external_dataset=test_dataset)
    timings = (
        data.loc[
            (data["scope"] == "test") & data[runtime_metric].notna(),
            ["model_mlflow_run_id", runtime_metric],
        ]
        .drop_duplicates("model_mlflow_run_id")
        .rename(columns={runtime_metric: "runtime"})
    )
    plot_data = comparison.merge(timings, on="model_mlflow_run_id")
    plot_data = _with_model_labels(plot_data)

    created_axis = ax is None
    if created_axis:
        _, ax = plt.subplots(figsize=(10, 6))
    instance_styles = _instance_plot_styles(plot_data)
    for instance, (style, label) in instance_styles.items():
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
        _plot_vertical_ci(
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

    metric_label_text = metric_label(metric)
    centers = ", ".join(dataset.upper() for dataset in plot_data["external_dataset"].unique())
    ax.set_xscale("log")
    ax.set(
        xlabel=_runtime_label(runtime_scope, runtime_metric),
        ylabel=f"External {metric_label_text}",
        title=f"External performance ({centers}) vs runtime ",
    )
    ax.grid(alpha=0.2)
    if created_axis:
        ax.figure.subplots_adjust(right=0.82)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False, ncol=2)
    return ax


def plot_over_training_size(
    data: pd.DataFrame,
    ignore_models: list[str] | None = None,
    include_models: list[str] | None = None,
    *,
    metric: str = "roc_auc",
    datasets: Sequence[str] = ("mimic", "tudd"),
    run_aggregation: Literal["average"] | None = None,
    log_x: bool = True,
    show_ci: bool = True,
    show_title: bool = True,
) -> Figure:
    """Plot a metric against the training sample size for each test dataset.

    Each subplot shows one test dataset, with one line per model tracing the
    metric as the training sample size grows. Lines use the shared model styles
    from :func:`model_styles`. Confidence intervals are drawn as translucent
    bands around each line when available. Models listed in ``ignore_models``
    are excluded from the plot. By default, repeated pipeline runs remain
    separate points. Set ``run_aggregation="average"`` to average the metric
    and available confidence interval bounds for each model instance, training
    size, and test dataset. Instance IDs keep duplicate model configurations
    separate while matching the same configuration across pipeline runs.
    """
    if metric not in data.columns:
        raise ValueError(f"Metric {metric!r} is not available in evaluation data")
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
    if data.empty:
        raise ValueError("No test rows are available for the requested datasets and metric")

    if run_aggregation == "average":
        # A model_instance is stable across equivalent pipeline configurations;
        # distinct duplicate configurations retain their __0/__1 instance IDs.
        group_columns = ["model_name", "model_instance", "training_size", "dataset"]
        value_columns = [metric] + ([ci_lower, ci_upper] if has_ci else [])
        data = data.groupby(group_columns, sort=False, dropna=False)[value_columns].mean().reset_index()

    instances = data["model_instance"].drop_duplicates().tolist()
    styles = model_styles(data["model_name"].unique().tolist())
    instance_model = data.drop_duplicates("model_instance").set_index("model_instance")["model_name"].to_dict()
    instance_counts = data.groupby("model_name")["model_instance"].nunique()
    # Draw lines and legend in canonical MODEL_ORDER rather than first-seen order.
    model_rank = {model: index for index, model in enumerate(ordered_models(list(instance_model.values())))}
    instances.sort(key=lambda instance: model_rank[instance_model[instance]])

    datasets = tuple(datasets)
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 5), squeeze=False)
    axes = axes[0]

    for ax, dataset in zip(axes, datasets, strict=True):
        sub = data.loc[data["dataset"].eq(dataset)]
        for instance in instances:
            rows = sub.loc[sub["model_instance"].eq(instance)].sort_values("training_size")
            if rows.empty:
                continue
            model = instance_model[instance]
            style = styles[model]
            label = instance if instance_counts[model] > 1 else style.label
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
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{int(size):,}" for size in sizes], rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Training sample size")
        ax.set_ylabel(metric_label(metric))
        ax.set_title(dataset_label(dataset))
        ax.grid(alpha=0.3, which="both")
        ax.margins(y=0.08)

    if len(datasets) == 1:
        # Single dataset: place the legend inside the plot.
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


def plot_model_setting_performance(
    data: pd.DataFrame,
    ignore_models: list[str] | None = None,
    include_models: list[str] | None = None,
    *,
    metric: str = "roc_auc",
    dataset: str = "tudd",
    setting_labels: Sequence[str] | None = None,
    show_ci: bool = True,
    title: str | None = None,
    legend_title: str = "Setting",
) -> Figure:
    """Plot adjacent performance bars for repeated settings of each model.

    Selected ``scope='test'``/``statistic='point'`` rows are filtered to one
    dataset before each model's occurrences are numbered in stable input order.
    All model names must have equal occurrence counts, and occurrences are not
    averaged. Because evaluation rows do not represent every model parameter,
    setting differences cannot be inferred: occurrence order must consistently
    identify settings across models. Use ``setting_labels`` to name them.
    """
    prepared = prepare_model_setting_plot_data(
        data,
        metric=metric,
        dataset=dataset,
        setting_labels=setting_labels,
        include_models=include_models,
        ignore_models=ignore_models,
        show_ci=show_ci,
    )
    frame = prepared.frame
    model_positions = np.arange(len(prepared.model_names), dtype=float)
    bar_width = 0.8 / len(prepared.setting_labels)
    fig, ax = plt.subplots(figsize=(max(7, 1.35 * len(prepared.model_names)), 5.5))

    ci_lower = f"{metric}_ci_lower"
    ci_upper = f"{metric}_ci_upper"
    for setting_index, (setting_label, color) in enumerate(
        zip(prepared.setting_labels, prepared.setting_colors, strict=True)
    ):
        rows = (
            frame.loc[frame["setting_index"].eq(setting_index)]
            .set_index("model_name")
            .loc[list(prepared.model_names)]
        )
        positions = model_positions + (setting_index - (len(prepared.setting_labels) - 1) / 2) * bar_width
        ax.bar(positions, rows[metric], width=bar_width, color=color, label=setting_label, zorder=3)
        if prepared.has_ci:
            _plot_vertical_ci(ax, positions, rows[metric], rows[ci_lower], rows[ci_upper], color)

    labels = [model_styles([model])[model].label for model in prepared.model_names]
    ax.set_xticks(model_positions, labels)
    ax.set(
        xlabel="Model",
        ylabel=metric_label(metric),
        title=title if title is not None else f"{metric_label(metric)} by model setting on {dataset_label(dataset)}",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title=legend_title, frameon=False)
    fig.tight_layout()
    return fig


def plot_model_setting_performance_vs_runtime(
    data: pd.DataFrame,
    ignore_models: list[str] | None = None,
    include_models: list[str] | None = None,
    *,
    metric: str = "roc_auc",
    runtime_metric: str = "total_time",
    dataset: str = "tudd",
    setting_labels: Sequence[str] | None = None,
    log_x: bool = True,
    show_ci: bool = True,
    title: str | None = None,
    legend_title: str = "Setting",
) -> Figure:
    """Plot setting-level model performance against runtime.

    Setting occurrences use the same filtering, stable ordering, equal-count
    validation, and no-aggregation semantics as
    :func:`plot_model_setting_performance`. Color identifies settings, while
    canonical model labels annotate points and model markers are shared with
    the other evaluation plots. The runtime axis is inverted so faster models
    appear to the right; logarithmic scaling is used by default.
    """
    prepared = prepare_model_setting_plot_data(
        data,
        metric=metric,
        dataset=dataset,
        setting_labels=setting_labels,
        include_models=include_models,
        ignore_models=ignore_models,
        show_ci=show_ci,
        runtime_metric=runtime_metric,
        log_x=log_x,
    )
    frame = prepared.frame
    styles = model_styles(list(prepared.model_names))
    fig, ax = plt.subplots(figsize=(10, 6))
    ci_lower = f"{metric}_ci_lower"
    ci_upper = f"{metric}_ci_upper"

    for _, row in frame.iterrows():
        setting_index = int(row["setting_index"])
        model = row["model_name"]
        color = prepared.setting_colors[setting_index]
        x = row[runtime_metric]
        y = row[metric]
        ax.scatter(x, y, color=color, marker=styles[model].marker, s=58, zorder=3)
        if prepared.has_ci:
            _plot_vertical_ci(
                ax,
                pd.Series([x]),
                pd.Series([y]),
                pd.Series([row[ci_lower]]),
                pd.Series([row[ci_upper]]),
                color,
            )
        ax.annotate(
            styles[model].label,
            (x, y),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )

    if log_x:
        ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set(
        xlabel=runtime_label(runtime_metric, log_x=log_x),
        ylabel=metric_label(metric),
        title=title
        if title is not None
        else f"{metric_label(metric)} vs model runtime on {dataset_label(dataset)}",
    )
    ax.grid(alpha=0.3, which="both")
    legend_handles = [
        Line2D([], [], marker="o", linestyle="none", color=color, markersize=7, label=label)
        for label, color in zip(prepared.setting_labels, prepared.setting_colors, strict=True)
    ]
    ax.legend(handles=legend_handles, title=legend_title, frameon=False)
    fig.tight_layout()
    return fig


def _test_scores(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    ci_lower = f"{metric}_ci_lower"
    ci_upper = f"{metric}_ci_upper"
    required = {"scope", "statistic", metric, ci_lower, ci_upper}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Missing required evaluation columns: {', '.join(missing)}")
    scores = results.loc[
        results["scope"].eq("test") & results["statistic"].eq("point") & results[metric].notna()
    ].copy()
    scores["metric"] = metric
    scores["value"] = scores[metric]
    scores["ci_lower"] = scores[ci_lower]
    scores["ci_upper"] = scores[ci_upper]
    return scores


def _instance_plot_styles(frame: pd.DataFrame) -> dict[str, tuple[ModelStyle, str]]:
    """Map each model instance to its shared style and a legend label.

    Single-instance models use the display label (e.g. ``TabPFNv3``); models
    with duplicate instances fall back to the raw instance id so legend
    entries never collide.
    """
    unique = frame.drop_duplicates("model_instance")
    instance_model: dict[str, str] = dict(
        zip(unique["model_instance"].astype(str), unique["model_name"].astype(str), strict=True)
    )
    styles = model_styles(list(instance_model.values()))
    instance_counts: dict[str, int] = {}
    for model in instance_model.values():
        instance_counts[model] = instance_counts.get(model, 0) + 1
    result: dict[str, tuple[ModelStyle, str]] = {}
    for instance, model in instance_model.items():
        style = styles[model]
        label = style.label if instance_counts[model] == 1 else instance
        result[instance] = (style, label)
    return result


def _with_model_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    labels = {instance: label for instance, (_, label) in _instance_plot_styles(frame).items()}
    frame["model_label"] = frame["model_instance"].map(labels.__getitem__)
    return frame


def _plot_horizontal_ci(
    ax: Axes,
    values: pd.Series,
    positions: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    color: str,
) -> None:
    values = np.asarray(values, dtype=float)
    positions = np.asarray(positions, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    present = ~np.isnan(lower) & ~np.isnan(upper)
    if present.any():
        ax.errorbar(
            values[present],
            positions[present],
            xerr=[
                values[present] - lower[present],
                upper[present] - values[present],
            ],
            fmt="none",
            ecolor=color,
            capsize=2,
            alpha=0.65,
            zorder=2,
        )


def _plot_vertical_ci(
    ax: Axes,
    x: pd.Series,
    values: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    color: str,
) -> None:
    x = np.asarray(x, dtype=float)
    values = np.asarray(values, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    present = ~np.isnan(lower) & ~np.isnan(upper)
    if present.any():
        ax.errorbar(
            x[present],
            values[present],
            yerr=np.vstack(
                [
                    values[present] - lower[present],
                    upper[present] - values[present],
                ]
            ),
            fmt="none",
            ecolor=color,
            capsize=2,
            alpha=0.65,
            zorder=2,
        )


def _runtime_label(scope: str, metric: str) -> str:
    if (scope, metric) == ("model", "total_time"):
        return "Model total time (seconds, log scale)"
    scope_label = scope.replace("_", " ").title()
    metric_label_text = metric.replace("_", " ")
    return f"{scope_label} {metric_label_text} (seconds, log scale)"
