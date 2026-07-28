from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

_MODEL_COLUMNS = [
    "pipeline_mlflow_run_id",
    "pipeline_run_name",
    "model_mlflow_run_id",
    "model_name",
    "model_instance",
    "target",
    "trained_on",
]
_CENTER_COLORS = {"mimic": "#315C73", "tudd": "#D17A3F"}


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
    external["generalizability_loss"] = external["external_score"] - external["training_score"]
    groups = ["target", "external_dataset"]
    external["best_external_score"] = external.groupby(groups, dropna=False)["external_score"].transform("max")
    external["comparative_generalizability_loss"] = external["external_score"] - external["best_external_score"]
    external["generalization_rank"] = (
        external.groupby(groups, dropna=False)["external_score"].rank(method="min", ascending=False).astype(int)
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
    """Compare one held-out metric between two test centers."""

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
        values.append(
            center[
                _MODEL_COLUMNS
                + [
                    f"{dataset}_value",
                    f"{dataset}_lower",
                    f"{dataset}_upper",
                ]
            ]
        )
    paired = values[0].merge(values[1], on=_MODEL_COLUMNS)
    paired["difference"] = paired[f"{first}_value"] - paired[f"{second}_value"]
    paired = _with_model_labels(paired).sort_values("difference")

    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(4, len(paired) * 0.42)))
    positions = np.arange(len(paired))
    ax.hlines(
        positions,
        paired[f"{first}_value"],
        paired[f"{second}_value"],
        color="#A7A7A7",
        linewidth=2,
        zorder=1,
    )
    for dataset in datasets:
        values_column = f"{dataset}_value"
        color = _CENTER_COLORS.get(dataset, "#777777")
        ax.scatter(
            paired[values_column],
            positions,
            color=color,
            s=42,
            label=dataset.upper(),
            zorder=3,
        )
        _plot_horizontal_ci(
            ax,
            paired[values_column],
            positions,
            paired[f"{dataset}_lower"],
            paired[f"{dataset}_upper"],
            color,
        )

    metric_label = _metric_label(metric)
    score_bounds = pd.concat(
        [paired[f"{dataset}_{bound}"] for dataset in datasets for bound in ("value", "lower", "upper")]
    ).dropna()
    lower_limit = max(0.0, score_bounds.min() - 0.03)
    upper_limit = min(1.0, score_bounds.max() + 0.03)
    ax.set_yticks(positions, paired["model_label"])
    ax.invert_yaxis()
    ax.set(
        xlabel=metric_label,
        ylabel="",
        title=f"{metric_label} by test center",
        xlim=(lower_limit, upper_limit),
    )
    ax.grid(axis="x", alpha=0.2)
    ax.legend(title="Test center")
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
    """Plot model-specific or comparative external-test loss."""

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
    ax.barh(
        positions,
        gaps[loss_column],
        color="#6F8FA6" if loss == "model_specific" else "#C2673D",
    )
    for position, row in zip(positions, gaps.itertuples(), strict=True):
        value = getattr(row, loss_column)
        ax.annotate(
            f"#{getattr(row, rank_column)}",
            (value, position),
            xytext=(-5, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    metric_label = _metric_label(metric)
    centers = ", ".join(dataset.upper() for dataset in gaps["external_dataset"].unique())
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(positions, gaps["model_label"])
    ax.invert_yaxis()
    ax.set(
        xlabel=f"{metric_label} difference (closer to zero is better)",
        ylabel="",
        title=f"{metric_label} {loss_label} (external: {centers})",
    )
    ax.grid(axis="x", alpha=0.2)
    return ax


def plot_performance_vs_runtime(
    results: pd.DataFrame,
    *,
    metric: str = "roc_auc",
    external_dataset: str | None = None,
    runtime_scope: str = "model",
    runtime_metric: str = "total_time",
    ax: Axes | None = None,
) -> Axes:
    """Plot external-test performance against model runtime."""

    comparison = calculate_comparative_generalizability(results, metric=metric, external_dataset=external_dataset)
    timings = (
        results.loc[
            (results["scope"] == "test") & results[runtime_metric].notna(),
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
    ax.scatter(
        plot_data["runtime"],
        plot_data["external_score"],
        color="#315C73",
        s=52,
        zorder=3,
    )
    _plot_vertical_ci(
        ax,
        plot_data["runtime"],
        plot_data["external_score"],
        plot_data["external_ci_lower"],
        plot_data["external_ci_upper"],
        "#315C73",
    )
    for row in plot_data.itertuples():
        ax.annotate(
            f"#{row.generalization_rank}",
            (row.runtime, row.external_score),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
        )
    rank_key = "\n".join(
        f"#{row.generalization_rank}  {row.model_label}"
        for row in plot_data.sort_values("generalization_rank").itertuples()
    )
    if created_axis:
        ax.figure.subplots_adjust(right=0.72)
    ax.text(
        1.03,
        1.0,
        rank_key,
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        linespacing=1.45,
    )

    metric_label = _metric_label(metric)
    centers = ", ".join(dataset.upper() for dataset in plot_data["external_dataset"].unique())
    ax.set_xscale("log")
    ax.set(
        xlabel=_runtime_label(runtime_scope, runtime_metric),
        ylabel=f"External {metric_label}",
        title=f"External performance vs runtime ({centers})",
    )
    ax.grid(alpha=0.2)
    return ax


def _test_scores(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in results:
        raise ValueError(f"Metric {metric!r} is not available in evaluation data")
    scores = results.loc[(results["scope"] == "test") & results[metric].notna()].copy()
    scores["metric"] = metric
    scores["value"] = scores[metric]
    scores["ci_lower"] = scores[f"{metric}_ci_lower"]
    scores["ci_upper"] = scores[f"{metric}_ci_upper"]
    return scores


def _with_model_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["model_label"] = frame["model_instance"]
    duplicates = frame["model_instance"].duplicated(keep=False)
    frame.loc[duplicates, "model_label"] = frame.loc[duplicates].apply(
        lambda row: f"{row['model_instance']} / {str(row['pipeline_run_name'])[-8:]}",
        axis=1,
    )
    return frame


def _plot_horizontal_ci(
    ax: Axes,
    values: pd.Series,
    positions: np.ndarray,
    lower: pd.Series,
    upper: pd.Series,
    color: str,
) -> None:
    present = lower.notna() & upper.notna()
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
    present = lower.notna() & upper.notna()
    if present.any():
        ax.errorbar(
            x[present],
            values[present],
            yerr=[
                values[present] - lower[present],
                upper[present] - values[present],
            ],
            fmt="none",
            ecolor=color,
            capsize=2,
            alpha=0.65,
            zorder=2,
        )


def _metric_label(metric: str) -> str:
    return {
        "roc_auc": "ROC AUC",
        "prc_auc": "PRC AUC",
        "f1": "F1",
    }.get(metric, metric.replace("_", " ").title())


def _runtime_label(scope: str, metric: str) -> str:
    if (scope, metric) == ("model", "total_time"):
        return "Model total time (seconds, log scale)"
    scope_label = scope.replace("_", " ").title()
    metric_label = metric.replace("_", " ")
    return f"{scope_label} {metric_label} (seconds, log scale)"
