"""Reusable data preparation and presentation helpers for evaluation plots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex

from src.plotting.defaults import model_label, ordered_models


@dataclass(frozen=True)
class ModelSettingPlotData:
    """Prepared rows and display metadata for model-setting comparisons."""

    frame: pd.DataFrame
    model_names: tuple[str, ...]
    setting_labels: tuple[str, ...]
    setting_colors: tuple[str, ...]
    has_ci: bool


def prepare_model_setting_plot_data(
    data: pd.DataFrame,
    *,
    metric: str,
    dataset: str,
    setting_labels: Sequence[str] | None,
    include_models: Sequence[str] | None,
    ignore_models: Sequence[str] | None,
    show_ci: bool,
) -> ModelSettingPlotData:
    """Filter rows for a model-setting comparison.

    Filtering by test scope, point statistic, and dataset happens before a
    setting index is assigned. Within each model, occurrences are numbered in
    stable input order. Every compared model must consequently occur the same
    number of times. The evaluation frame does not contain every model
    parameter, so parameter differences cannot be inferred here: callers must
    ensure that the same occurrence index represents the same setting for all
    models and may supply labels describing those settings.

    Rows are never averaged or otherwise deduplicated.
    """
    if "pipeline_run_name" not in data:
        raise ValueError("Missing required evaluation columns: pipeline_run_name")

    frame = data.loc[data["scope"].eq("test") & data["statistic"].eq("point") & data["dataset"].eq(dataset)].copy()
    if ignore_models:
        frame = frame.loc[~frame["model_name"].isin(ignore_models)]
    if include_models:
        frame = frame.loc[frame["model_name"].isin(include_models)]
    if frame.empty:
        raise ValueError(
            f"No scope='test', statistic='point' rows are available for dataset {dataset!r} and the model filters"
        )
    valid_run_names = frame["pipeline_run_name"].map(lambda value: isinstance(value, str) and bool(value.strip()))
    if not valid_run_names.all():
        raise ValueError("pipeline_run_name must be a nonblank string; fix the selected evaluation rows")

    frame["setting_index"] = frame.groupby("model_name", sort=False).cumcount()
    counts = frame.groupby("model_name", sort=False).size()
    if counts.nunique() != 1:
        details = ", ".join(f"{model}={count}" for model, count in counts.items())
        raise ValueError(
            f"All compared models must have the same number of setting occurrences after filtering; found {details}"
        )
    setting_count = int(counts.iloc[0])
    labels = _setting_labels(setting_labels, setting_count)

    ci_lower = f"{metric}_ci_lower"
    ci_upper = f"{metric}_ci_upper"
    has_ci = show_ci and ci_lower in frame.columns and ci_upper in frame.columns

    models = tuple(ordered_models(frame["model_name"].drop_duplicates().tolist()))
    model_rank = {model: index for index, model in enumerate(models)}
    frame["_model_order"] = frame["model_name"].map(model_rank)
    frame = frame.sort_values(["_model_order", "setting_index"], kind="stable").drop(columns="_model_order")

    return ModelSettingPlotData(
        frame=frame,
        model_names=models,
        setting_labels=labels,
        setting_colors=_setting_colors(setting_count),
        has_ci=has_ci,
    )


def runtime_label(runtime_metric: str, *, log_x: bool, scope: str = "model") -> str:
    """Return the standard runtime axis label."""
    metric_text = "total time" if runtime_metric == "total_time" else runtime_metric.replace("_", " ")
    scale_text = ", log scale" if log_x else ""
    return f"{scope.replace('_', ' ').title()} {metric_text} (seconds{scale_text})"


def format_model_setting_mapping(frame: pd.DataFrame, setting_labels: Sequence[str]) -> str:
    """Format setting-to-pipeline/model provenance from prepared plot rows."""
    lines = ["Model setting mapping:"]
    for setting_index, setting_label in enumerate(setting_labels):
        setting_rows = frame.loc[frame["setting_index"].eq(setting_index)]
        pipeline_summaries = []
        for pipeline_run_name in setting_rows["pipeline_run_name"].drop_duplicates():
            run_rows = setting_rows.loc[setting_rows["pipeline_run_name"].eq(pipeline_run_name)]
            models = ordered_models(run_rows["model_name"].drop_duplicates().tolist())
            model_names = ", ".join(model_label(model) for model in models)
            pipeline_summaries.append(f"{pipeline_run_name}: {model_names}")
        lines.append(f"{setting_label}: {'; '.join(pipeline_summaries)}")
    return "\n".join(lines)


def calculate_y_limits(
    values: Sequence[float],
    y_limits: Literal["auto"] | tuple[float, float] | None,
    *,
    ci_lower: Sequence[float] | None = None,
    ci_upper: Sequence[float] | None = None,
    natural_bounds: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    """Return explicit limits or calculate padded limits around plotted values.

    ``None`` leaves axis limit selection to matplotlib. Automatic limits include
    complete confidence intervals and may be clipped to known natural metric
    bounds, such as ``(0, 1)`` for classification scores.
    """
    if y_limits is None:
        return None
    if y_limits != "auto":
        try:
            lower, upper = map(float, y_limits)
        except (TypeError, ValueError):
            raise ValueError("y_limits must contain exactly two finite numeric bounds") from None
        if not np.isfinite((lower, upper)).all() or lower >= upper:
            raise ValueError("y_limits must contain finite bounds with lower < upper")
        return lower, upper

    bounds = [np.asarray(values, dtype=float).ravel()]
    if ci_lower is not None and ci_upper is not None:
        lower_values = np.asarray(ci_lower, dtype=float)
        upper_values = np.asarray(ci_upper, dtype=float)
        complete = np.isfinite(lower_values) & np.isfinite(upper_values)
        bounds.extend((lower_values[complete], upper_values[complete]))

    finite_bounds = np.concatenate(bounds)
    finite_bounds = finite_bounds[np.isfinite(finite_bounds)]
    lower = float(finite_bounds.min())
    upper = float(finite_bounds.max())
    span = upper - lower
    scale = max(abs(lower), abs(upper), 1.0)
    padding = 0.08 * span if span > scale * 1e-9 else 0.05 * scale
    lower -= padding
    upper += padding

    if natural_bounds is not None:
        natural_lower, natural_upper = natural_bounds
        lower = max(lower, natural_lower)
        upper = min(upper, natural_upper)
    return lower, upper


def _setting_labels(labels: Sequence[str] | None, setting_count: int) -> tuple[str, ...]:
    if labels is None:
        return tuple(f"Setting {index + 1}" for index in range(setting_count))
    if isinstance(labels, str) or len(labels) != setting_count:
        actual_count = len(labels) if not isinstance(labels, str) else 1
        raise ValueError(f"Expected {setting_count} setting labels, received {actual_count}")
    return tuple(str(label) for label in labels)


def _setting_colors(setting_count: int) -> tuple[str, ...]:
    if setting_count <= 10:
        palette = plt.get_cmap("tab10")
        return tuple(to_hex(palette(index)) for index in range(setting_count))
    palette = plt.get_cmap("turbo")
    return tuple(to_hex(palette(value)) for value in np.linspace(0.05, 0.95, setting_count))
