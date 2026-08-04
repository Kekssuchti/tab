"""Reusable data preparation and presentation helpers for evaluation plots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex

from src.plotting.defaults import ordered_models


@dataclass(frozen=True)
class ModelSettingPlotData:
    """Validated rows and display metadata for model-setting comparisons."""

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
    runtime_metric: str | None = None,
    log_x: bool = False,
) -> ModelSettingPlotData:
    """Filter and validate rows for a model-setting comparison.

    Filtering by test scope, point statistic, and dataset happens before a
    setting index is assigned. Within each model, occurrences are numbered in
    stable input order. Every compared model must consequently occur the same
    number of times. The evaluation frame does not contain every model
    parameter, so parameter differences cannot be inferred here: callers must
    ensure that the same occurrence index represents the same setting for all
    models and may supply labels describing those settings.

    Rows are never averaged or otherwise deduplicated.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be one pandas DataFrame")

    required = {"scope", "statistic", "dataset", "model_name", metric}
    if runtime_metric is not None:
        required.add(runtime_metric)
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required evaluation columns: {', '.join(missing)}")

    frame = data.loc[
        data["scope"].eq("test") & data["statistic"].eq("point") & data["dataset"].eq(dataset)
    ].copy()
    if ignore_models:
        frame = frame.loc[~frame["model_name"].isin(ignore_models)]
    if include_models:
        frame = frame.loc[frame["model_name"].isin(include_models)]
    if frame.empty:
        raise ValueError(
            f"No scope='test', statistic='point' rows are available for dataset {dataset!r} and the model filters"
        )

    invalid_names = frame["model_name"].isna() | ~frame["model_name"].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    )
    if invalid_names.any():
        raise ValueError("Column 'model_name' must contain a non-empty string for every selected row")

    _coerce_finite_numeric(frame, metric, allow_missing=False)
    if runtime_metric is not None:
        _coerce_finite_numeric(frame, runtime_metric, allow_missing=False)
        if log_x and frame[runtime_metric].le(0).any():
            invalid = frame.loc[frame[runtime_metric].le(0), runtime_metric].tolist()
            raise ValueError(
                f"Runtime column {runtime_metric!r} must contain strictly positive values when log_x=True; "
                f"found {invalid}"
            )

    frame["setting_index"] = frame.groupby("model_name", sort=False).cumcount()
    counts = frame.groupby("model_name", sort=False).size()
    if counts.nunique() != 1:
        details = ", ".join(f"{model}={count}" for model, count in counts.items())
        raise ValueError(
            "All compared models must have the same number of setting occurrences after filtering; "
            f"found {details}"
        )
    setting_count = int(counts.iloc[0])
    labels = _setting_labels(setting_labels, setting_count)

    ci_lower = f"{metric}_ci_lower"
    ci_upper = f"{metric}_ci_upper"
    has_ci = show_ci and ci_lower in frame.columns and ci_upper in frame.columns
    if has_ci:
        _coerce_finite_numeric(frame, ci_lower, allow_missing=True)
        _coerce_finite_numeric(frame, ci_upper, allow_missing=True)
        complete_ci = frame[ci_lower].notna() & frame[ci_upper].notna()
        invalid_ci = complete_ci & ((frame[ci_lower] > frame[metric]) | (frame[ci_upper] < frame[metric]))
        if invalid_ci.any():
            raise ValueError(
                f"Confidence interval columns {ci_lower!r} and {ci_upper!r} must bound {metric!r}"
            )

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


def runtime_label(runtime_metric: str, *, log_x: bool) -> str:
    """Return the standard model-runtime axis label."""
    metric_text = "total time" if runtime_metric == "total_time" else runtime_metric.replace("_", " ")
    scale_text = ", log scale" if log_x else ""
    return f"Model {metric_text} (seconds{scale_text})"


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


def _coerce_finite_numeric(frame: pd.DataFrame, column: str, *, allow_missing: bool) -> None:
    original = frame[column]
    numeric = pd.to_numeric(original, errors="coerce")
    invalid = numeric.isna() & (original.notna() if allow_missing else pd.Series(True, index=frame.index))
    if not allow_missing:
        invalid |= numeric.isna()
    finite = numeric.notna() & ~np.isfinite(numeric)
    if invalid.any() or finite.any():
        requirement = "numeric and finite when present" if allow_missing else "numeric, nonmissing, and finite"
        raise ValueError(f"Column {column!r} must be {requirement} for every selected row")
    frame[column] = numeric
