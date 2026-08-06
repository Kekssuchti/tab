"""Reusable data preparation and presentation helpers for evaluation plots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex

from src.plotting.defaults import model_label, ordered_models


@dataclass(frozen=True)
class ModelSettingPlotData:
    """Prepared rows and display metadata for model-setting comparisons."""

    frame: pd.DataFrame
    model_instances: tuple[str, ...]
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
    excluded_models_by_setting: Mapping[str, Sequence[str]] | None = None,
) -> ModelSettingPlotData:
    """Filter rows for a model-setting comparison.

    Filtering by test scope, point statistic, and dataset happens before a
    setting index and label are assigned. Within each model instance,
    occurrences are numbered in stable input order. Every compared instance
    must consequently occur the same number of times before setting-specific
    exclusions. The evaluation frame does not contain every model parameter,
    so parameter differences cannot be inferred here: callers must ensure that
    the same occurrence index represents the same setting for all models and
    may supply labels describing those settings.

    ``excluded_models_by_setting`` is applied only after occurrence validation.
    Its keys are resolved setting labels; each selector excludes either every
    instance with that model name or the exact matching model instance. Removed
    cells are not renumbered, and selectors absent from the selected model
    subset have no effect. Frames without ``model_instance`` fall back to using
    ``model_name`` as the plotting identity.

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

    if "model_instance" not in frame:
        frame["model_instance"] = frame["model_name"]

    frame["setting_index"] = frame.groupby("model_instance", sort=False).cumcount()
    counts = frame.groupby("model_instance", sort=False).size()
    if counts.nunique() != 1:
        details = ", ".join(f"{instance}={count}" for instance, count in counts.items())
        raise ValueError(
            "All compared model instances must have the same number of setting occurrences after filtering; "
            f"found {details}"
        )
    setting_count = int(counts.iloc[0])
    labels = _setting_labels(setting_labels, setting_count)
    frame["setting_label"] = frame["setting_index"].map(dict(enumerate(labels)))
    frame = _exclude_model_setting_rows(frame, labels, excluded_models_by_setting)
    if frame.empty:
        raise ValueError("Model setting exclusions removed all selected rows")

    ci_lower = f"{metric}_ci_lower"
    ci_upper = f"{metric}_ci_upper"
    has_ci = show_ci and ci_lower in frame.columns and ci_upper in frame.columns

    identities = frame[["model_name", "model_instance"]].drop_duplicates()
    model_rank = {model: index for index, model in enumerate(ordered_models(identities["model_name"].tolist()))}
    identities["_model_order"] = identities["model_name"].map(model_rank)
    identities = identities.sort_values("_model_order", kind="stable")
    instances = tuple(identities["model_instance"])
    instance_rank = {instance: index for index, instance in enumerate(instances)}
    frame["_model_order"] = frame["model_instance"].map(instance_rank)
    frame = frame.sort_values(["_model_order", "setting_index"], kind="stable").drop(columns="_model_order")

    return ModelSettingPlotData(
        frame=frame,
        model_instances=instances,
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
    if "model_instance" not in frame:
        frame = frame.copy()
        frame["model_instance"] = frame["model_name"]
    identity_column = "model_instance"
    identities = frame[["model_name", identity_column]].drop_duplicates()
    duplicate_models = set(identities.loc[identities["model_name"].duplicated(keep=False), "model_name"])
    lines = ["Model setting mapping:"]
    for setting_index, setting_label in enumerate(setting_labels):
        setting_rows = frame.loc[frame["setting_index"].eq(setting_index)]
        pipeline_summaries = []
        for pipeline_run_name in setting_rows["pipeline_run_name"].drop_duplicates():
            run_rows = setting_rows.loc[setting_rows["pipeline_run_name"].eq(pipeline_run_name)]
            run_identities = run_rows[["model_name", identity_column]].drop_duplicates()
            model_rank = {
                model: index for index, model in enumerate(ordered_models(run_identities["model_name"].tolist()))
            }
            run_identities["_model_order"] = run_identities["model_name"].map(model_rank)
            run_identities = run_identities.sort_values("_model_order", kind="stable")
            model_names = ", ".join(
                str(getattr(row, identity_column))
                if row.model_name in duplicate_models
                else model_label(row.model_name)
                for row in run_identities.itertuples()
            )
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


def _exclude_model_setting_rows(
    frame: pd.DataFrame,
    setting_labels: Sequence[str],
    excluded_models_by_setting: Mapping[str, Sequence[str]] | None,
) -> pd.DataFrame:
    if not excluded_models_by_setting:
        return frame
    if len(set(setting_labels)) != len(setting_labels):
        raise ValueError("setting_labels must be unique when excluding models by setting")

    unknown_labels = [label for label in excluded_models_by_setting if label not in setting_labels]
    if unknown_labels:
        labels = ", ".join(repr(label) for label in unknown_labels)
        raise ValueError(f"Unknown setting label(s): {labels}")

    excluded = pd.Series(False, index=frame.index)
    for setting_label, selectors in excluded_models_by_setting.items():
        excluded |= frame["setting_label"].eq(setting_label) & (
            frame["model_name"].isin(selectors) | frame["model_instance"].isin(selectors)
        )
    return frame.loc[~excluded].copy()


def _setting_colors(setting_count: int) -> tuple[str, ...]:
    if setting_count <= 10:
        palette = plt.get_cmap("tab10")
        return tuple(to_hex(palette(index)) for index in range(setting_count))
    palette = plt.get_cmap("turbo")
    return tuple(to_hex(palette(value)) for value in np.linspace(0.05, 0.95, setting_count))
