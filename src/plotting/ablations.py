"""Reusable plots and preparation helpers for setting/ablation experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.plotting.defaults import POINT_SCALE_METRICS, metric_label, metric_scale, model_label, ordered_models
from src.plotting.plot_support import draw_confidence_intervals, instance_plot_styles
from src.plotting.plot_utils import runtime_label


@dataclass(frozen=True)
class ModelSettingPlotData:
    """Prepared rows and display metadata for model-setting comparisons."""

    frame: pd.DataFrame
    model_instances: tuple[str, ...]
    setting_labels: tuple[str, ...]
    setting_colors: tuple[str, ...]
    has_ci: bool


def plot_model_setting_performance(
    data: pd.DataFrame,
    ignore_models: Sequence[str] | None = None,
    include_models: Sequence[str] | None = None,
    *,
    metric: str = "roc_auc",
    dataset: str = "tudd",
    setting_run_ids: Mapping[str, str | Sequence[str]] | None = None,
    setting_labels: Sequence[str] | None = None,
    excluded_models_by_setting: Mapping[str, Sequence[str]] | None = None,
    show_ci: bool = True,
    title: str | None = None,
    legend_title: str = "Setting",
    y_limits: Literal["auto"] | tuple[float, float] | None = None,
) -> Figure:
    """Plot adjacent performance bars for repeated model settings.

    ``setting_run_ids`` is the preferred identity mechanism. Its keys are
    display labels in plotting order and each value is one pipeline MLflow run
    ID or a sequence of IDs belonging to that setting. Multiple IDs may
    partition a setting's models across disjoint pipeline runs, but each model
    instance must still contribute at most one row to a setting because these
    plots do not aggregate runs. When omitted, the legacy behavior assigns
    ``setting_labels`` by stable occurrence order within each model instance.
    """
    prepared = prepare_model_setting_plot_data(
        data,
        metric=metric,
        dataset=dataset,
        setting_run_ids=setting_run_ids,
        setting_labels=setting_labels,
        include_models=include_models,
        ignore_models=ignore_models,
        show_ci=show_ci,
        excluded_models_by_setting=excluded_models_by_setting,
    )
    frame = _scale_metric_columns(prepared.frame, metric, prepared.has_ci)
    model_positions = np.arange(len(prepared.model_instances), dtype=float)
    instance_positions = dict(zip(prepared.model_instances, model_positions, strict=True))
    bar_width = 0.8 / len(prepared.setting_labels)
    fig, ax = plt.subplots(figsize=(max(7, 1.35 * len(prepared.model_instances)), 5.5))

    ci_lower = f"{metric}_ci_lower"
    ci_upper = f"{metric}_ci_upper"
    for setting_index, (setting_label, color) in enumerate(
        zip(prepared.setting_labels, prepared.setting_colors, strict=True)
    ):
        rows = frame.loc[frame["setting_index"].eq(setting_index)]
        positions = rows["model_instance"].map(instance_positions).to_numpy(dtype=float)
        positions += (setting_index - (len(prepared.setting_labels) - 1) / 2) * bar_width
        ax.bar(positions, rows[metric], width=bar_width, color=color, label=setting_label, zorder=3)
        if prepared.has_ci:
            draw_confidence_intervals(ax, positions, rows[metric], rows[ci_lower], rows[ci_upper], color)

    styles = instance_plot_styles(frame)
    labels = [styles[instance][1] for instance in prepared.model_instances]
    ax.set_xticks(model_positions, labels)
    ax.set(xlabel="Model", ylabel=metric_label(metric))
    resolved_y_limits = calculate_y_limits(
        frame[metric],
        y_limits,
        ci_lower=frame[ci_lower] if prepared.has_ci else None,
        ci_upper=frame[ci_upper] if prepared.has_ci else None,
        natural_bounds=(0.0, 100.0) if metric in POINT_SCALE_METRICS else None,
    )
    if resolved_y_limits is not None:
        ax.set_ylim(resolved_y_limits)
    ax.grid(axis="y", alpha=0.3)
    apply_model_setting_header(
        fig,
        ax,
        title=title,
        legend_title=legend_title,
        legend_columns=min(len(prepared.setting_labels), 4),
    )
    print(format_model_setting_mapping(frame, prepared.setting_labels))
    return fig


def plot_model_setting_performance_vs_runtime(
    data: pd.DataFrame,
    ignore_models: Sequence[str] | None = None,
    include_models: Sequence[str] | None = None,
    *,
    metric: str = "roc_auc",
    runtime_metric: str = "total_time",
    dataset: str = "tudd",
    setting_run_ids: Mapping[str, str | Sequence[str]] | None = None,
    setting_labels: Sequence[str] | None = None,
    excluded_models_by_setting: Mapping[str, Sequence[str]] | None = None,
    run_aggregation: Literal["average"] | None = None,
    log_x: bool = True,
    show_ci: bool = True,
    title: str | None = None,
    legend_title: str = "Setting",
    invert_x: bool = False,
    x_axis_label: str | None = None,
    keep_labels_inside: bool = True,
) -> Figure:
    """Plot setting-level model performance against runtime.

    By default, prefer ``setting_run_ids`` to assign settings by pipeline
    MLflow run ID. Multiple IDs may partition a setting's models across
    disjoint runs, but cannot produce more than one row for the same model
    instance and setting. The legacy occurrence-order path remains available
    when it is omitted.

    Set ``run_aggregation="average"`` to average repeated runs into one point
    per model instance instead of comparing settings. This mode also averages
    the selected runtime metric and available confidence interval bounds. When
    ``keep_labels_inside`` is true, labels at the visual right edge are placed
    to the left of their points.
    """
    _validate_run_aggregation_options(
        run_aggregation,
        setting_run_ids=setting_run_ids,
        setting_labels=setting_labels,
        excluded_models_by_setting=excluded_models_by_setting,
    )
    _validate_required_columns(data, (metric, runtime_metric))

    if run_aggregation == "average":
        frame, has_ci = _prepare_averaged_run_plot_data(
            data,
            metric=metric,
            runtime_metric=runtime_metric,
            dataset=dataset,
            include_models=include_models,
            ignore_models=ignore_models,
            show_ci=show_ci,
        )
        prepared = None
    else:
        prepared = prepare_model_setting_plot_data(
            data,
            metric=metric,
            dataset=dataset,
            setting_run_ids=setting_run_ids,
            setting_labels=setting_labels,
            include_models=include_models,
            ignore_models=ignore_models,
            show_ci=show_ci,
            excluded_models_by_setting=excluded_models_by_setting,
        )
        frame = prepared.frame
        has_ci = prepared.has_ci

    frame = _scale_metric_columns(frame, metric, has_ci)
    if log_x and frame[runtime_metric].le(0).any():
        raise ValueError(f"{runtime_metric} must be strictly positive when log_x=True")
    styles = instance_plot_styles(frame)
    fig, ax = plt.subplots(figsize=(10, 6))
    ci_lower = f"{metric}_ci_lower"
    ci_upper = f"{metric}_ci_upper"
    right_edge_x = frame[runtime_metric].min() if invert_x else frame[runtime_metric].max()

    for _, row in frame.iterrows():
        instance = row["model_instance"]
        style, instance_label = styles[instance]
        color = style.color if prepared is None else prepared.setting_colors[int(row["setting_index"])]
        x = row[runtime_metric]
        y = row[metric]
        ax.scatter(x, y, color=color, marker=style.marker, s=58, zorder=3)
        if has_ci:
            draw_confidence_intervals(
                ax,
                [x],
                [y],
                [row[ci_lower]],
                [row[ci_upper]],
                color,
            )
        label_on_left = keep_labels_inside and x == right_edge_x
        ax.annotate(
            instance_label,
            (x, y),
            xytext=((-5 if label_on_left else 5), 4),
            textcoords="offset points",
            ha="right" if label_on_left else "left",
            fontsize=8,
        )
    if log_x:
        ax.set_xscale("log")
    if invert_x:
        ax.invert_xaxis()
    ax.set(xlabel=runtime_label(runtime_metric, log_x=log_x), ylabel=metric_label(metric))
    if x_axis_label:
        ax.set_xlabel(x_axis_label)
    ax.grid(alpha=0.3, which="both")
    if prepared is None:
        if title:
            ax.set_title(title, fontweight="bold", pad=10)
        fig.tight_layout()
        return fig

    legend_handles = [
        Line2D([], [], marker="o", linestyle="none", color=color, markersize=7, label=label)
        for label, color in zip(prepared.setting_labels, prepared.setting_colors, strict=True)
    ]
    apply_model_setting_header(
        fig,
        ax,
        title=title,
        legend_title=legend_title,
        legend_columns=min(len(prepared.setting_labels), 4),
        legend_handles=legend_handles,
    )
    print(format_model_setting_mapping(frame, prepared.setting_labels))
    return fig


def _scale_metric_columns(frame: pd.DataFrame, metric: str, has_ci: bool) -> pd.DataFrame:
    """Copy plot data and convert bounded classification scores to points."""
    scale = metric_scale(metric)
    if scale == 1:
        return frame
    columns = [metric]
    if has_ci:
        columns.extend((f"{metric}_ci_lower", f"{metric}_ci_upper"))
    scaled = frame.copy()
    scaled.loc[:, columns] = scaled[columns] * scale
    return scaled


def _validate_run_aggregation_options(
    run_aggregation: Literal["average"] | None,
    *,
    setting_run_ids: Mapping[str, str | Sequence[str]] | None,
    setting_labels: Sequence[str] | None,
    excluded_models_by_setting: Mapping[str, Sequence[str]] | None,
) -> None:
    if run_aggregation not in (None, "average"):
        raise ValueError(f"Unsupported run_aggregation {run_aggregation!r}; expected None or 'average'")
    if run_aggregation is None:
        return

    conflicts = [
        name
        for name, value in (
            ("setting_run_ids", setting_run_ids),
            ("setting_labels", setting_labels),
            ("excluded_models_by_setting", excluded_models_by_setting),
        )
        if value is not None
    ]
    if conflicts:
        raise ValueError(
            "run_aggregation='average' cannot be combined with setting-specific options: " + ", ".join(conflicts)
        )


def _prepare_averaged_run_plot_data(
    data: pd.DataFrame,
    *,
    metric: str,
    runtime_metric: str,
    dataset: str,
    include_models: Sequence[str] | None,
    ignore_models: Sequence[str] | None,
    show_ci: bool,
) -> tuple[pd.DataFrame, bool]:
    _validate_required_columns(
        data,
        ("scope", "statistic", "dataset", "model_name", metric, runtime_metric),
    )
    frame = data.loc[data["scope"].eq("test") & data["statistic"].eq("point") & data["dataset"].eq(dataset)].copy()
    if ignore_models:
        frame = frame.loc[~frame["model_name"].isin(ignore_models)]
    if include_models:
        frame = frame.loc[frame["model_name"].isin(include_models)]
    if frame.empty:
        raise ValueError(
            f"No scope='test', statistic='point' rows are available for dataset {dataset!r} and the model filters"
        )

    selected_columns = list(dict.fromkeys((metric, runtime_metric)))
    non_numeric = [column for column in selected_columns if not pd.api.types.is_numeric_dtype(frame[column])]
    if non_numeric:
        raise ValueError("Selected metric and runtime columns must be numeric: " + ", ".join(non_numeric))
    infinite_selected = [
        column for column in selected_columns if np.isinf(frame[column].dropna().to_numpy(dtype=float)).any()
    ]
    if infinite_selected:
        raise ValueError(
            "Selected metric and runtime columns must not contain infinite values: " + ", ".join(infinite_selected)
        )

    frame = frame.dropna(subset=selected_columns)
    if frame.empty:
        raise ValueError(f"No usable rows have both a finite {metric!r} metric and finite {runtime_metric!r} runtime")
    if "model_instance" not in frame:
        frame["model_instance"] = frame["model_name"]

    names_per_instance = frame.groupby("model_instance", sort=False, dropna=False)["model_name"].nunique()
    ambiguous_instances = names_per_instance[names_per_instance > 1]
    if not ambiguous_instances.empty:
        details = ", ".join(map(str, ambiguous_instances.index))
        raise ValueError(f"Each model_instance must identify exactly one model_name; ambiguous: {details}")

    ci_lower = f"{metric}_ci_lower"
    ci_upper = f"{metric}_ci_upper"
    has_ci = show_ci and ci_lower in frame.columns and ci_upper in frame.columns
    average_columns = selected_columns.copy()
    if has_ci:
        ci_columns = [ci_lower, ci_upper]
        non_numeric_ci = [column for column in ci_columns if not pd.api.types.is_numeric_dtype(frame[column])]
        if non_numeric_ci:
            raise ValueError("Confidence interval columns must be numeric: " + ", ".join(non_numeric_ci))
        infinite_ci = [column for column in ci_columns if np.isinf(frame[column].dropna().to_numpy(dtype=float)).any()]
        if infinite_ci:
            raise ValueError(
                "Confidence interval columns must not contain infinite values when show_ci=True: "
                + ", ".join(infinite_ci)
            )
        average_columns.extend(ci_columns)

    frame = frame.groupby(["model_name", "model_instance"], sort=False, as_index=False, dropna=False)[
        average_columns
    ].mean()
    identities = frame[["model_name", "model_instance"]]
    model_rank = {model: index for index, model in enumerate(ordered_models(identities["model_name"].tolist()))}
    frame["_model_order"] = frame["model_name"].map(model_rank)
    frame = frame.sort_values("_model_order", kind="stable").drop(columns="_model_order")
    return frame, has_ci


def _validate_required_columns(data: pd.DataFrame, required: Sequence[str]) -> None:
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required evaluation columns: {', '.join(missing)}")


def prepare_model_setting_plot_data(
    data: pd.DataFrame,
    *,
    metric: str,
    dataset: str,
    setting_run_ids: Mapping[str, str | Sequence[str]] | None = None,
    setting_labels: Sequence[str] | None = None,
    include_models: Sequence[str] | None = None,
    ignore_models: Sequence[str] | None = None,
    show_ci: bool = True,
    excluded_models_by_setting: Mapping[str, Sequence[str]] | None = None,
) -> ModelSettingPlotData:
    """Filter and identify rows for an ablation comparison.

    Explicit ``setting_run_ids`` assignment is preferred. Multiple run IDs for
    one setting may partition its model set across disjoint pipeline runs. The
    resulting matrix must contain at most one row per model instance/setting,
    and every selected model must be represented in each setting unless that
    pair is selected by a setting-specific exclusion. Occurrence-order
    assignment is retained only for compatibility with existing notebooks.
    Rows are never averaged.
    """
    frame = data.loc[data["scope"].eq("test") & data["statistic"].eq("point") & data["dataset"].eq(dataset)].copy()
    if frame.empty:
        raise ValueError(f"No scope='test', statistic='point' rows are available for dataset {dataset!r}")

    explicit_settings = _normalize_setting_run_ids(setting_run_ids)
    if explicit_settings is not None:
        if setting_labels is not None:
            raise ValueError("setting_labels cannot be combined with setting_run_ids; use the mapping keys as labels")
        _validate_pipeline_run_ids(frame)
        requested_ids = {run_id for _, run_ids in explicit_settings for run_id in run_ids}
        unknown_ids = requested_ids - set(frame["pipeline_mlflow_run_id"].astype(str))
        if unknown_ids:
            raise ValueError("Unknown pipeline_mlflow_run_id value(s): " + ", ".join(sorted(unknown_ids)))

    if ignore_models:
        frame = frame.loc[~frame["model_name"].isin(ignore_models)]
    if include_models:
        frame = frame.loc[frame["model_name"].isin(include_models)]
    if frame.empty:
        raise ValueError(
            f"No scope='test', statistic='point' rows are available for dataset {dataset!r} and the model filters"
        )
    if "model_instance" not in frame:
        frame["model_instance"] = frame["model_name"]

    if explicit_settings is None:
        _validate_provenance_columns(frame)
        frame["setting_index"] = frame.groupby("model_instance", sort=False).cumcount()
        counts = frame.groupby("model_instance", sort=False).size()
        if counts.nunique() != 1:
            details = ", ".join(f"{instance}={count}" for instance, count in counts.items())
            raise ValueError(
                "All compared model instances must have the same number of setting occurrences after filtering; "
                f"found {details}"
            )
        labels = _setting_labels(setting_labels, int(counts.iloc[0]))
        frame["setting_label"] = frame["setting_index"].map(dict(enumerate(labels)))
    else:
        labels = tuple(label for label, _ in explicit_settings)
        assignment = {
            run_id: (setting_index, label)
            for setting_index, (label, run_ids) in enumerate(explicit_settings)
            for run_id in run_ids
        }
        frame = frame.loc[frame["pipeline_mlflow_run_id"].astype(str).isin(assignment)].copy()
        frame["setting_index"] = (
            frame["pipeline_mlflow_run_id"]
            .astype(str)
            .map({run_id: identity[0] for run_id, identity in assignment.items()})
        )
        frame["setting_label"] = (
            frame["pipeline_mlflow_run_id"]
            .astype(str)
            .map({run_id: identity[1] for run_id, identity in assignment.items()})
        )
        _validate_explicit_setting_matrix(frame, labels, excluded_models_by_setting)

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
        setting_colors=_setting_colors(len(labels)),
        has_ci=has_ci,
    )


def format_model_setting_mapping(frame: pd.DataFrame, setting_labels: Sequence[str]) -> str:
    """Format setting provenance, using pipeline MLflow run IDs when present."""
    if "model_instance" not in frame:
        frame = frame.copy()
        frame["model_instance"] = frame["model_name"]
    identities = frame[["model_name", "model_instance"]].drop_duplicates()
    duplicate_models = set(identities.loc[identities["model_name"].duplicated(keep=False), "model_name"])
    use_run_ids = "pipeline_mlflow_run_id" in frame and frame["pipeline_mlflow_run_id"].notna().all()
    identity_column = "pipeline_mlflow_run_id" if use_run_ids else "pipeline_run_name"
    lines = ["Model setting mapping:"]
    for setting_index, setting_label in enumerate(setting_labels):
        setting_rows = frame.loc[frame["setting_index"].eq(setting_index)]
        pipeline_summaries = []
        for pipeline_identity in setting_rows[identity_column].drop_duplicates():
            run_rows = setting_rows.loc[setting_rows[identity_column].eq(pipeline_identity)]
            run_identities = run_rows[["model_name", "model_instance"]].drop_duplicates()
            model_rank = {
                model: index for index, model in enumerate(ordered_models(run_identities["model_name"].tolist()))
            }
            run_identities["_model_order"] = run_identities["model_name"].map(model_rank)
            run_identities = run_identities.sort_values("_model_order", kind="stable")
            model_names = ", ".join(
                str(row.model_instance) if row.model_name in duplicate_models else model_label(row.model_name)
                for row in run_identities.itertuples()
            )
            display_identity = str(pipeline_identity)
            if use_run_ids and "pipeline_run_name" in run_rows:
                names = run_rows["pipeline_run_name"].dropna().astype(str).drop_duplicates().tolist()
                if names:
                    display_identity += f" ({names[0]})"
            pipeline_summaries.append(f"{display_identity}: {model_names}")
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
    """Return explicit limits or padded limits around plotted values and CIs."""
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
        lower = max(lower, natural_bounds[0])
        upper = min(upper, natural_bounds[1])
    return lower, upper


def apply_model_setting_header(
    fig: Figure,
    ax: Axes,
    *,
    title: str | None,
    legend_title: str,
    legend_columns: int,
    legend_handles: Sequence[Line2D] | None = None,
) -> None:
    """Place an ablation title and setting legend above the plotting area."""
    if legend_handles is None:
        handles, labels = ax.get_legend_handles_labels()
    else:
        handles = list(legend_handles)
        labels = [handle.get_label() for handle in handles]
    if title:
        ax.set_title(title, fontweight="bold", pad=10)
    ax.legend(
        handles,
        labels,
        title=legend_title,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.13 if title else 1.02),
        ncol=legend_columns,
        frameon=False,
    )
    fig.tight_layout()


def _normalize_setting_run_ids(
    setting_run_ids: Mapping[str, str | Sequence[str]] | None,
) -> tuple[tuple[str, tuple[str, ...]], ...] | None:
    if setting_run_ids is None:
        return None
    if not setting_run_ids:
        raise ValueError("setting_run_ids must contain at least one setting")
    normalized = []
    seen: set[str] = set()
    for label, values in setting_run_ids.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError("setting_run_ids keys must be nonblank display labels")
        run_ids = (values,) if isinstance(values, str) else tuple(values)
        if not run_ids or not all(isinstance(run_id, str) and run_id.strip() for run_id in run_ids):
            raise ValueError(f"Run IDs for setting {label!r} must be one or more nonblank strings")
        duplicate_run_ids = [run_id for run_id in dict.fromkeys(run_ids) if run_ids.count(run_id) > 1]
        if duplicate_run_ids:
            raise ValueError(
                f"Pipeline run IDs must not repeat within setting {label!r}: " + ", ".join(duplicate_run_ids)
            )
        duplicates = seen.intersection(run_ids)
        if duplicates:
            raise ValueError("Pipeline run IDs cannot belong to multiple settings: " + ", ".join(sorted(duplicates)))
        seen.update(run_ids)
        normalized.append((label, tuple(run_ids)))
    return tuple(normalized)


def _validate_pipeline_run_ids(frame: pd.DataFrame) -> None:
    if "pipeline_mlflow_run_id" not in frame:
        raise ValueError("Missing required evaluation columns: pipeline_mlflow_run_id")
    valid = frame["pipeline_mlflow_run_id"].map(lambda value: isinstance(value, str) and bool(value.strip()))
    if not valid.all():
        raise ValueError("pipeline_mlflow_run_id must be a nonblank string; fix the selected evaluation rows")


def _validate_provenance_columns(frame: pd.DataFrame) -> None:
    if "pipeline_mlflow_run_id" in frame:
        _validate_pipeline_run_ids(frame)
        return
    if "pipeline_run_name" not in frame:
        raise ValueError("Missing required evaluation columns: pipeline_run_name")
    valid = frame["pipeline_run_name"].map(lambda value: isinstance(value, str) and bool(value.strip()))
    if not valid.all():
        raise ValueError("pipeline_run_name must be a nonblank string; fix the selected evaluation rows")


def _validate_explicit_setting_matrix(
    frame: pd.DataFrame,
    labels: Sequence[str],
    excluded_models_by_setting: Mapping[str, Sequence[str]] | None,
) -> None:
    if frame.empty:
        raise ValueError("No rows match setting_run_ids after applying model filters")
    counts = frame.groupby(["model_instance", "setting_index"], sort=False).size()
    duplicates = counts[counts > 1]
    if not duplicates.empty:
        details = ", ".join(f"{instance}/{labels[index]}={count}" for (instance, index), count in duplicates.items())
        raise ValueError(
            "Each model instance must have at most one row per explicit setting because setting plots do not "
            "aggregate runs; multiple run IDs may partition models but cannot repeat a model instance. "
            "Found duplicates: " + details
        )
    observed = set(counts.index)
    instances = frame["model_instance"].drop_duplicates().tolist()
    model_names_by_instance = (
        frame[["model_name", "model_instance"]]
        .drop_duplicates()
        .groupby("model_instance", sort=False)["model_name"]
        .agg(set)
        .to_dict()
    )
    excluded_selectors = {label: set(selectors) for label, selectors in (excluded_models_by_setting or {}).items()}
    missing = [
        f"{instance}/{label}"
        for instance in instances
        for index, label in enumerate(labels)
        if (instance, index) not in observed
        and instance not in excluded_selectors.get(label, set())
        and model_names_by_instance[instance].isdisjoint(excluded_selectors.get(label, set()))
    ]
    if missing:
        raise ValueError("Missing model/setting rows for explicit setting_run_ids: " + ", ".join(missing))


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
