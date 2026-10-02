"""Low-level drawing primitives shared by publication figure scripts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from src.plotting.defaults import BASELINE_MARKER, ModelStyle, model_styles, ordered_models
from src.plotting.scientific_figstyle import MUTED


def instance_plot_styles(frame: pd.DataFrame) -> dict[str, tuple[ModelStyle, str]]:
    """Map model instances to canonical styles and collision-free labels."""
    required = {"model_instance", "model_name"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Missing model style columns: " + ", ".join(missing))
    unique = frame.drop_duplicates("model_instance").assign(
        model_instance=lambda rows: rows["model_instance"].astype(str),
        model_name=lambda rows: rows["model_name"].astype(str),
    )
    models = ordered_models(unique["model_name"].tolist())
    styles = model_styles(models)
    counts = unique["model_name"].value_counts()
    return {
        row.model_instance: (styles[model], styles[model].label if counts[model] == 1 else row.model_instance)
        for model in models
        for row in unique.loc[unique["model_name"].eq(model)].itertuples()
    }


def draw_model_forest(
    ax: Axes,
    rows: pd.DataFrame,
    model_instances: Sequence[str],
    styles: dict[str, tuple[ModelStyle, str]],
    *,
    scale: float = 1.0,
    show_ci: bool = True,
    show_model_labels: bool = True,
    shade_baselines: bool = True,
    baseline_band_alpha: float = 0.13,
    marker_size: float = 4.4,
    ci_line_width: float = 1.0,
    cap_size: float = 2.0,
) -> None:
    """Draw one horizontal model dot-and-whisker panel."""
    required = {"model_instance", "estimate", "lower", "upper"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError("Missing forest plot columns: " + ", ".join(missing))
    indexed = rows.assign(model_instance=rows["model_instance"].astype(str)).set_index("model_instance")
    missing_instances = [instance for instance in model_instances if instance not in indexed.index]
    if missing_instances:
        raise ValueError("Forest rows are missing model instances: " + ", ".join(missing_instances))

    positions = np.arange(len(model_instances), dtype=float)
    for y, instance in zip(positions, model_instances, strict=True):
        style, _ = styles[instance]
        if shade_baselines and style.marker == BASELINE_MARKER:
            ax.axhspan(y - 0.45, y + 0.45, color=MUTED, alpha=baseline_band_alpha, zorder=0)
        row = indexed.loc[instance]
        estimate = scale * float(row["estimate"])
        lower = scale * float(row["lower"])
        upper = scale * float(row["upper"])
        xerr = np.array([[estimate - lower], [upper - estimate]]) if show_ci else None
        ax.errorbar(
            estimate,
            y,
            xerr=xerr,
            fmt=style.marker,
            color=style.color,
            ecolor=style.color,
            elinewidth=ci_line_width,
            capsize=cap_size if show_ci else 0,
            capthick=0.8,
            markersize=marker_size,
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=3,
        )

    labels = [styles[instance][1] for instance in model_instances]
    ax.set_yticks(positions, labels)
    ax.set_ylim(len(model_instances) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0, labelleft=show_model_labels)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)


def interval_axis_limits(
    rows: pd.DataFrame,
    *,
    scale: float = 1.0,
    include_zero: bool = False,
    show_ci: bool = True,
    padding_fraction: float = 0.08,
    minimum_padding: float = 0.5,
) -> tuple[float, float]:
    """Return padded limits around estimates or confidence bounds."""
    columns = ["lower", "upper"] if show_ci else ["estimate"]
    values = scale * rows[columns].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot derive axis limits without finite plot values")
    lower = float(finite.min())
    upper = float(finite.max())
    if include_zero:
        lower = min(lower, 0.0)
        upper = max(upper, 0.0)
    span = upper - lower
    padding = padding_fraction * span if span > 1e-9 else minimum_padding
    return lower - padding, upper + padding


def short_count(value: int) -> str:
    """Format a sample count compactly for an axis tick."""
    if value < 1_000:
        return f"{value}"
    thousands = f"{value / 1_000:.1f}".rstrip("0").rstrip(".")
    return f"{thousands}k"


def log_sample_ticks(
    sample_sizes: Sequence[int],
    maximum: int,
    *,
    min_log_gap: float = 1.2,
) -> tuple[int, ...]:
    """Choose legible ticks for a logarithmic sample-count axis.

    Measured counts are rarely evenly spaced, so evenly spaced *indices* can put
    two labels on top of each other (25,600 next to 44,454, say). Ticks are
    therefore chosen as measured counts that are at least min_log_gap apart in
    log space, the smallest and largest count are always kept, and the requested
    maximum widens the gap further when the measured range is wide.
    """
    if maximum < 2:
        raise ValueError("maximum must be at least two")
    ordered = tuple(sorted({int(value) for value in sample_sizes}))
    if len(ordered) <= maximum:
        return ordered
    logs = np.log2(np.asarray(ordered, dtype=float))
    gap = max(min_log_gap, float(logs[-1] - logs[0]) / (maximum - 1))
    chosen = [0]
    previous = logs[0]
    for index in range(1, len(ordered) - 1):
        if logs[index] - previous >= gap and logs[-1] - logs[index] >= gap:
            chosen.append(index)
            previous = logs[index]
    chosen.append(len(ordered) - 1)
    return tuple(ordered[index] for index in chosen)


def sample_ticks(sample_sizes: Sequence[int], maximum: int) -> tuple[int, ...]:
    """Choose at most `maximum` evenly spaced ticks from measured counts."""
    if maximum < 2:
        raise ValueError("maximum must be at least two")
    if len(sample_sizes) <= maximum:
        return tuple(sample_sizes)
    indices = np.linspace(0, len(sample_sizes) - 1, maximum).round().astype(int)
    return tuple(sample_sizes[index] for index in np.unique(indices))
