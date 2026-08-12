"""Small helpers shared by evaluation plot families."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from src.plotting.defaults import ModelStyle, model_styles, ordered_models


def instance_plot_styles(frame: pd.DataFrame) -> dict[str, tuple[ModelStyle, str]]:
    """Map model instances to shared styles and collision-free labels."""
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


def draw_confidence_intervals(
    ax: Axes,
    positions: Sequence[float],
    values: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    color: str,
    *,
    horizontal: bool = False,
) -> None:
    """Draw confidence intervals for rows whose two bounds are present."""
    positions_array = np.asarray(positions, dtype=float)
    values_array = np.asarray(values, dtype=float)
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    present = ~np.isnan(lower_array) & ~np.isnan(upper_array)
    if present.any():
        errors = np.vstack(
            (
                values_array[present] - lower_array[present],
                upper_array[present] - values_array[present],
            )
        )
        x, y = (values_array, positions_array) if horizontal else (positions_array, values_array)
        ax.errorbar(
            x[present],
            y[present],
            **{"xerr" if horizontal else "yerr": errors},
            fmt="none",
            ecolor=color,
            capsize=2,
            alpha=0.65,
            zorder=2,
        )
