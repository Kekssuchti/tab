"""Low-level feature-distribution drawing helpers."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from src.plotting.defaults import DATASET_COLORS, dataset_label, ordered_datasets


def draw_feature_distribution(
    ax: Axes,
    frames: Mapping[str, pd.DataFrame],
    feature: str,
    *,
    bins: int,
    kde: bool,
    alpha: float,
) -> None:
    """Draw density-normalized feature histograms on an existing axes."""
    missing = [dataset for dataset, frame in frames.items() if feature not in frame.columns]
    if missing:
        raise ValueError(f"Feature {feature!r} is missing from datasets: {', '.join(missing)}")
    for dataset in ordered_datasets(list(frames)):
        sns.histplot(
            data=frames[dataset],
            x=feature,
            bins=bins,
            stat="density",
            kde=kde,
            alpha=alpha,
            color=DATASET_COLORS.get(dataset),
            label=dataset_label(dataset),
            ax=ax,
        )
