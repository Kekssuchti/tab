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
    available = [dataset for dataset in ordered_datasets(list(frames)) if feature in frames[dataset].columns]
    if not available:
        raise ValueError(f"Feature {feature!r} is absent from every selected dataset")
    for dataset in available:
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
