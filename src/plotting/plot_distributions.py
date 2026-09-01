"""Minimal plotting for one numeric feature distribution."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from src.plotting.defaults import DATASET_COLORS, FEATURE_ALIASES, dataset_label, ordered_datasets, set_plot_style


def plot_feature_distribution(
    feature: str,
    *,
    mimic: pd.DataFrame | None = None,
    tudd: pd.DataFrame | None = None,
    alias: str | None = None,
) -> Figure:
    supplied = {"mimic": mimic, "tudd": tudd}
    datasets = [name for name in ordered_datasets(list(supplied)) if supplied[name] is not None]

    set_plot_style()
    figure, axis = plt.subplots(figsize=(8, 6))
    for name in datasets:
        sns.histplot(
            data=supplied[name],
            x=feature,
            bins=50,
            stat="density",
            kde=True,
            alpha=0.4,
            color=DATASET_COLORS[name],
            label=dataset_label(name),
            ax=axis,
        )

    axis.set(xlabel=alias or feature, ylabel="Density", yticks=[])
    axis.legend(title="Dataset")
    figure.tight_layout()
    return figure


def plot_feature_distributions(
    mimic: pd.DataFrame | None = None,
    tudd: pd.DataFrame | None = None,
    exclude_features: list[str] | None = None,
) -> dict[str, Figure]:
    if mimic is None:
        features = [col for col in tudd.columns if col not in (exclude_features or [])]
    else:
        features = [col for col in mimic.columns if col not in (exclude_features or [])]
    return {
        FEATURE_ALIASES.get(feature, feature): plot_feature_distribution(
            feature, mimic=mimic, tudd=tudd, alias=FEATURE_ALIASES.get(feature)
        )
        for feature in features
    }
