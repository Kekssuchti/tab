"""Canonical plot and table defaults for evaluation results."""

from __future__ import annotations

from typing import NamedTuple, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex

# --- Metric defaults -------------------------------------------------------

METRIC_LABELS = {
    "roc_auc": "AUROC",
    "prc_auc": "AUPRC",
    "f1": "F1",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "sensitivity": "Sensitivity",
    "mae": "MAE",
    "mse": "MSE",
    "rmse": "RMSE",
    "r2": "R²",
}

LOWER_IS_BETTER_METRICS = frozenset({"mae", "mse", "rmse"})


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def metric_lower_is_better(metric: str) -> bool:
    """Return whether smaller values represent better performance."""
    return metric in LOWER_IS_BETTER_METRICS


# --- Dataset defaults ------------------------------------------------------

DATASET_COLORS = {"mimic": "#315C73", "tudd": "#D17A3F"}
DATASET_NAMES = {"mimic": "MIMIC-IV", "tudd": "EUH"}
DATASET_ORDER = ["tudd", "mimic"]


def dataset_label(dataset: str) -> str:
    """Return a shared display label, with a useful fallback."""
    return DATASET_NAMES.get(dataset, dataset.upper())


def ordered_datasets(dataset_names: Sequence[str]) -> list[str]:
    """Order datasets canonically, appending unknown datasets stably."""
    present = set(dataset_names)
    known = [name for name in DATASET_ORDER if name in present]
    unknown = [name for name in dict.fromkeys(dataset_names) if name not in DATASET_ORDER]
    return known + unknown

# --- Model defaults --------------------------------------------------------


class ModelStyle(NamedTuple):
    """Visual style of one model across all plots."""

    color: str
    linestyle: str
    marker: str
    label: str


MODEL_STYLES: dict[str, ModelStyle] = {
    # Classical ML baselines: muted gray/neutral colors, circle markers.
    "logistic-regression": ModelStyle("#757575", "-", "o", "LR"),
    "ebm": ModelStyle("#B3B3B3", "-", "o", "EBM"),
    "xgboost": ModelStyle("#222222", "-", "o", "XGBoost"),
    # Tabular foundation models. Strongest colors go to the key models, in
    # priority order: TabPFNv3, TabICLv2, TabFM, TabSwift, LimiX16M.
    "tabpfn-2.5": ModelStyle("#AEC7E8", "-", "^", "TabPFNv2.5"),
    "tabpfn-2.6": ModelStyle("#56B4E9", "-", "^", "TabPFNv2.6"),
    "tabpfn-3": ModelStyle("#0072B2", "-", "^", "TabPFNv3"),
    "tabicl-2": ModelStyle("#E69F00", "-", "^", "TabICLv2"),
    "limix-2m": ModelStyle("#C5B0D5", "-", "^", "LimiX2M"),
    "limix-16m": ModelStyle("#CC79A7", "-", "^", "LimiX16M"),
    "mitra": ModelStyle("#BCBD22", "-", "^", "Mitra"),
    "orion-msp": ModelStyle("#17BECF", "-", "^", "OrionMSP"),
    "orion-bix": ModelStyle("#98DF8A", "-", "^", "OrionBIX"),
    "tabswift": ModelStyle("#D55E00", "-", "^", "TabSwift"),
    "tabfm": ModelStyle("#009E73", "-", "^", "TabFM"),
}

MODEL_ORDER = list(MODEL_STYLES)
MODEL_LABELS = {name: style.label for name, style in MODEL_STYLES.items()}

_FALLBACK_COLORS = [to_hex(color) for color in plt.get_cmap("tab10")(np.linspace(0, 1, 10))]
_FALLBACK_LINESTYLES = ["-", "--", "-."]
_FALLBACK_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def ordered_models(model_names: Sequence[str]) -> list[str]:
    """Order models canonically, appending unknown models in first-seen order."""
    present = set(model_names)
    known_names = set(MODEL_ORDER)
    known = [name for name in MODEL_ORDER if name in present]
    unknown = [name for name in dict.fromkeys(model_names) if name not in known_names]
    return known + unknown


def model_label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def model_styles(model_names: Sequence[str]) -> dict[str, ModelStyle]:
    """Return the shared style for each model, in canonical order.

    Known models use their ``MODEL_STYLES`` entry; unknown models receive a
    deterministic fallback style so they remain distinguishable.
    """
    styles = {}
    for index, name in enumerate(ordered_models(model_names)):
        style = MODEL_STYLES.get(name)
        if style is None:
            style = ModelStyle(
                color=_FALLBACK_COLORS[index % len(_FALLBACK_COLORS)],
                linestyle=_FALLBACK_LINESTYLES[index % len(_FALLBACK_LINESTYLES)],
                marker=_FALLBACK_MARKERS[index % len(_FALLBACK_MARKERS)],
                label=name,
            )
        styles[name] = style
    return styles


# --- Figure style ----------------------------------------------------------


def set_plot_style() -> None:
    """Apply consistent matplotlib defaults used across result plots."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
