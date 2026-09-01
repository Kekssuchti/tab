"""Canonical plot and table defaults for evaluation results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex

from src.plotting.scientific_figstyle import PALETTE, use_style

# Metric alias

POINT_SCALE_METRICS = frozenset({"roc_auc", "prc_auc", "f1", "accuracy", "precision", "sensitivity"})

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


# helper
def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def metric_scale(metric: str) -> int:
    """Return the display scale, leaving time metrics in their original units."""
    if "time" in metric.lower():
        return 1
    return 100 if metric in POINT_SCALE_METRICS else 1


# --- Dataset and task defaults --------------------------------------------

# Dataset and task colors are separate semantic namespaces. A plot comparing
# datasets uses DATASET_COLORS; one comparing tasks uses TASK_COLORS. Neither
# changes the stable model identity colors below.
DATASET_COLORS = {"mimic": PALETTE["red"], "tudd": PALETTE["blue"]}
DATASET_NAMES = {"mimic": "MIMIC-IV", "tudd": "EUH"}
DATASET_ORDER = ["tudd", "mimic"]

TASK_COLORS = {
    "mortality": PALETTE["orange"],
    "LOS7": PALETTE["blue"],
    "hours_to_readmit_72": PALETTE["green"],
}
TASK_NAMES = {
    "mortality": "Mortality",
    "LOS7": "LOS > 7 d",
    "hours_to_readmit_72": "72 h readmission",
}
TASK_ORDER = ["mortality", "LOS7", "hours_to_readmit_72"]


def dataset_label(dataset: str) -> str:
    """Return a shared display label, with a useful fallback."""
    return DATASET_NAMES.get(dataset, dataset.upper())


def ordered_datasets(dataset_names: Sequence[str]) -> list[str]:
    """Order datasets canonically, appending unknown datasets stably."""
    present = set(dataset_names)
    known = [name for name in DATASET_ORDER if name in present]
    unknown = [name for name in dict.fromkeys(dataset_names) if name not in DATASET_ORDER]
    return known + unknown


def task_label(task: str) -> str:
    """Return the shared display label for a prediction task."""
    return TASK_NAMES.get(task, task.replace("_", " ").title())


def ordered_tasks(task_names: Sequence[str]) -> list[str]:
    """Order tasks canonically, appending unknown tasks stably."""
    present = set(task_names)
    known = [name for name in TASK_ORDER if name in present]
    unknown = [name for name in dict.fromkeys(task_names) if name not in TASK_ORDER]
    return known + unknown


# --- Model defaults --------------------------------------------------------


class ModelStyle(NamedTuple):
    """Visual style of one model across all plots."""

    color: str
    linestyle: str
    marker: str
    label: str


BASELINE_MARKER = "o"
TFM_MARKER = "^"

MODEL_STYLES: dict[str, ModelStyle] = {
    # Classical ML baselines: neutral colors and circle markers.
    "logistic-regression": ModelStyle("#757575", "-", BASELINE_MARKER, "LR"),
    "ebm": ModelStyle("#B3B3B3", "-", BASELINE_MARKER, "EBM"),
    "xgboost": ModelStyle("#222222", "-", BASELINE_MARKER, "XGBoost"),
    # Retained foundation models receive the strongest accessible colors.
    "tabpfn-2.5": ModelStyle("#A6CEE3", "-", TFM_MARKER, "TabPFNv2.5"),
    "tabpfn-2.6": ModelStyle(PALETTE["cyan"], "-", TFM_MARKER, "TabPFNv2.6"),
    "tabpfn-3": ModelStyle(PALETTE["blue"], "-", TFM_MARKER, "TabPFNv3"),
    "tabicl-2": ModelStyle(PALETTE["orange"], "-", TFM_MARKER, "TabICLv2"),
    "tabfm": ModelStyle(PALETTE["green"], "-", TFM_MARKER, "TabFM"),
    "exaone": ModelStyle(PALETTE["purple"], "-", TFM_MARKER, "EXAONE"),
    # Legacy models keep stable but less prominent secondary colors.
    "limix-2m": ModelStyle("#D5C4D8", "-", TFM_MARKER, "LimiX2M"),
    "limix-16m": ModelStyle("#A98BB5", "-", TFM_MARKER, "LimiX16M"),
    "mitra": ModelStyle("#8A7A00", "-", TFM_MARKER, "Mitra"),
    "orion-msp": ModelStyle("#17BECF", "-", TFM_MARKER, "OrionMSP"),
    "orion-bix": ModelStyle("#74A66A", "-", TFM_MARKER, "OrionBIX"),
    "tabswift": ModelStyle(PALETTE["red"], "-", TFM_MARKER, "TabSwift"),
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
    """Apply the vendored publication style used by all result plots."""
    use_style()


# Feature aliases

FEATURE_ALIASES: dict[str, str] = {
    "mortality": "Mortality",
    "LOS": "Length of Stay",
    "hours_to_readmit": "Hours to Readmit",
    "Age": "Age",
    "Sex": "Sex",
    "Height+100%mean": "Height",
    "Weight+100%mean": "Weight",
    "Bmi+100%mean": "BMI",
    "ALAT+100%mean": "ALAT",
    "ASAT+100%mean": "ASAT",
    "Albumin+100%mean": "Albumin",
    "AnionGAP+100%mean": "Anion Gap",
    "Bilirubin+100%mean": "Bilirubin",
    "FiO2+100%mean": "Fraction of Inspired Oxygen",
    "GCST+100%mean": "Glasgow Coma Scale",
    "GLU+100%mean": "Glucose",
    "HCO3+100%mean": "Bicarbonate",
    "HR+100%mean": "Heart Rate",
    "Urea+100%mean": "Urea",
    "Hb+100%mean": "Hemoglobin",
    "Kalium+100%mean": "Potassium",
    "Kreatinin+100%mean": "Creatinine",
    "Lactate+100%mean": "Lactate",
    "Leukocyten+100%mean": "Leukocytes",
    "MBP+100%mean": "Mean Blood Pressure",
    "Natrium+100%mean": "Sodium",
    "PaCO2+100%mean": "Partial Pressure of CO2",
    "PaO2+100%mean": "Partial Pressure of O2",
    "Quick+100%mean": "Prothrombin Time",
    "RR+100%mean": "Respiratory Rate",
    "Temp+100%mean": "Temperature",
    "Thrombocyten+100%mean": "Thrombocyten",
    "Ph+100%mean": "Potential Hydrogen",
}

SMALL_FEATURE_NAMES: dict[str, str] = {
    "Fraction of Inspired Oxygen": "FiO2",
    "Glasgow Coma Scale": "GCST",
    "Partial Pressure of CO2": "PaCO2",
    "Partial Pressure of O2": "PaO2",
}
