"""Shared metric/model selection and output layout for every figure family."""

import os
from collections.abc import Iterable
from pathlib import Path

from src.config import config

MODEL_SET = os.environ.get("PLOT_MODEL_SET", "all")
if MODEL_SET not in ("all", "main"):
    raise ValueError("PLOT_MODEL_SET must be 'all' or 'main'")

MAIN_MODELS = ("logistic-regression", "ebm", "xgboost", "tabicl-2", "tabfm", "tabpfn-3.5", "causilo")

METRIC_SET = os.environ.get("PLOT_METRIC", "auroc")
if METRIC_SET not in ("auroc", "auprc"):
    raise ValueError("PLOT_METRIC must be 'auroc' or 'auprc'")
PLOT_METRIC = {"auroc": "roc_auc", "auprc": "prc_auc"}[METRIC_SET]
PLOT_METRICS = (PLOT_METRIC,)


def excluded_models(model_names: Iterable[str]) -> tuple[str, ...]:
    """Exclude only available non-main models; 'all' excludes nothing."""
    return tuple(name for name in dict.fromkeys(model_names) if MODEL_SET == "main" and name not in MAIN_MODELS)


def plot_dir(family: str) -> Path:
    return config.dir_plots / family / METRIC_SET / MODEL_SET
