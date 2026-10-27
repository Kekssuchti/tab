"""Mortality explanation experiments, with separate computation and plotting."""

# Experiment identity, shared so computation and plotting cannot drift apart.
TARGET = "mortality"
SOURCES = ("tudd", "mimic")
DEFAULT_MODELS = ("ebm", "xgboost", "tabpfn-3.5")
MODELS = (*DEFAULT_MODELS, "tabpfn-3.5-fast")
