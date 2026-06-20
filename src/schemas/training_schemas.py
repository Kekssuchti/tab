from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import Field

from src.config import config
from src.schemas.base_schemas import StrictParams

ScoringMethod = Literal[
    "roc_auc", "prc_auc", "f1", "accuracy", "sensitivity", "precision"
]


class CVParams(StrictParams):
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = Field(default_factory=lambda: config.seed)


class HPOParams(StrictParams):
    search_grid: dict[str, Any]
    # random or maybe even optuna / baysian style
    search_method: Literal["grid"] = "grid"
    scoring: ScoringMethod = "roc_auc"
    cv: CVParams = Field(default_factory=CVParams)


class ModelParams(StrictParams):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    task_type: Literal["classification", "regression"] = "classification"
    optimize_hyperparameters: bool = False
    hyperparameter_optimization_params: HPOParams | None = None


class TrainingParams(StrictParams):
    models: tuple[ModelParams, ...] = (ModelParams(name="tabpfn-3"),)


@dataclass
class FoldResult:
    """Metrics for a single CV fold."""

    fold_index: int
    train_score: float
    test_score: float


@dataclass
class HPOResult:
    best_params: dict[str, Any]
    best_score: float
    scoring: ScoringMethod
    cv_results: dict[str, Any]
    fold_results: list[FoldResult] = field(default_factory=list)


@dataclass
class ModelTrainingResult:
    """Result of training a single model (with or without HPO)."""

    model_name: str
    task_type: str
    trained_model: Any
    optimized_hyperparameters: bool
    fit_time: float
    hpo_result: HPOResult | None = None
