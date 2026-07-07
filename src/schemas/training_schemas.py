from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import Field

from src.config import config
from src.schemas.base_schemas import StrictConfig, TaskType
from src.schemas.preprocessing_schemas import ImputerConfig, ScalerEncoderConfig
from src.utils.evaluation_utils import (
    ClassificationMetrics,
    CVFinalTestMetrics,
    RegressionMetrics,
    ScoringMethodCLS,
    ScoringMethodREG,
)


class CrossValidationConfig(StrictConfig):
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = Field(default_factory=lambda: config.seed)


class OptunaConfig(StrictConfig):
    n_trials: int = Field(default=30, ge=1)
    sampler: Literal["tpe", "random"] = "tpe"
    n_startup_trials: int = Field(default=5, ge=0)
    timeout: float | None = Field(default=None, gt=0)


class TuningConfig(StrictConfig):
    method: Literal["grid", "optuna"] = "grid"
    search_space: str | None = "default"
    grid: dict[str, list[Any]] | None = None
    scoring: ScoringMethodCLS = "roc_auc"
    cv: CrossValidationConfig = Field(default_factory=CrossValidationConfig)
    optuna: OptunaConfig = Field(default_factory=OptunaConfig)


class ModelPreprocessingConfig(StrictConfig):
    imputer: ImputerConfig | None = None
    scaler_encoder: ScalerEncoderConfig | None = None


class ModelConfig(StrictConfig):
    name: str
    task_type: TaskType = "classification"
    params: dict[str, Any] = Field(default_factory=dict)
    preprocessing: ModelPreprocessingConfig | None = None
    tuning: TuningConfig | None = None


@dataclass
class FoldResult:
    """Validation metrics for one candidate on one CV fold."""

    candidate_index: int
    fold_index: int
    metrics: ClassificationMetrics | RegressionMetrics
    time: float
    params: dict[str, Any]


@dataclass
class TuningResult:
    best_params: dict[str, Any]
    scoring: ScoringMethodCLS | ScoringMethodREG
    test_metrics: CVFinalTestMetrics
    fold_results: list[FoldResult] = field(default_factory=list)
    method: Literal["grid", "optuna"] = "optuna"
    # TODO: if reg ever needed adjust metrics like cls

    @property
    def total_time(self) -> float:
        return sum(fold.time for fold in self.fold_results)


@dataclass
class ModelTrainingResult:
    """Result of fitting a single model, optionally after tuning."""

    model_name: str
    task_type: TaskType
    tuned: bool
    fit_time: float
    trained_model: Any | None = field(default=None, repr=False, compare=False)
    tuning_result: TuningResult | None = None
    error: str | None = None
    failure_stage: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None
