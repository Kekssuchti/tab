from typing import Any, Literal

from pydantic import Field

from src.config import config
from src.schemas.base_schemas import StrictConfig, TaskType
from src.schemas.preprocessing_schemas import ImputerConfig, ScalerEncoderConfig
from src.utils.evaluation_utils import ScoringMethodCLS


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
