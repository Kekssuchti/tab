from typing import Any, Literal

from pydantic import Field

from src.config import config
from src.schemas.base_schemas import StrictConfig
from src.schemas.preprocessing_schemas import ImputerConfig, ScalerEncoderConfig
from src.utils.evaluation_utils import ScoringMethodCLS


class CrossValidationConfig(StrictConfig):
    """
    Configuration for cross-validation.

    ---
    Attributes:
        n_splits: int, default=5
            Number of cross-validation folds.

        shuffle: bool, default=True
            Whether to shuffle the data before splitting.

        random_state: int, default=config.seed
            Seed used for reproducible shuffling.
    """

    n_splits: int = Field(default=5, ge=2)
    shuffle: bool = True
    random_state: int = Field(default_factory=lambda: config.seed)


class OptunaConfig(StrictConfig):
    """
    Configuration for Optuna hyperparameter search.

    ---
    Attributes:
        n_trials: int, default=20
            Maximum number of optimization trials.

        sampler: {"tpe", "random"}, default="tpe"
            Optuna sampler used to propose trials.

        n_startup_trials: int, default=5
            Number of random startup trials for TPE.

        timeout: float or None, default=None
            Maximum optimization time in seconds.

        patience: int, default=10
            Number of trials to wait before early stopping.
    """

    n_trials: int = Field(default=20, ge=1)
    sampler: Literal["tpe", "random"] = "tpe"
    n_startup_trials: int = Field(default=5, ge=0)
    patience: int = Field(default=10, ge=0)
    timeout: float | None = Field(default=None, gt=0)


class TuningConfig(StrictConfig):
    """
    Configuration for model hyperparameter tuning.

    ---
    Attributes:
        method: {"grid", "optuna"}, default="optuna"
            Search algorithm used for tuning.

        search_space: str or None, default="default"
            Named registry search space used when no grid is supplied.

        grid: dict or None, default=None
            Explicit parameter grid overriding the registry search space.

        scoring: str, default="roc_auc"
            Metric used to choose the best candidate.

        cv: CrossValidationConfig, default=CrossValidationConfig()
            Cross-validation split settings.

        optuna: OptunaConfig, default=OptunaConfig()
            Optuna-specific search settings.
    """

    method: Literal["grid", "optuna"] = "optuna"
    search_space: str | None = "default"
    grid: dict[str, list[Any]] | None = None
    scoring: ScoringMethodCLS = "roc_auc"
    cv: CrossValidationConfig = Field(default_factory=CrossValidationConfig)
    optuna: OptunaConfig = Field(default_factory=OptunaConfig)


class ModelPreprocessingConfig(StrictConfig):
    """
    Optional preprocessing override for a single model.

    ---
    Attributes:
        imputer: ImputerConfig or None, default=None
            Model-specific imputation settings.

        scaler_encoder: ScalerEncoderConfig or None, default=None
            Model-specific scaling and encoding settings.
    """

    imputer: ImputerConfig | None = None
    scaler_encoder: ScalerEncoderConfig | None = None


class ModelConfig(StrictConfig):
    """
    Configuration for one model run.

    ---
    Attributes:
        name: str
            Registered model name.

        tuning: TuningConfig, default=TuningConfig()
            Hyperparameter tuning settings.

        preprocessing: ModelPreprocessingConfig or None, default=None
            Optional preprocessing override for this model.
    """

    name: str
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    preprocessing: ModelPreprocessingConfig | None = None
