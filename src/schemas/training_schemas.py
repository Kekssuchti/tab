from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from src.schemas.base_schemas import StrictConfig, TaskType
from src.schemas.preprocessing_schemas import ImputerConfig, ScalerEncoderConfig


class _Scoring(StrEnum):
    """Scoring metric with the metadata required for candidate selection."""

    def __new__(cls, value: str, lower_is_better: bool):
        member = str.__new__(cls, value)
        member._value_ = value
        member._lower_is_better = lower_is_better
        return member

    @property
    def lower_is_better(self) -> bool:
        return self._lower_is_better

    @property
    def optimization_direction(self) -> Literal["minimize", "maximize"]:
        return "minimize" if self.lower_is_better else "maximize"

    @property
    def task_type(self) -> TaskType:
        raise NotImplementedError


class ClassificationScoring(_Scoring):
    ROC_AUC = ("roc_auc", False)
    F1 = ("f1", False)
    ACCURACY = ("accuracy", False)

    @property
    def task_type(self) -> TaskType:
        return "classification"


class RegressionScoring(_Scoring):
    R2 = ("r2", False)
    MAE = ("mae", True)
    MSE = ("mse", True)
    RMSE = ("rmse", True)

    @property
    def task_type(self) -> TaskType:
        return "regression"


ScoringMethod = ClassificationScoring | RegressionScoring
TuningMethod = Literal["grid", "optuna"]


def scoring_is_lower_better(scoring: str | ScoringMethod) -> bool:
    """Return whether a known tuning score is minimized.

    Unknown reporting metrics retain the existing larger-is-better default.
    """
    if isinstance(scoring, _Scoring):
        return scoring.lower_is_better

    for scoring_type in (ClassificationScoring, RegressionScoring):
        try:
            return scoring_type(scoring).lower_is_better
        except ValueError:
            continue
    return False


class CrossValidationConfig(StrictConfig):
    """
    Configuration for cross-validation.

    ---
    Attributes:
        n_splits: int, default=5
            Number of cross-validation folds.

        shuffle: bool, default=True
            Whether to shuffle the data before splitting.
    """

    n_splits: int = Field(default=5, ge=2)
    shuffle: bool = True


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

        scoring: ScoringMethod, default="roc_auc"
            Metric used to choose the best candidate.

        cv: CrossValidationConfig, default=CrossValidationConfig()
            Cross-validation split settings.

        optuna: OptunaConfig, default=OptunaConfig()
            Optuna-specific search settings.
    """

    method: TuningMethod = "optuna"
    search_space: str | None = "default"
    grid: dict[str, list[Any]] | None = None
    scoring: ScoringMethod = ClassificationScoring.ROC_AUC
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
