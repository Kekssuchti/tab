from dataclasses import dataclass, field
from typing import Any

from pydantic import Field

from src.config import config
from src.schemas.base_schemas import StrictParams, TaskType
from src.schemas.preprocessing_schemas import ImputerParams, ScalerEncoderParams
from src.utils.evaluation_utils import (
    ClassificationMetrics,
    RegressionMetrics,
    ScoringMethodCLS,
    ScoringMethodREG,
    classification_score,
)


class CVParams(StrictParams):
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = Field(default_factory=lambda: config.seed)


class TuningParams(StrictParams):
    search_space: str | None = "default"
    grid: dict[str, list[Any]] | None = None
    scoring: ScoringMethodCLS = "roc_auc"
    cv: CVParams = Field(default_factory=CVParams)


class ModelPreprocessingParams(StrictParams):
    imputer: ImputerParams | None = None
    scaler_encoder: ScalerEncoderParams | None = None


class ModelParams(StrictParams):
    name: str
    task_type: TaskType = "classification"
    params: dict[str, Any] = Field(default_factory=dict)
    preprocessing: ModelPreprocessingParams | None = None
    tuning: TuningParams | None = None


@dataclass
class FoldResult:
    """Validation metrics for one candidate on one CV fold."""

    candidate_index: int
    fold_index: int
    metrics: ClassificationMetrics | RegressionMetrics
    time: float
    params: dict[str, Any]


@dataclass
class TuningCVResults:
    params: list[dict[str, Any]]
    mean_scores: list[float]
    std_scores: list[float]
    fold_scores: list[list[float]]
    fold_times: list[list[float]]
    mean_metrics: list[ClassificationMetrics] | list[RegressionMetrics]


@dataclass
class TuningResult:
    best_params: dict[str, Any]
    scoring: ScoringMethodCLS
    best_metrics: ClassificationMetrics | RegressionMetrics
    cv_results: TuningCVResults
    fold_results: list[FoldResult] = field(default_factory=list)

    @property
    def best_score(self) -> float:
        if isinstance(self.best_metrics, ClassificationMetrics):
            return classification_score(self.best_metrics, self.scoring)
        raise NotImplementedError("Regression tuning metrics are not implemented yet")

    @property
    def total_time(self) -> float:
        return sum(fold.time for fold in self.fold_results)


@dataclass
class ModelTrainingResult:
    """Result of fitting a single model, optionally after tuning."""

    model_name: str
    task_type: TaskType
    trained_model: Any
    tuned: bool
    fit_time: float
    training_metrics: ClassificationMetrics | RegressionMetrics | None = None
    tuning_result: TuningResult | None = None
