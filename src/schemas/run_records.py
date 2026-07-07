from dataclasses import dataclass, field
from typing import Any, Literal

from src.schemas.base_schemas import TaskType
from src.schemas.dataset_schemas import DatasetSummary
from src.schemas.metrics import (
    AggregatedFinalTestMetrics,
    ClassificationMetrics,
    FinalTestMetrics,
    RegressionMetrics,
)
from src.utils.evaluation_utils import ScoringMethodCLS, ScoringMethodREG


@dataclass
class FoldRecord:
    """Validation metrics for one candidate on one CV fold."""

    candidate_index: int
    fold_index: int
    metrics: ClassificationMetrics | RegressionMetrics
    time: float
    model_params: dict[str, Any]


@dataclass
class TuningRecord:
    best_params: dict[str, Any]
    scoring: ScoringMethodCLS | ScoringMethodREG
    final_test_metrics: AggregatedFinalTestMetrics
    fold_results: list[FoldRecord] = field(default_factory=list)
    method: Literal["grid", "optuna"] = "optuna"

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
    tuning_result: TuningRecord | None = None
    error: str | None = None
    failure_stage: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class TestSetEvaluationRecord:
    dataset_name: str
    metrics: ClassificationMetrics
    predict_time: float


@dataclass(frozen=True)
class ModelEvaluationRecord:
    model_name: str
    test_results: tuple[TestSetEvaluationRecord, ...]
    final_test_metrics: FinalTestMetrics
    fit_time: float

    @property
    def total_time(self) -> float:
        return self.fit_time + sum(result.predict_time for result in self.test_results)

    @property
    def metrics_by_test_set(self) -> dict[str, ClassificationMetrics]:
        return {result.dataset_name: result.metrics for result in self.test_results}


@dataclass(frozen=True)
class ModelRunRecord:
    model_instance_id: str
    training_result: ModelTrainingResult
    evaluation: ModelEvaluationRecord | None

    @property
    def model_name(self) -> str:
        return self.training_result.model_name

    @property
    def succeeded(self) -> bool:
        return self.training_result.succeeded


@dataclass(frozen=True)
class PipelineRunRecord:
    run_id: str
    dataset_summary: DatasetSummary
    model_runs: tuple[ModelRunRecord, ...]
    total_time: float

    @property
    def model_results(self) -> tuple[ModelEvaluationRecord, ...]:
        return tuple(run.evaluation for run in self.model_runs if run.evaluation is not None)

    @property
    def training_results(self) -> tuple[ModelTrainingResult, ...]:
        return tuple(run.training_result for run in self.model_runs)
