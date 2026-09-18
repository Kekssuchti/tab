from dataclasses import dataclass, field
from typing import Any

from src.schemas.base_schemas import TaskType
from src.schemas.dataset_schemas import DatasetSummary
from src.schemas.metrics import ClassificationMetrics, FinalTestMetrics, RegressionMetrics
from src.schemas.training_schemas import ScoringMethod, TuningMethod


@dataclass
class FoldRecord[MetricT: (ClassificationMetrics, RegressionMetrics)]:
    """Validation metrics for one candidate on one CV fold."""

    candidate_index: int
    fold_index: int
    metrics: MetricT
    time: float
    model_params: dict[str, Any]


@dataclass
class TuningRecord[MetricT: (ClassificationMetrics, RegressionMetrics)]:
    """Candidate selection plus the final model's held-out point metrics."""

    best_params: dict[str, Any]
    scoring: ScoringMethod
    final_test_metrics: FinalTestMetrics[MetricT]
    fold_results: list[FoldRecord[MetricT]] = field(default_factory=list)
    method: TuningMethod = "optuna"

    @property
    def total_time(self) -> float:
        return sum(fold.time for fold in self.fold_results)


@dataclass
class ModelTrainingResult[MetricT: (ClassificationMetrics, RegressionMetrics)]:
    """Result of fitting and evaluating one model."""

    model_name: str
    task_type: TaskType
    tuned: bool
    fit_time: float
    tuning_result: TuningRecord[MetricT] | None = None
    error: str | None = None
    failure_stage: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


type ClassificationModelTrainingResult = ModelTrainingResult[ClassificationMetrics]
type RegressionModelTrainingResult = ModelTrainingResult[RegressionMetrics]


@dataclass(frozen=True)
class TestSetEvaluationRecord[MetricT: (ClassificationMetrics, RegressionMetrics)]:
    """Point metrics and prediction time for one held-out dataset."""

    dataset_name: str
    metrics: MetricT
    predict_time: float


@dataclass(frozen=True)
class ModelEvaluationRecord[MetricT: (ClassificationMetrics, RegressionMetrics)]:
    """Point evaluation for one fully trained model."""

    model_name: str
    test_results: tuple[TestSetEvaluationRecord[MetricT], ...]
    final_test_metrics: FinalTestMetrics[MetricT]
    fit_time: float

    @property
    def total_time(self) -> float:
        return self.fit_time + sum(result.predict_time for result in self.test_results)

    @property
    def metrics_by_test_set(self) -> dict[str, MetricT]:
        return {result.dataset_name: result.metrics for result in self.test_results}


type ClassificationModelEvaluationRecord = ModelEvaluationRecord[ClassificationMetrics]
type RegressionModelEvaluationRecord = ModelEvaluationRecord[RegressionMetrics]


@dataclass(frozen=True)
class ModelRunRecord[MetricT: (ClassificationMetrics, RegressionMetrics)]:
    """Training and point-evaluation record for one model instance."""

    model_instance_id: str
    training_result: ModelTrainingResult[MetricT]
    evaluation: ModelEvaluationRecord[MetricT] | None

    @property
    def model_name(self) -> str:
        return self.training_result.model_name

    @property
    def succeeded(self) -> bool:
        return self.training_result.succeeded


type ClassificationModelRunRecord = ModelRunRecord[ClassificationMetrics]
type RegressionModelRunRecord = ModelRunRecord[RegressionMetrics]
type ModelRunFamily = ClassificationModelRunRecord | RegressionModelRunRecord


@dataclass(frozen=True)
class PipelineRunRecord:
    """Complete in-memory record for one pipeline run."""

    run_id: str
    dataset_summary: DatasetSummary
    model_runs: tuple[ModelRunFamily, ...]
    total_time: float

    @property
    def model_results(self) -> tuple[ClassificationModelEvaluationRecord | RegressionModelEvaluationRecord, ...]:
        return tuple(run.evaluation for run in self.model_runs if run.evaluation is not None)

    @property
    def training_results(self) -> tuple[ClassificationModelTrainingResult | RegressionModelTrainingResult, ...]:
        return tuple(run.training_result for run in self.model_runs)
