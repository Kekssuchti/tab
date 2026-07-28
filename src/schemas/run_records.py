from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeAlias, TypeVar

from src.schemas.base_schemas import TaskType
from src.schemas.dataset_schemas import DatasetSummary
from src.schemas.metrics import (
    AggregatedFinalTestMetrics,
    ClassificationMetricsAggregate,
    ClassificationMetrics,
    FinalTestMetrics,
    RegressionMetricsAggregate,
    RegressionMetrics,
)
from src.utils.evaluation_utils import ScoringMethodCLS, ScoringMethodREG


MetricT = TypeVar("MetricT", ClassificationMetrics, RegressionMetrics)
AggregateMetricT = TypeVar("AggregateMetricT", ClassificationMetricsAggregate, RegressionMetricsAggregate)


@dataclass
class FoldRecord(Generic[MetricT]):
    """
    Validation metrics for one candidate on one CV fold.

    ---
    Attributes:
        candidate_index: int
            Index of the tuned candidate.

        fold_index: int
            Cross-validation fold index.

        metrics: ClassificationMetrics or RegressionMetrics
            Metrics measured on the validation fold.

        time: float
            Fit and validation time for the fold, in seconds.

        model_params: dict
            Model parameters used by the candidate.
    """

    candidate_index: int
    fold_index: int
    metrics: MetricT
    time: float
    model_params: dict[str, Any]


ClassificationFoldRecord: TypeAlias = FoldRecord[ClassificationMetrics]
RegressionFoldRecord: TypeAlias = FoldRecord[RegressionMetrics]


@dataclass
class TuningRecord(Generic[MetricT, AggregateMetricT]):
    """
    Result of tuning one model.

    ---
    Attributes:
        best_params: dict
            Parameters selected by tuning.

        scoring: str
            Metric used to rank candidates.

        final_test_metrics: AggregatedFinalTestMetrics
            Aggregated final-test metrics from tuned folds.

        fold_results: list of FoldRecord, default=[]
            Per-fold validation records.

        method: {"grid", "optuna"}, default="optuna"
            Tuning method used.
    """

    best_params: dict[str, Any]
    scoring: ScoringMethodCLS | ScoringMethodREG
    final_test_metrics: AggregatedFinalTestMetrics[AggregateMetricT]
    fold_results: list[FoldRecord[MetricT]] = field(default_factory=list)
    method: Literal["grid", "optuna"] = "optuna"

    @property
    def total_time(self) -> float:
        return sum(fold.time for fold in self.fold_results)


ClassificationTuningRecord: TypeAlias = TuningRecord[ClassificationMetrics, ClassificationMetricsAggregate]
RegressionTuningRecord: TypeAlias = TuningRecord[RegressionMetrics, RegressionMetricsAggregate]


@dataclass
class ModelTrainingResult(Generic[MetricT, AggregateMetricT]):
    """
    Result of fitting a single model, optionally after tuning.

    ---
    Attributes:
        model_name: str
            Registered model name.

        task_type: {"classification", "regression"}
            Prediction task type.

        tuned: bool
            Whether hyperparameter tuning was run.

        fit_time: float
            Final fit time in seconds.

        tuning_result: TuningRecord or None, default=None
            Tuning result when tuning was run.

        error: str or None, default=None
            Error message when training failed.

        failure_stage: str or None, default=None
            Pipeline stage where failure occurred.
    """

    model_name: str
    task_type: TaskType
    tuned: bool
    fit_time: float
    tuning_result: TuningRecord[MetricT, AggregateMetricT] | None = None
    error: str | None = None
    failure_stage: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


ClassificationModelTrainingResult: TypeAlias = ModelTrainingResult[
    ClassificationMetrics, ClassificationMetricsAggregate
]
RegressionModelTrainingResult: TypeAlias = ModelTrainingResult[RegressionMetrics, RegressionMetricsAggregate]


@dataclass(frozen=True)
class TestSetEvaluationRecord(Generic[MetricT]):
    """
    Evaluation result for one held-out test set.

    ---
    Attributes:
        dataset_name: str
            Name of the evaluated test dataset.

        metrics: ClassificationMetrics
            Classification metrics for the test set.

        predict_time: float
            Prediction time in seconds.
    """

    dataset_name: str
    metrics: MetricT
    predict_time: float


ClassificationTestSetEvaluationRecord: TypeAlias = TestSetEvaluationRecord[ClassificationMetrics]
RegressionTestSetEvaluationRecord: TypeAlias = TestSetEvaluationRecord[RegressionMetrics]


@dataclass(frozen=True)
class ModelEvaluationRecord(Generic[MetricT]):
    """
    Evaluation result for one trained model.

    ---
    Attributes:
        model_name: str
            Registered model name.

        test_results: tuple of TestSetEvaluationRecord
            Per-test-set evaluation records.

        final_test_metrics: FinalTestMetrics
            Combined MIMIC and TUDD final-test metrics.

        fit_time: float
            Final model fit time in seconds.
    """

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


ClassificationModelEvaluationRecord: TypeAlias = ModelEvaluationRecord[ClassificationMetrics]
RegressionModelEvaluationRecord: TypeAlias = ModelEvaluationRecord[RegressionMetrics]


@dataclass(frozen=True)
class ModelRunRecord(Generic[MetricT, AggregateMetricT]):
    """
    Training and evaluation record for one model instance.

    ---
    Attributes:
        model_instance_id: str
            Unique identifier for this model within the run.

        training_result: ModelTrainingResult
            Training result for the model.

        evaluation: ModelEvaluationRecord or None
            Evaluation result, or None when training failed.
    """

    model_instance_id: str
    training_result: ModelTrainingResult[MetricT, AggregateMetricT]
    evaluation: ModelEvaluationRecord[MetricT] | None

    @property
    def model_name(self) -> str:
        return self.training_result.model_name

    @property
    def succeeded(self) -> bool:
        return self.training_result.succeeded


ClassificationModelRunRecord: TypeAlias = ModelRunRecord[ClassificationMetrics, ClassificationMetricsAggregate]
RegressionModelRunRecord: TypeAlias = ModelRunRecord[RegressionMetrics, RegressionMetricsAggregate]
ModelRunFamily: TypeAlias = ClassificationModelRunRecord | RegressionModelRunRecord


@dataclass(frozen=True)
class PipelineRunRecord:
    """
    Complete result record for one pipeline run.

    ---
    Attributes:
        run_id: str
            Pipeline run identifier.

        dataset_summary: DatasetSummary
            Summary of datasets used by the run.

        model_runs: tuple of ModelRunRecord
            Training and evaluation records for all model instances.

        total_time: float
            Total pipeline runtime in seconds.
    """

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
