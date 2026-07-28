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
    metrics: ClassificationMetrics | RegressionMetrics
    time: float
    model_params: dict[str, Any]


@dataclass
class TuningRecord:
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
    final_test_metrics: AggregatedFinalTestMetrics
    fold_results: list[FoldRecord] = field(default_factory=list)
    method: Literal["grid", "optuna"] = "optuna"

    @property
    def total_time(self) -> float:
        return sum(fold.time for fold in self.fold_results)


@dataclass
class ModelTrainingResult:
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

        trained_model: Any or None, default=None
            Live model object, omitted from serialization and usually released.

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
    trained_model: Any | None = field(default=None, repr=False, compare=False)
    tuning_result: TuningRecord | None = None
    error: str | None = None
    failure_stage: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class TestSetEvaluationRecord:
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
    metrics: ClassificationMetrics | RegressionMetrics
    predict_time: float


@dataclass(frozen=True)
class ModelEvaluationRecord:
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
    test_results: tuple[TestSetEvaluationRecord, ...]
    final_test_metrics: FinalTestMetrics
    fit_time: float

    @property
    def total_time(self) -> float:
        return self.fit_time + sum(result.predict_time for result in self.test_results)

    @property
    def metrics_by_test_set(self) -> dict[str, ClassificationMetrics | RegressionMetrics]:
        return {result.dataset_name: result.metrics for result in self.test_results}


@dataclass(frozen=True)
class ModelRunRecord:
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
    model_runs: tuple[ModelRunRecord, ...]
    total_time: float

    @property
    def model_results(self) -> tuple[ModelEvaluationRecord, ...]:
        return tuple(run.evaluation for run in self.model_runs if run.evaluation is not None)

    @property
    def training_results(self) -> tuple[ModelTrainingResult, ...]:
        return tuple(run.training_result for run in self.model_runs)
