"""Shared factories for building typed schemas in tests.

Centralizes the metric and pipeline-result builders that were previously
copy-pasted across the observability, trainer, and pipeline tests.
"""
from dataclasses import replace

import numpy as np

from src.mlflow.observation import MetricLog
from src.schemas.dataset_schemas import (
    ClassificationTargetSummary,
    DatasetFileSummary,
    DatasetPartSummary,
    DatasetSummary,
    DataSplitConfig,
    RegressionTargetSummary,
    Target,
)
from src.schemas.metrics import (
    BootstrapClassificationMetrics,
    BootstrapFinalTestMetrics,
    BootstrapRegressionMetrics,
    ClassificationMetrics,
    FinalTestMetrics,
    RegressionMetrics,
)
from src.schemas.pipeline_schemas import MLflowConfig, PipelineConfig
from src.schemas.run_records import (
    FoldRecord,
    ModelEvaluationRecord,
    ModelRunRecord,
    ModelTrainingResult,
    PipelineRunRecord,
    TestSetEvaluationRecord,
    TuningRecord,
)
from src.schemas.training_schemas import ModelConfig


def classification_metrics(value: float = 1.0) -> ClassificationMetrics:
    return ClassificationMetrics(
        roc_auc=value,
        prc_auc=value,
        f1=value,
        accuracy=value,
        sensitivity=value,
        precision=value,
        confusion_matrix=np.array([[value, 0.0], [0.0, value]]),
        n_classes=2,
    )


def bootstrap_classification_metrics(
    value: float,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> BootstrapClassificationMetrics:
    lower = value if lower is None else lower
    upper = value if upper is None else upper
    return BootstrapClassificationMetrics(
        metrics=classification_metrics(value),
        ci_95_roc_auc_lower=lower,
        ci_95_roc_auc_upper=upper,
        ci_95_prc_auc_lower=lower,
        ci_95_prc_auc_upper=upper,
        ci_95_f1_lower=lower,
        ci_95_f1_upper=upper,
        ci_95_accuracy_lower=lower,
        ci_95_accuracy_upper=upper,
        ci_95_sensitivity_lower=lower,
        ci_95_sensitivity_upper=upper,
        ci_95_precision_lower=lower,
        ci_95_precision_upper=upper,
        n_bootstrap=5000,
    )


def bootstrap_regression_metrics(metrics: RegressionMetrics) -> BootstrapRegressionMetrics:
    return BootstrapRegressionMetrics(
        metrics=metrics,
        ci_95_r2_lower=metrics.r2 - 0.1,
        ci_95_r2_upper=metrics.r2 + 0.1,
        ci_95_mae_lower=metrics.mae - 0.1,
        ci_95_mae_upper=metrics.mae + 0.1,
        ci_95_mse_lower=metrics.mse - 0.1,
        ci_95_mse_upper=metrics.mse + 0.1,
        ci_95_rmse_lower=metrics.rmse - 0.1,
        ci_95_rmse_upper=metrics.rmse + 0.1,
        n_bootstrap=5000,
    )


def tuning_result() -> TuningRecord:
    return TuningRecord(
        best_params={"C": 1.0},
        scoring="accuracy",
        final_test_metrics=BootstrapFinalTestMetrics(
            mimic_test=bootstrap_classification_metrics(0.95, lower=0.9, upper=1.0),
            mimic_prediction_time=0.03,
            tudd_test=bootstrap_classification_metrics(0.95, lower=0.9, upper=1.0),
            tudd_prediction_time=0.04,
        ),
        fold_results=[
            FoldRecord(0, 0, classification_metrics(0.7), 0.01, {"C": 0.1}),
            FoldRecord(0, 1, classification_metrics(0.8), 0.02, {"C": 0.1}),
            FoldRecord(1, 0, classification_metrics(0.9), 0.03, {"C": 1.0}),
            FoldRecord(1, 1, classification_metrics(1.0), 0.04, {"C": 1.0}),
        ],
    )


def pipeline_config(
    tracking_uri: str,
    artifact_location: str | None = None,
    run_name: str | None = None,
    target: Target = "mortality",
    model_name: str = "logistic-regression",
    run_id: str = "test-pipeline-id",
) -> PipelineConfig:
    return PipelineConfig(
        run_id=run_id,
        dataset={
            "target": target,
            "random_state": 42,
            "train_size": 0.75,
            "train_on": (DataSplitConfig(dataset="mimic", fraction=1.0),),
        },
        training=(
            ModelConfig(
                name=model_name,
                preprocessing={
                    "imputer": {"imputation_method": "mean"},
                    "scaler_encoder": {"type": "standardization"},
                },
            ),
        ),
        mlflow=MLflowConfig(
            enabled=True,
            tracking_uri=tracking_uri,
            artifact_location=artifact_location,
            experiment_name="test-tab",
            run_name=run_name,
        ),
    )


def _dataset_summary(target: Target, *, data_files: tuple[DatasetFileSummary, ...]) -> DatasetSummary:
    if target == "LOS":
        train_summary = RegressionTargetSummary(count=8, mean=4.0, std=2.0, min=1.0, max=7.0)
        test_mimic_summary = RegressionTargetSummary(count=4, mean=3.0, std=1.0, min=2.0, max=4.0)
        test_tudd_summary = RegressionTargetSummary(count=4, mean=5.0, std=1.0, min=4.0, max=6.0)
    else:
        train_summary = ClassificationTargetSummary(class_balance={"0": 4, "1": 4})
        test_mimic_summary = ClassificationTargetSummary(class_balance={"0": 2, "1": 2})
        test_tudd_summary = ClassificationTargetSummary(class_balance={"0": 2, "1": 2})

    return DatasetSummary(
        target=target,
        train=DatasetPartSummary(row_count=8, target_summary=train_summary),
        test_mimic=DatasetPartSummary(row_count=4, target_summary=test_mimic_summary),
        test_tudd=DatasetPartSummary(row_count=4, target_summary=test_tudd_summary),
        data_files=data_files,
    )


def _classification_data_files() -> tuple[DatasetFileSummary, ...]:
    return (
        DatasetFileSummary(
            dataset_name="mimic",
            data_origin="mimic",
            file_name="mimic.csv",
            path="/tmp/mimic.csv",
            sha256="a" * 64,
        ),
        DatasetFileSummary(
            dataset_name="tudd",
            data_origin="tudd",
            file_name="tudd.csv",
            path="/tmp/tudd.csv",
            sha256="b" * 64,
        ),
    )


def pipeline_result(*, tuned: bool = False) -> PipelineRunRecord:
    metrics = classification_metrics()
    tuning = tuning_result() if tuned else None
    training_result = ModelTrainingResult(
        model_name="logistic-regression",
        task_type="classification",
        tuned=tuned,
        fit_time=0.2,
        tuning_result=tuning,
    )
    model_result = ModelEvaluationRecord(
        model_name="logistic-regression",
        fit_time=0.2,
        test_results=(
            TestSetEvaluationRecord("mimic", metrics, 0.03),
            TestSetEvaluationRecord("tudd", metrics, 0.04),
        ),
        final_test_metrics=FinalTestMetrics(
            mimic_test=metrics,
            mimic_prediction_time=0.0,
            tudd_test=metrics,
            tudd_prediction_time=0.0,
        ),
    )
    return PipelineRunRecord(
        run_id="test-pipeline-id",
        dataset_summary=_dataset_summary("mortality", data_files=_classification_data_files()),
        model_runs=(
            ModelRunRecord(
                model_instance_id="logistic-regression",
                training_result=training_result,
                evaluation=model_result,
            ),
        ),
        total_time=0.5,
    )


def failed_result() -> PipelineRunRecord:
    result = pipeline_result()
    failed_training_result = ModelTrainingResult(
        model_name="logistic-regression",
        task_type="classification",
        tuned=False,
        fit_time=0.1,
        error="ValueError: bad params",
        failure_stage="training",
    )
    return PipelineRunRecord(
        run_id=result.run_id,
        dataset_summary=result.dataset_summary,
        model_runs=(
            ModelRunRecord(
                model_instance_id="logistic-regression",
                training_result=failed_training_result,
                evaluation=None,
            ),
        ),
        total_time=0.2,
    )


def bootstrap_result() -> PipelineRunRecord:
    result = pipeline_result(tuned=True)
    bootstrap_metrics = bootstrap_classification_metrics(0.85, lower=0.75, upper=0.95)
    training_result = result.model_runs[0].training_result
    tuning = replace(
        training_result.tuning_result,
        final_test_metrics=BootstrapFinalTestMetrics(
            mimic_test=bootstrap_metrics,
            mimic_prediction_time=0.03,
            tudd_test=bootstrap_metrics,
            tudd_prediction_time=0.04,
        ),
    )
    model_run = replace(
        result.model_runs[0],
        training_result=replace(training_result, tuning_result=tuning),
    )
    return replace(result, model_runs=(model_run,))


def regression_result(*, tuned: bool = True) -> PipelineRunRecord:
    mimic_metrics = RegressionMetrics(r2=0.8, mae=0.2, mse=0.1, rmse=0.3)
    tudd_metrics = RegressionMetrics(r2=0.6, mae=0.4, mse=0.3, rmse=0.5)
    tuning = None
    if tuned:
        mimic_bootstrap = RegressionMetrics(r2=0.75, mae=0.25, mse=0.15, rmse=0.35)
        tudd_bootstrap = RegressionMetrics(r2=0.55, mae=0.45, mse=0.35, rmse=0.55)
        tuning = TuningRecord(
            best_params={"alpha": 0.5},
            scoring="rmse",
            final_test_metrics=BootstrapFinalTestMetrics(
                mimic_test=bootstrap_regression_metrics(mimic_bootstrap),
                mimic_prediction_time=0.03,
                tudd_test=bootstrap_regression_metrics(tudd_bootstrap),
                tudd_prediction_time=0.04,
            ),
            fold_results=[FoldRecord(0, 0, mimic_metrics, 0.01, {"alpha": 0.5})],
            method="grid",
        )
    training_result = ModelTrainingResult(
        model_name="linear-regression",
        task_type="regression",
        tuned=tuned,
        fit_time=0.2,
        tuning_result=tuning,
    )
    evaluation = ModelEvaluationRecord(
        model_name="linear-regression",
        fit_time=0.2,
        test_results=(
            TestSetEvaluationRecord("mimic", mimic_metrics, 0.03),
            TestSetEvaluationRecord("tudd", tudd_metrics, 0.04),
        ),
        final_test_metrics=FinalTestMetrics(
            mimic_test=mimic_metrics,
            mimic_prediction_time=0.03,
            tudd_test=tudd_metrics,
            tudd_prediction_time=0.04,
        ),
    )
    return PipelineRunRecord(
        run_id="regression-pipeline-id",
        dataset_summary=_dataset_summary("LOS", data_files=_classification_data_files()),
        model_runs=(ModelRunRecord("linear-regression", training_result, evaluation),),
        total_time=0.5,
    )


def metric_value(metrics: tuple[MetricLog, ...], name: str) -> float:
    return next(metric.value for metric in metrics if metric.name == name)
