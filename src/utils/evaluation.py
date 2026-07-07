from dataclasses import dataclass

from src.schemas.dataset import DatasetBundle, XYDataset

from src.schemas.base_schemas import TaskType
from src.utils.evaluation_utils import (
    ClassificationMetrics,
    FinalTestMetrics,
    evaluate_classification_predictions,
)


@dataclass(frozen=True)
class TestSetEvaluationResult:
    dataset_name: str
    metrics: ClassificationMetrics
    predict_time: float


def evaluate_trained_model(
    trained_model,
    task_type: TaskType,
    data: DatasetBundle,
) -> FinalTestMetrics:
    """
    Runs standard evaluation on both mimic and tudd test set with full metrics

    Returns:
        ModelRunResult: metrics of tested model
    """
    test_results = (
        _evaluate_test_set("mimic", trained_model, task_type, data.test_mimic),
        _evaluate_test_set("tudd", trained_model, task_type, data.test_tudd),
    )

    return FinalTestMetrics(
        mimic_test=test_results[0].metrics,
        mimic_prediction_time=test_results[0].predict_time,
        tudd_test=test_results[1].metrics,
        tudd_prediction_time=test_results[1].predict_time,
    )


def _evaluate_test_set(
    dataset_name: str,
    trained_model,
    task_type: TaskType,
    test_set: XYDataset,
) -> TestSetEvaluationResult:
    if task_type != "classification":
        raise NotImplementedError("Regression evaluation is not implemented yet")

    predictions, predict_time = trained_model.predict(test_set.X)
    metrics = evaluate_classification_predictions(predictions, test_set.y.to_numpy())

    return TestSetEvaluationResult(
        dataset_name=dataset_name,
        metrics=metrics,
        predict_time=predict_time,
    )
