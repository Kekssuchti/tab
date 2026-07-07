from src.schemas.base_schemas import TaskType
from src.schemas.dataset_schemas import DatasetBundle, XYDataset
from src.schemas.metrics import ClassificationMetrics, FinalTestMetrics
from src.schemas.run_records import TestSetEvaluationRecord
from src.utils.evaluation_utils import (
    evaluate_classification_predictions,
)


def evaluate_trained_model(
    trained_model,
    task_type: TaskType,
    data: DatasetBundle,
) -> FinalTestMetrics:
    """
    Runs standard evaluation on both mimic and tudd test set with full metrics

    Returns:
        FinalTestMetrics: metrics of tested model
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
) -> TestSetEvaluationRecord:
    if task_type != "classification":
        raise NotImplementedError("Regression evaluation is not implemented yet")

    predictions, predict_time = trained_model.predict(test_set.X)
    metrics = evaluate_classification_predictions(predictions, test_set.y.to_numpy())

    return TestSetEvaluationRecord(
        dataset_name=dataset_name,
        metrics=metrics,
        predict_time=predict_time,
    )
