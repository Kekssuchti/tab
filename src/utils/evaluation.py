from src.schemas.base_schemas import TaskType
from src.schemas.dataset_schemas import DatasetBundle, XYDataset
from src.schemas.metrics import FinalTestMetrics
from src.schemas.run_records import TestSetEvaluationRecord
from src.utils.evaluation_utils import (
    evaluate_classification_predictions,
    evaluate_regression_predictions,
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
    mimic_test_results = _evaluate_test_set("mimic", trained_model, task_type, data.test_mimic)
    tudd_test_results = _evaluate_test_set("tudd", trained_model, task_type, data.test_tudd)

    return FinalTestMetrics(
        mimic_test=mimic_test_results.metrics,
        mimic_prediction_time=mimic_test_results.predict_time,
        tudd_test=tudd_test_results.metrics,
        tudd_prediction_time=tudd_test_results.predict_time,
    )


def _evaluate_test_set(
    dataset_name: str,
    trained_model,
    task_type: TaskType,
    test_set: XYDataset,
) -> TestSetEvaluationRecord:
    prediction = trained_model.predict(test_set.X)
    if task_type == "classification":
        metrics = evaluate_classification_predictions(prediction.values, test_set.y.to_numpy())
    else:
        metrics = evaluate_regression_predictions(prediction.values, test_set.y.to_numpy())

    return TestSetEvaluationRecord(
        dataset_name=dataset_name,
        metrics=metrics,
        predict_time=prediction.seconds,
    )
