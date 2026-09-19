from dataclasses import dataclass

from src.schemas.base_schemas import TaskType
from src.schemas.dataset_schemas import DatasetBundle
from src.schemas.metrics import FinalTestMetrics
from src.utils.classification_metrics import classification_metrics
from src.utils.evaluation_utils import (
    classification_prediction_batch,
)
from src.utils.prediction_tables import BinaryTestPredictions, FinalTestPredictions


@dataclass(frozen=True)
class TrainedModelEvaluation:
    """Point metrics and optional binary prediction probabilities."""

    metrics: FinalTestMetrics
    test_predictions: FinalTestPredictions


def evaluate_trained_model(
    trained_model,
    task_type: TaskType,
    data: DatasetBundle,
) -> TrainedModelEvaluation:
    """Predict both held-out datasets once and compute point metrics."""

    mimic_prediction = trained_model.predict(data.test_mimic.X)
    tudd_prediction = trained_model.predict(data.test_tudd.X)

    if task_type != "classification":
        raise NotImplementedError("Only Classification supported")

    mimic_batch = classification_prediction_batch(mimic_prediction.values, data.test_mimic.y)
    tudd_batch = classification_prediction_batch(tudd_prediction.values, data.test_tudd.y)
    metrics = FinalTestMetrics(
        mimic_test=classification_metrics(mimic_batch),
        mimic_prediction_time=mimic_prediction.seconds,
        tudd_test=classification_metrics(tudd_batch),
        tudd_prediction_time=tudd_prediction.seconds,
    )
    predictions = FinalTestPredictions(
        mimic=BinaryTestPredictions(
            test_set_id=data.test_mimic.y.index.to_numpy(copy=True),
            y_true=mimic_batch.y_true,
            positive_class_probability=mimic_batch.probabilities[:, 1],
        ),
        tudd=BinaryTestPredictions(
            test_set_id=data.test_tudd.y.index.to_numpy(copy=True),
            y_true=tudd_batch.y_true,
            positive_class_probability=tudd_batch.probabilities[:, 1],
        ),
    )
    return TrainedModelEvaluation(metrics=metrics, test_predictions=predictions)
