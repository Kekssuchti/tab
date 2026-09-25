from dataclasses import dataclass

import numpy as np

from src.schemas.base_schemas import TaskType
from src.schemas.dataset_schemas import DatasetBundle, XYDataset
from src.schemas.metrics import ClassificationMetrics, FinalTestMetrics
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
    """Evaluate either the retriever cohort or the two standard holdouts."""

    if task_type != "classification":
        raise NotImplementedError("Only Classification supported")

    if data.test_retriever is not None:
        retriever_prediction = trained_model.predict(data.test_retriever.X)
        retriever_batch = classification_prediction_batch(retriever_prediction.values, data.test_retriever.y)
        return TrainedModelEvaluation(
            metrics=FinalTestMetrics(
                mimic_test=_placeholder_classification_metrics(),
                mimic_prediction_time=0.0,
                tudd_test=_placeholder_classification_metrics(),
                tudd_prediction_time=0.0,
                retriever_test=classification_metrics(retriever_batch),
                retriever_prediction_time=retriever_prediction.seconds,
            ),
            test_predictions=FinalTestPredictions(
                mimic=_placeholder_binary_predictions(),
                tudd=_placeholder_binary_predictions(),
                retriever=_binary_test_predictions(data.test_retriever, retriever_batch),
            ),
        )

    mimic_prediction = trained_model.predict(data.test_mimic.X)
    tudd_prediction = trained_model.predict(data.test_tudd.X)
    mimic_batch = classification_prediction_batch(mimic_prediction.values, data.test_mimic.y)
    tudd_batch = classification_prediction_batch(tudd_prediction.values, data.test_tudd.y)
    return TrainedModelEvaluation(
        metrics=FinalTestMetrics(
            mimic_test=classification_metrics(mimic_batch),
            mimic_prediction_time=mimic_prediction.seconds,
            tudd_test=classification_metrics(tudd_batch),
            tudd_prediction_time=tudd_prediction.seconds,
        ),
        test_predictions=FinalTestPredictions(
            mimic=_binary_test_predictions(data.test_mimic, mimic_batch),
            tudd=_binary_test_predictions(data.test_tudd, tudd_batch),
        ),
    )


def _binary_test_predictions(test_set: XYDataset, prediction_batch) -> BinaryTestPredictions:
    return BinaryTestPredictions(
        test_set_id=test_set.y.index.to_numpy(copy=True),
        y_true=prediction_batch.y_true,
        positive_class_probability=prediction_batch.probabilities[:, 1],
    )


def _placeholder_classification_metrics() -> ClassificationMetrics:
    """Preserve the fixed metric contract without reporting unevaluated cohorts."""

    return ClassificationMetrics(
        roc_auc=None,
        prc_auc=None,
        f1=0.0,
        accuracy=0.0,
        sensitivity=0.0,
        precision=0.0,
        n_classes=0,
        confusion_matrix=None,
    )


def _placeholder_binary_predictions() -> BinaryTestPredictions:
    """Preserve required prediction fields while marking the cohort absent."""

    return BinaryTestPredictions(
        test_set_id=np.array([], dtype=np.int64),
        y_true=np.array([], dtype=np.int8),
        positive_class_probability=np.array([], dtype=float),
    )
