import numpy as np
import pandas as pd

from src.classes.pipeline import Pipeline
from src.schemas.dataset_schemas import DatasetBundle, XYDataset
from src.schemas.training_schemas import ModelTrainingResult


class _PredictsFromFirstColumn:
    def predict(self, X_test):
        positive_probability = np.asarray(X_test)[:, 0]
        predictions = np.column_stack(
            [1 - positive_probability, positive_probability]
        )
        return predictions, 0.1


def _test_set(labels):
    y = pd.Series(labels)
    return XYDataset(X=pd.DataFrame({"signal": labels}), y=y)


def test_pipeline_evaluates_mimic_and_tudd_test_sets_separately():
    bundle = DatasetBundle(
        train_data=_test_set([0, 1]),
        test_mimic=_test_set([0, 1, 0, 1]),
        test_tudd=_test_set([1, 0, 1, 0]),
    )
    training_result = ModelTrainingResult(
        model_name="fake-classifier",
        task_type="classification",
        trained_model=_PredictsFromFirstColumn(),
        tuned=False,
        fit_time=0.2,
    )

    result = Pipeline._evaluate_trained_model(
        object.__new__(Pipeline), training_result, bundle
    )

    assert result.model_name == "fake-classifier"
    assert np.isclose(result.total_time, 0.4)
    assert set(result.metrics_by_test_set) == {"mimic", "tudd"}
    assert result.metrics_by_test_set["mimic"].accuracy == 1.0
    assert result.metrics_by_test_set["tudd"].accuracy == 1.0
    assert result.metrics_by_test_set["mimic"].roc_auc == 1.0
    assert result.metrics_by_test_set["tudd"].roc_auc == 1.0
