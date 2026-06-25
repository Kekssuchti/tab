from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.classes import pipeline as pipeline_module
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


class _ReleasablePredictor(_PredictsFromFirstColumn):
    active = 0
    peak = 0

    def __init__(self):
        self.released = False
        type(self).active += 1
        type(self).peak = max(type(self).peak, type(self).active)

    def release(self):
        if self.released:
            return
        self.released = True
        type(self).active -= 1


class _FailingReleasablePredictor(_ReleasablePredictor):
    active = 0
    peak = 0

    def predict(self, X_test):
        raise RuntimeError("bad evaluation")


def _test_set(labels, signal=None):
    y = pd.Series(labels)
    signal = labels if signal is None else signal
    return XYDataset(X=pd.DataFrame({"signal": signal}), y=y)


def test_pipeline_evaluates_mimic_and_tudd_test_sets_separately():
    bundle = DatasetBundle(
        train_data=_test_set([0, 1]),
        test_mimic=_test_set([0, 1, 0, 1]),
        test_tudd=_test_set([1, 0, 1, 0], signal=[0, 1, 0, 1]),
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
    assert result.metrics_by_test_set["tudd"].accuracy == 0.0
    assert result.metrics_by_test_set["mimic"].roc_auc == 1.0
    assert result.metrics_by_test_set["tudd"].roc_auc == 0.0
    assert result.final_test_metrics.mimic_test.accuracy == 1.0
    assert result.final_test_metrics.tudd_test.accuracy == 0.0
    assert result.final_test_metrics.mimic_minus_tudd.accuracy == 1.0
    assert result.final_test_metrics.mimic_minus_tudd.roc_auc == 1.0


def test_pipeline_releases_each_model_before_training_next(monkeypatch):
    _ReleasablePredictor.active = 0
    _ReleasablePredictor.peak = 0
    bundle = DatasetBundle(
        train_data=_test_set([0, 1]),
        test_mimic=_test_set([0, 1, 0, 1]),
        test_tudd=_test_set([1, 0, 1, 0], signal=[0, 1, 0, 1]),
    )

    class _FakeDataset:
        def get_dataset(self):
            return bundle

        def summarize(self, data):
            return SimpleNamespace()

    class _FakeTrainer:
        def __init__(self, params, default_imputer, default_scaler):
            self.params = params

        def validate_model_configs(self):
            pass

        def validate_training_data(self, X_train, y_train):
            pass

        def train_model(self, model_params, X_train, y_train):
            assert _ReleasablePredictor.active == 0
            return ModelTrainingResult(
                model_name=model_params.name,
                task_type="classification",
                trained_model=_ReleasablePredictor(),
                tuned=False,
                fit_time=0.2,
            )

    monkeypatch.setattr(pipeline_module, "Trainer", _FakeTrainer)

    pipeline = object.__new__(Pipeline)
    pipeline.dataset = _FakeDataset()
    pipeline.params = SimpleNamespace(
        run_id="run",
        dataset=SimpleNamespace(imputer=None, scaler_encoder=None),
        training=(
            SimpleNamespace(name="model-a"),
            SimpleNamespace(name="model-b"),
        ),
    )

    result = pipeline.run()

    assert [model.model_name for model in result.model_results] == [
        "model-a",
        "model-b",
    ]
    assert _ReleasablePredictor.peak == 1
    assert _ReleasablePredictor.active == 0
    assert [tr.trained_model for tr in result.training_results] == [None, None]


def test_pipeline_records_failed_model_and_continues(monkeypatch):
    bundle = DatasetBundle(
        train_data=_test_set([0, 1]),
        test_mimic=_test_set([0, 1, 0, 1]),
        test_tudd=_test_set([1, 0, 1, 0], signal=[0, 1, 0, 1]),
    )

    class _FakeDataset:
        def get_dataset(self):
            return bundle

        def summarize(self, data):
            return SimpleNamespace()

    class _FakeTrainer:
        def __init__(self, params, default_imputer, default_scaler):
            self.params = params

        def validate_model_configs(self):
            pass

        def validate_training_data(self, X_train, y_train):
            pass

        def train_model(self, model_params, X_train, y_train):
            if model_params.name == "model-a":
                raise ValueError("bad params")
            return ModelTrainingResult(
                model_name=model_params.name,
                task_type="classification",
                trained_model=_PredictsFromFirstColumn(),
                tuned=False,
                fit_time=0.2,
            )

    monkeypatch.setattr(pipeline_module, "Trainer", _FakeTrainer)

    pipeline = object.__new__(Pipeline)
    pipeline.dataset = _FakeDataset()
    pipeline.params = SimpleNamespace(
        run_id="run",
        dataset=SimpleNamespace(imputer=None, scaler_encoder=None),
        training=(
            SimpleNamespace(name="model-a", task_type="classification"),
            SimpleNamespace(name="model-b", task_type="classification"),
        ),
    )

    result = pipeline.run()

    assert [tr.model_name for tr in result.training_results] == ["model-a", "model-b"]
    assert result.training_results[0].failure_stage == "training"
    assert result.training_results[0].error == "ValueError: bad params"
    assert result.training_results[1].succeeded
    assert [model.model_name for model in result.model_results] == ["model-b"]


def test_pipeline_releases_model_after_evaluation_failure_and_continues(monkeypatch):
    _FailingReleasablePredictor.active = 0
    _FailingReleasablePredictor.peak = 0
    _ReleasablePredictor.active = 0
    _ReleasablePredictor.peak = 0
    bundle = DatasetBundle(
        train_data=_test_set([0, 1]),
        test_mimic=_test_set([0, 1, 0, 1]),
        test_tudd=_test_set([1, 0, 1, 0], signal=[0, 1, 0, 1]),
    )

    class _FakeDataset:
        def get_dataset(self):
            return bundle

        def summarize(self, data):
            return SimpleNamespace()

    class _FakeTrainer:
        def __init__(self, params, default_imputer, default_scaler):
            self.params = params

        def validate_model_configs(self):
            pass

        def train_model(self, model_params, X_train, y_train):
            model = (
                _FailingReleasablePredictor()
                if model_params.name == "model-a"
                else _ReleasablePredictor()
            )
            return ModelTrainingResult(
                model_name=model_params.name,
                task_type="classification",
                trained_model=model,
                tuned=False,
                fit_time=0.2,
            )

    monkeypatch.setattr(pipeline_module, "Trainer", _FakeTrainer)

    pipeline = object.__new__(Pipeline)
    pipeline.dataset = _FakeDataset()
    pipeline.params = SimpleNamespace(
        run_id="run",
        dataset=SimpleNamespace(imputer=None, scaler_encoder=None),
        training=(
            SimpleNamespace(name="model-a", task_type="classification"),
            SimpleNamespace(name="model-b", task_type="classification"),
        ),
    )

    result = pipeline.run()

    assert result.training_results[0].failure_stage == "evaluation"
    assert result.training_results[0].error == "RuntimeError: bad evaluation"
    assert result.training_results[0].trained_model is None
    assert result.training_results[1].succeeded
    assert [model.model_name for model in result.model_results] == ["model-b"]
    assert _FailingReleasablePredictor.active == 0
    assert _ReleasablePredictor.active == 0
