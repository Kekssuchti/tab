from types import SimpleNamespace

import numpy as np
import pandas as pd
from src.schemas.dataset_schemas import DatasetBundle, XYDataset

from src.classes import pipeline as pipeline_module
from src.classes.pipeline import Pipeline
from src.schemas.metrics import (
    BootstrapClassificationMetrics,
    BootstrapFinalTestMetrics,
    ClassificationMetrics,
)
from src.schemas.run_records import FoldRecord, ModelTrainingResult, TuningRecord


def _test_set(labels, signal=None):
    y = pd.Series(labels)
    signal = labels if signal is None else signal
    return XYDataset(X=pd.DataFrame({"signal": signal}), y=y)


def _metrics(value: float) -> ClassificationMetrics:
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


def _tuned_training_result(model_name: str) -> ModelTrainingResult:
    mimic = _bootstrap_metrics(1.0)
    tudd = _bootstrap_metrics(0.0)
    mean_metrics = _metrics(1.0)
    return ModelTrainingResult(
        model_name=model_name,
        task_type="classification",
        tuned=True,
        fit_time=0.2,
        tuning_result=TuningRecord(
            best_params={},
            scoring="accuracy",
            final_test_metrics=BootstrapFinalTestMetrics(
                mimic_test=mimic,
                mimic_prediction_time=0.1,
                tudd_test=tudd,
                tudd_prediction_time=0.2,
            ),
            fold_results=[FoldRecord(0, 0, mean_metrics, 0.0, {})],
        ),
    )


def _bootstrap_metrics(value: float) -> BootstrapClassificationMetrics:
    return BootstrapClassificationMetrics(
        metrics=_metrics(value),
        ci_95_roc_auc_lower=value,
        ci_95_roc_auc_upper=value,
        ci_95_prc_auc_lower=value,
        ci_95_prc_auc_upper=value,
        ci_95_f1_lower=value,
        ci_95_f1_upper=value,
        ci_95_accuracy_lower=value,
        ci_95_accuracy_upper=value,
        ci_95_sensitivity_lower=value,
        ci_95_sensitivity_upper=value,
        ci_95_precision_lower=value,
        ci_95_precision_upper=value,
        n_bootstrap=100,
    )


def test_pipeline_exposes_tuned_test_metrics_as_model_result():
    result = Pipeline._model_result_from_training_result(_tuned_training_result("fake-classifier"))

    assert result.model_name == "fake-classifier"
    assert np.isclose(result.total_time, 0.5)
    assert set(result.metrics_by_test_set) == {"mimic", "tudd"}
    assert result.metrics_by_test_set["mimic"].accuracy == 1.0
    assert result.metrics_by_test_set["tudd"].accuracy == 0.0
    assert result.metrics_by_test_set["mimic"].roc_auc == 1.0
    assert result.metrics_by_test_set["tudd"].roc_auc == 0.0
    assert result.final_test_metrics.mimic_test.accuracy == 1.0
    assert result.final_test_metrics.tudd_test.accuracy == 0.0
    assert result.final_test_metrics.mimic_minus_tudd.accuracy == 1.0
    assert result.final_test_metrics.mimic_minus_tudd.roc_auc == 1.0


def test_pipeline_records_do_not_own_live_models(monkeypatch):
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
        def __init__(self, task_type, default_imputer, default_scaler, log_transform_target):
            self.task_type = task_type

        def validate_model_configs(self):
            pass

        def validate_training_data(self, X_train, y_train):
            pass

        def train_evaluate_model(self, model_params, data):
            return ModelTrainingResult(
                model_name=model_params.name,
                task_type="classification",
                tuned=False,
                fit_time=0.2,
            )

    monkeypatch.setattr(pipeline_module, "Trainer", _FakeTrainer)

    pipeline = object.__new__(Pipeline)
    pipeline.dataset = _FakeDataset()
    pipeline.pipeline_config = SimpleNamespace(
        run_id="run",
        dataset=SimpleNamespace(
            target="mortality", imputer=None, scaler_encoder=None, log_transform_target=False
        ),
        training=(
            SimpleNamespace(name="model-a"),
            SimpleNamespace(name="model-b"),
        ),
    )

    result = pipeline.run()

    assert result.model_results == ()
    assert len(result.training_results) == 2
    assert all(not hasattr(tr, "trained_model") for tr in result.training_results)


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
        def __init__(self, task_type, default_imputer, default_scaler, log_transform_target):
            self.task_type = task_type

        def validate_model_configs(self):
            pass

        def validate_training_data(self, X_train, y_train):
            pass

        def train_evaluate_model(self, model_params, data):
            if model_params.name == "model-a":
                raise ValueError("bad params")
            return _tuned_training_result(model_params.name)

    monkeypatch.setattr(pipeline_module, "Trainer", _FakeTrainer)

    pipeline = object.__new__(Pipeline)
    pipeline.dataset = _FakeDataset()
    pipeline.pipeline_config = SimpleNamespace(
        run_id="run",
        dataset=SimpleNamespace(
            target="mortality", imputer=None, scaler_encoder=None, log_transform_target=False
        ),
        training=(
            SimpleNamespace(name="model-a"),
            SimpleNamespace(name="model-b"),
        ),
    )

    result = pipeline.run()

    assert [tr.model_name for tr in result.training_results] == ["model-a", "model-b"]
    assert result.training_results[0].failure_stage == "training_evaluation"
    assert result.training_results[0].error == "ValueError: bad params"
    assert result.training_results[0].task_type == "classification"
    assert result.training_results[1].succeeded
    assert [model.model_name for model in result.model_results] == ["model-b"]


def test_pipeline_records_evaluation_failure_and_continues(monkeypatch):
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
        def __init__(self, task_type, default_imputer, default_scaler, log_transform_target):
            self.task_type = task_type

        def validate_model_configs(self):
            pass

        def train_evaluate_model(self, model_params, data):
            if model_params.name == "model-a":
                raise RuntimeError("bad evaluation")
            return _tuned_training_result(model_params.name)

    monkeypatch.setattr(pipeline_module, "Trainer", _FakeTrainer)

    pipeline = object.__new__(Pipeline)
    pipeline.dataset = _FakeDataset()
    pipeline.pipeline_config = SimpleNamespace(
        run_id="run",
        dataset=SimpleNamespace(
            target="mortality", imputer=None, scaler_encoder=None, log_transform_target=False
        ),
        training=(
            SimpleNamespace(name="model-a"),
            SimpleNamespace(name="model-b"),
        ),
    )

    result = pipeline.run()

    assert result.training_results[0].failure_stage == "training_evaluation"
    assert result.training_results[0].error == "RuntimeError: bad evaluation"
    assert not hasattr(result.training_results[0], "trained_model")
    assert result.training_results[1].succeeded
    assert [model.model_name for model in result.model_results] == ["model-b"]
