from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.classes import pipeline as pipeline_module
from src.classes.pipeline import Pipeline
from src.classes.trainer import TrainingOutcome
from src.schemas.dataset_schemas import DatasetBundle, XYDataset
from src.schemas.metrics import FinalTestMetrics
from src.schemas.pipeline_schemas import RandomStates
from src.schemas.run_records import FoldRecord, ModelTrainingResult, TuningRecord
from src.utils.prediction_tables import BinaryTestPredictions, FinalTestPredictions
from tests.factories import classification_metrics


def _test_set(labels, signal=None):
    y = pd.Series(labels)
    signal = labels if signal is None else signal
    return XYDataset(X=pd.DataFrame({"signal": signal}), y=y)


def _bundle():
    return DatasetBundle(
        train_data=_test_set([0, 1]),
        test_mimic=_test_set([0, 1, 0, 1]),
        test_tudd=_test_set([1, 0, 1, 0], signal=[0, 1, 0, 1]),
    )


def _final_test_predictions() -> FinalTestPredictions:
    return FinalTestPredictions(
        mimic=BinaryTestPredictions(
            test_set_id=np.array([10, 11, 12, 13]),
            y_true=np.array([0, 1, 0, 1]),
            positive_class_probability=np.array([0.1, 0.9, 0.2, 0.8]),
        ),
        tudd=BinaryTestPredictions(
            test_set_id=np.array([20, 21, 22, 23]),
            y_true=np.array([1, 0, 1, 0]),
            positive_class_probability=np.array([0.8, 0.2, 0.7, 0.3]),
        ),
    )


def _tuned_training_result(model_name: str) -> ModelTrainingResult:
    return ModelTrainingResult(
        model_name=model_name,
        task_type="classification",
        tuned=True,
        fit_time=0.2,
        tuning_result=TuningRecord(
            best_params={},
            scoring="accuracy",
            final_test_metrics=FinalTestMetrics(
                mimic_test=classification_metrics(1.0),
                mimic_prediction_time=0.1,
                tudd_test=classification_metrics(0.0),
                tudd_prediction_time=0.2,
            ),
            fold_results=[FoldRecord(0, 0, classification_metrics(1.0), 0.0, {})],
        ),
    )


def _build_pipeline(monkeypatch, train_fn):
    """Build a Pipeline whose Trainer is a fake that delegates to train_fn."""
    bundle = _bundle()

    class FakeDataset:
        def get_dataset(self):
            return bundle

        def summarize(self, data):
            return SimpleNamespace(
                target="mortality",
                data_files=(
                    SimpleNamespace(data_origin="mimic", sha256="a" * 64),
                    SimpleNamespace(data_origin="tudd", sha256="b" * 64),
                ),
            )

    class FakeTrainer:
        def __init__(self, task_type, default_imputer, default_scaler, random_states, log_transform_target):
            self.task_type = task_type

        def validate_model_configs(self):
            pass

        def validate_training_data(self, X_train, y_train):
            pass

        def train_evaluate_model(self, model_params, data):
            result = train_fn(model_params)
            return TrainingOutcome(
                result=result,
                test_predictions=_final_test_predictions() if result.tuning_result is not None else None,
            )

    monkeypatch.setattr(pipeline_module, "Trainer", FakeTrainer)

    pipeline = object.__new__(Pipeline)
    pipeline.dataset = FakeDataset()
    pipeline.pipeline_config = SimpleNamespace(
        run_id="run",
        random_states=RandomStates.from_seed(1),
        dataset=SimpleNamespace(target="mortality", imputer=None, scaler_encoder=None, log_transform_target=False),
        training=(
            SimpleNamespace(name="model-a"),
            SimpleNamespace(name="model-b"),
        ),
    )
    return pipeline


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
    def train_fn(model_params):
        return ModelTrainingResult(model_name=model_params.name, task_type="classification", tuned=False, fit_time=0.2)

    result = _build_pipeline(monkeypatch, train_fn).run()

    assert result.model_results == ()
    assert len(result.training_results) == 2
    assert all(not hasattr(tr, "trained_model") for tr in result.training_results)


def test_pipeline_records_failed_model_and_continues(monkeypatch):
    def train_fn(model_params):
        if model_params.name == "model-a":
            raise ValueError("bad params")
        return _tuned_training_result(model_params.name)

    result = _build_pipeline(monkeypatch, train_fn).run()

    assert [tr.model_name for tr in result.training_results] == ["model-a", "model-b"]
    assert result.training_results[0].failure_stage == "training_evaluation"
    assert result.training_results[0].error == "ValueError: bad params"
    assert result.training_results[0].task_type == "classification"
    assert result.training_results[1].succeeded
    assert [model.model_name for model in result.model_results] == ["model-b"]


def test_pipeline_records_evaluation_failure_and_continues(monkeypatch):
    def train_fn(model_params):
        if model_params.name == "model-a":
            raise RuntimeError("bad evaluation")
        return _tuned_training_result(model_params.name)

    result = _build_pipeline(monkeypatch, train_fn).run()

    assert result.training_results[0].failure_stage == "training_evaluation"
    assert result.training_results[0].error == "RuntimeError: bad evaluation"
    assert not hasattr(result.training_results[0], "trained_model")
    assert result.training_results[1].succeeded
    assert [model.model_name for model in result.model_results] == ["model-b"]


def test_pipeline_accumulates_one_probability_column_per_model(monkeypatch):
    pipeline = _build_pipeline(monkeypatch, lambda model_params: _tuned_training_result(model_params.name))

    pipeline.run()

    tables = pipeline.prediction_tables.frames()
    assert list(tables) == ["mimic", "tudd"]
    assert list(tables["mimic"].columns) == [
        "test_set_id",
        "y_true",
        "y_pred_model-a",
        "y_pred_model-b",
    ]
    assert tables["mimic"]["test_set_id"].str.fullmatch(r"ts_[0-9a-f]{64}").all()
    assert tables["mimic"]["y_true"].tolist() == [0, 1, 0, 1]


def test_pipeline_continues_when_model_completion_callback_fails(monkeypatch):
    pipeline = _build_pipeline(monkeypatch, lambda model_params: _tuned_training_result(model_params.name))
    completed = []

    def failing_callback(partial_result, model_run):
        completed.append(model_run.model_instance_id)
        raise RuntimeError("tracking unavailable")

    result = pipeline.run(on_model_complete=failing_callback)

    assert completed == ["model-a", "model-b"]
    assert len(result.model_runs) == 2
    assert all(model_run.succeeded for model_run in result.model_runs)


def test_pipeline_preserves_model_success_when_prediction_accumulation_fails(monkeypatch):
    class FailingPredictionAccumulator:
        def __init__(self, *args, **kwargs):
            pass

        def add(self, model_instance_id, predictions):
            raise OSError("artifact storage unavailable")

    monkeypatch.setattr(pipeline_module, "PredictionTableAccumulator", FailingPredictionAccumulator)
    pipeline = _build_pipeline(monkeypatch, lambda model_params: _tuned_training_result(model_params.name))

    result = pipeline.run()

    assert len(result.model_runs) == 2
    assert all(model_run.succeeded for model_run in result.model_runs)
