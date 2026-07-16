from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from src.classes.trainer import Trainer
from src.schemas.dataset_schemas import DatasetBundle, XYDataset
from src.schemas.preprocessing_schemas import ImputerConfig, ScalerEncoderConfig
from src.schemas.training_schemas import (
    CrossValidationConfig,
    ModelConfig,
    OptunaConfig,
    TuningConfig,
)
from src.utils import model_registry
from src.utils.load_data import load_toy_data_cls, load_toy_data_reg
from src.utils.model_lifecycle import release_model
from src.utils.tuning_distributions import LogUniform


def _classification_data():
    X, y = load_toy_data_cls()
    return X, y


def _regression_data():
    X, y = load_toy_data_reg()
    y = ((y - y.mean()) / y.std()).astype(np.float32)

    return X, y


def _preprocess_pipeline():
    return {
        "default_imputer": ImputerConfig(imputation_method="none"),
        "default_scaler": ScalerEncoderConfig(type="none"),
    }


def _bundle(X, y):
    xy = XYDataset(X=X, y=pd.Series(y))
    return DatasetBundle(train_data=xy, test_mimic=xy, test_tudd=xy)


@contextmanager
def _trained_result(trainer: Trainer, model_params: ModelConfig, X, y):
    result = trainer.train_evaluate_model(model_params, _bundle(X, y))
    try:
        yield result
    finally:
        release_model(result.trained_model)


class _ReleasableFoldModel:
    active = 0
    peak = 0
    releases = 0

    def __init__(self):
        self.released = False
        type(self).active += 1
        type(self).peak = max(type(self).peak, type(self).active)

    def predict(self, X_test):
        probabilities = np.tile([0.4, 0.6], (len(X_test), 1))
        return probabilities, 0.0

    def release(self):
        if self.released:
            return
        self.released = True
        type(self).active -= 1
        type(self).releases += 1


class _ReleasablePredictFailureModel(_ReleasableFoldModel):
    def predict(self, X_test):
        raise RuntimeError("training metric prediction failed")


def test_trainer_records_final_metrics_after_training():
    X, y = _classification_data()
    model_params = ModelConfig(
        name="logistic-regression",
        task_type="classification",
        params={"max_iter": 200},
        tuning=TuningConfig(
            method="grid",
            grid={"C": [1.0]},
            scoring="accuracy",
            cv=CrossValidationConfig(n_splits=2, random_state=1),
        ),
    )
    trainer = Trainer(
        configs=(model_params,),
        **_preprocess_pipeline(),
    )

    with _trained_result(trainer, model_params, X, y) as result:
        assert result.model_name == "logistic-regression"
        assert result.tuned
        assert result.fit_time >= 0
        assert result.trained_model is None
        assert result.tuning_result is not None
        assert result.tuning_result.final_test_metrics.mimic_test.mean_accuracy >= 0.0
        assert result.tuning_result.final_test_metrics.tudd_test.mean_accuracy >= 0.0


def test_trainer_uses_tuning_grid_and_returns_best_params():
    X, y = _classification_data()
    model_params = ModelConfig(
        name="logistic-regression",
        task_type="classification",
        params={"max_iter": 200},
        tuning=TuningConfig(
            method="grid",
            grid={"C": [0.1, 1.0]},
            scoring="accuracy",
            cv=CrossValidationConfig(n_splits=2, random_state=1),
        ),
    )
    trainer = Trainer(
        configs=(model_params,),
        **_preprocess_pipeline(),
    )

    with _trained_result(trainer, model_params, X, y) as result:
        assert result.tuned
        assert result.tuning_result is not None
        assert set(result.tuning_result.best_params) == {"C"}
        assert not hasattr(result.tuning_result, "cv_results")
        assert not hasattr(result.tuning_result, "best_score")
        assert not hasattr(result.tuning_result, "best_metrics")
        assert {fold.candidate_index for fold in result.tuning_result.fold_results} == {
            0,
            1,
        }
        assert len(result.tuning_result.fold_results) == 4
        assert all(
            "accuracy" in fold.metrics.scores
            for fold in result.tuning_result.fold_results
        )
        assert result.tuning_result.final_test_metrics.mimic_test.mean_accuracy >= 0.0
        assert result.trained_model is None


def test_trainer_can_tune_with_optuna_categorical_grid():
    X, y = _classification_data()
    model_params = ModelConfig(
        name="logistic-regression",
        task_type="classification",
        params={"max_iter": 200},
        tuning=TuningConfig(
            method="optuna",
            grid={"C": [0.1, 1.0]},
            scoring="accuracy",
            cv=CrossValidationConfig(n_splits=2, random_state=1),
            optuna=OptunaConfig(n_trials=2, sampler="random"),
        ),
    )
    trainer = Trainer(
        configs=(model_params,),
        **_preprocess_pipeline(),
    )

    with _trained_result(trainer, model_params, X, y) as result:
        assert result.tuned
        assert result.tuning_result is not None
        assert result.tuning_result.method == "optuna"
        assert {fold.candidate_index for fold in result.tuning_result.fold_results} == {
            0,
            1,
        }
        assert len(result.tuning_result.fold_results) == 4
        assert set(result.tuning_result.best_params) == {"C"}
        assert result.tuning_result.final_test_metrics.tudd_test.mean_accuracy >= 0.0
        assert result.trained_model is None


def test_trainer_can_tune_with_mixed_optuna_search_space():
    X, y = _classification_data()
    model_params = ModelConfig(
        name="logistic-regression",
        task_type="classification",
        params={"max_iter": 200},
        tuning=TuningConfig(
            method="optuna",
            search_space="mixed",
            scoring="accuracy",
            cv=CrossValidationConfig(n_splits=2, random_state=1),
            optuna=OptunaConfig(n_trials=2, sampler="random"),
        ),
    )
    registered_spec = model_registry.get_model_spec(model_params)
    mixed_spec = model_registry.ModelSpec(
        adapter_path=registered_spec.adapter_path,
        default_params=registered_spec.default_params,
        search_spaces={
            "mixed": {
                "C": LogUniform(0.1, 1.0),
                "fit_intercept": [True, False],
            }
        },
    )
    trainer = Trainer(
        configs=(model_params,),
        **_preprocess_pipeline(),
    )

    result = trainer._tune_model(model_params, mixed_spec, _bundle(X, y))

    assert result.tuning_result is not None
    assert result.tuning_result.method == "optuna"
    assert len(result.tuning_result.fold_results) == 4
    assert 0.1 <= result.tuning_result.best_params["C"] <= 1.0
    assert result.tuning_result.best_params["fit_intercept"] in (True, False)


def test_trainer_merges_nested_tuning_params():
    assert Trainer._merge_params(
        {"inference_config": {"SUBSAMPLE_SAMPLES": 128, "OTHER": True}},
        {"inference_config": {"SUBSAMPLE_SAMPLES": 256}},
    ) == {"inference_config": {"SUBSAMPLE_SAMPLES": 256, "OTHER": True}}


def test_trainer_releases_models_between_tuning_folds(monkeypatch):
    _ReleasableFoldModel.active = 0
    _ReleasableFoldModel.peak = 0
    _ReleasableFoldModel.releases = 0
    X, y = _classification_data()
    trainer = Trainer(
        configs=(),
        **_preprocess_pipeline(),
    )
    model_params = ModelConfig(
        name="logistic-regression",
        task_type="classification",
        tuning=TuningConfig(
            method="grid",
            grid={"C": [0.1, 1.0]},
            scoring="accuracy",
            cv=CrossValidationConfig(n_splits=2, random_state=1),
        ),
    )

    model_spec = model_registry.get_model_spec(
        ModelConfig(
            name="logistic-regression",
            task_type="classification",
            tuning=TuningConfig(
                method="grid",
                grid={"C": [0.1, 1.0]},
                scoring="accuracy",
                cv=CrossValidationConfig(n_splits=2, random_state=1),
            ),
        )
    )

    def _fit_model(model_params, spec, params, X_train, y_train):
        return _ReleasableFoldModel(), 0.0

    monkeypatch.setattr(trainer, "_fit_model", _fit_model)

    result = trainer._tune_model(model_params, model_spec, _bundle(X, y))

    assert result.tuned
    assert _ReleasableFoldModel.peak == 1
    assert _ReleasableFoldModel.releases == 6
    assert _ReleasableFoldModel.active == 0


def test_trainer_rejects_regression_tuning_until_metrics_are_implemented():
    X, y = _regression_data()
    model_params = ModelConfig(
        name="xgboost",
        task_type="regression",
        params={"n_estimators": 2, "max_depth": 1, "n_jobs": 1},
        tuning=TuningConfig(
            method="grid",
            grid={"n_estimators": [2]},
            scoring="accuracy",
            cv=CrossValidationConfig(n_splits=2, random_state=1),
        ),
    )
    trainer = Trainer(
        configs=(model_params,),
        **_preprocess_pipeline(),
    )

    with pytest.raises(NotImplementedError, match="Regression tuning metrics"):
        trainer.train_evaluate_model(model_params, _bundle(X, y))


def test_trainer_uses_model_specific_preprocessing_override():
    X, y = _classification_data()
    X.loc[3, "regulatory_score"] = np.nan
    model_params = ModelConfig(
        name="logistic-regression",
        task_type="classification",
        params={"max_iter": 200},
        tuning=TuningConfig(
            method="grid",
            grid={"C": [1.0]},
            scoring="accuracy",
            cv=CrossValidationConfig(n_splits=2, random_state=1),
        ),
        preprocessing={
            "imputer": {"imputation_method": "mean"},
            "scaler_encoder": {"type": "none"},
        },
    )
    trainer = Trainer(
        configs=(model_params,),
        default_imputer=ImputerConfig(imputation_method="none"),
        default_scaler=ScalerEncoderConfig(type="none"),
    )

    with _trained_result(trainer, model_params, X, y) as result:
        assert result.model_name == "logistic-regression"
        assert result.trained_model is None
        assert result.tuning_result is not None
        assert result.tuning_result.final_test_metrics.mimic_test.mean_accuracy >= 0.0


def test_trainer_encodes_categorical_columns_before_xgboost():
    X, y = _classification_data()
    X["Sex"] = np.random.choice(["F", "M"], len(X))
    model_params = ModelConfig(
        name="xgboost",
        task_type="classification",
        params={"n_estimators": 2, "max_depth": 1, "n_jobs": 1},
        tuning=TuningConfig(
            method="grid",
            grid={"n_estimators": [2]},
            scoring="accuracy",
            cv=CrossValidationConfig(n_splits=2, random_state=1),
        ),
    )
    trainer = Trainer(
        configs=(model_params,),
        default_imputer=ImputerConfig(imputation_method="none"),
        default_scaler=ScalerEncoderConfig(type="none"),
    )

    with _trained_result(trainer, model_params, X, y) as result:
        assert result.model_name == "xgboost"
        assert result.trained_model is None
        assert result.tuning_result is not None
        assert result.tuning_result.final_test_metrics.mimic_test.mean_accuracy >= 0.0


def test_trainer_releases_final_model_when_training_metrics_fail(monkeypatch):
    _ReleasableFoldModel.active = 0
    _ReleasableFoldModel.peak = 0
    _ReleasableFoldModel.releases = 0
    _ReleasablePredictFailureModel.active = 0
    _ReleasablePredictFailureModel.peak = 0
    _ReleasablePredictFailureModel.releases = 0
    X, y = _classification_data()
    trainer = Trainer(
        configs=(),
        **_preprocess_pipeline(),
    )
    model_params = ModelConfig(
        name="logistic-regression",
        task_type="classification",
        tuning=TuningConfig(
            method="grid",
            grid={"C": [1.0]},
            scoring="accuracy",
            cv=CrossValidationConfig(n_splits=2, random_state=1),
        ),
    )

    fit_calls = 0

    def _fit_model(model_params, spec, params, X_train, y_train):
        nonlocal fit_calls
        fit_calls += 1
        if fit_calls <= 2:
            return _ReleasableFoldModel(), 0.0
        return _ReleasablePredictFailureModel(), 0.0

    monkeypatch.setattr(trainer, "_fit_model", _fit_model)

    with pytest.raises(RuntimeError, match="training metric prediction failed"):
        trainer._tune_model(
            model_params,
            model_registry.get_model_spec(model_params),
            _bundle(X, y),
        )

    assert _ReleasablePredictFailureModel.peak == 1
    assert _ReleasablePredictFailureModel.releases == 1
    assert _ReleasablePredictFailureModel.active == 0
    assert _ReleasableFoldModel.releases == 2
