from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from src.classes.trainer import Trainer
from src.schemas.preprocessing_schemas import ImputerParams, ScalerEncoderParams
from src.schemas.training_schemas import (
    CVParams,
    ModelParams,
    OptunaParams,
    TuningParams,
)
from src.utils.model_lifecycle import release_model


def _classification_data():
    X = pd.DataFrame(
        {
            "age": [30, 35, 40, 45, 60, 65, 70, 75],
            "lab": [1.0, 1.2, 1.5, 1.7, 4.0, 4.2, 4.5, 4.7],
        }
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


def _regression_data():
    X = pd.DataFrame(
        {
            "age": [30, 35, 40, 45, 60, 65],
            "lab": [1.0, 1.5, 2.0, 2.5, 4.0, 4.5],
        }
    )
    y = np.array([10.0, 12.0, 14.0, 16.0, 24.0, 26.0])
    return X, y


def _preprocess_pipeline():
    return {
        "default_imputer": ImputerParams(imputation_method="none"),
        "default_scaler": ScalerEncoderParams(type="none"),
    }


@contextmanager
def _trained_result(trainer: Trainer, model_params: ModelParams, X, y):
    result = trainer.train_model(model_params, X, y)
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


def test_trainer_returns_adapter_that_predicts_after_pipeline_training():
    X, y = _classification_data()
    model_params = ModelParams(
        name="logistic-regression",
        task_type="classification",
        params={"max_iter": 200},
    )
    trainer = Trainer(
        params=(model_params,),
        **_preprocess_pipeline(),
    )

    with _trained_result(trainer, model_params, X, y) as result:
        predictions, predict_time = result.trained_model.predict(X)

        assert result.model_name == "logistic-regression"
        assert not result.tuned
        assert result.training_metrics is not None
        assert result.training_metrics.roc_auc is not None
        assert "accuracy" in result.training_metrics.scores
        assert result.fit_time >= 0
        assert predict_time >= 0
        assert predictions.shape == (len(X), 2)
        assert np.isfinite(predictions).all()


def test_trainer_uses_tuning_grid_and_returns_best_params():
    X, y = _classification_data()
    model_params = ModelParams(
        name="logistic-regression",
        task_type="classification",
        params={"max_iter": 200},
        tuning=TuningParams(
            grid={"C": [0.1, 1.0]},
            scoring="accuracy",
            cv=CVParams(n_splits=2, random_state=1),
        ),
    )
    trainer = Trainer(
        params=(model_params,),
        **_preprocess_pipeline(),
    )

    with _trained_result(trainer, model_params, X, y) as result:
        predictions, _ = result.trained_model.predict(X)

        assert result.tuned
        assert result.tuning_result is not None
        assert set(result.tuning_result.best_params) == {"C"}
        assert result.tuning_result.best_score >= 0
        assert (
            result.tuning_result.best_score
            == result.tuning_result.best_metrics.accuracy
        )
        assert len(result.tuning_result.cv_results.params) == 2
        assert len(result.tuning_result.cv_results.mean_scores) == 2
        assert len(result.tuning_result.cv_results.mean_metrics) == 2
        assert len(result.tuning_result.fold_results) == 4
        assert all(
            "accuracy" in fold.metrics.scores
            for fold in result.tuning_result.fold_results
        )
        assert predictions.shape == (len(X), 2)


def test_trainer_can_tune_with_optuna_categorical_grid():
    X, y = _classification_data()
    model_params = ModelParams(
        name="logistic-regression",
        task_type="classification",
        params={"max_iter": 200},
        tuning=TuningParams(
            method="optuna",
            grid={"C": [0.1, 1.0]},
            scoring="accuracy",
            cv=CVParams(n_splits=2, random_state=1),
            optuna=OptunaParams(n_trials=2, sampler="random"),
        ),
    )
    trainer = Trainer(
        params=(model_params,),
        **_preprocess_pipeline(),
    )

    with _trained_result(trainer, model_params, X, y) as result:
        predictions, _ = result.trained_model.predict(X)

        assert result.tuned
        assert result.tuning_result is not None
        assert result.tuning_result.method == "optuna"
        assert len(result.tuning_result.cv_results.params) == 2
        assert result.tuning_result.cv_results.trial_numbers == [0, 1]
        assert len(result.tuning_result.fold_results) == 4
        assert set(result.tuning_result.best_params) == {"C"}
        assert predictions.shape == (len(X), 2)


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
        params=(),
        **_preprocess_pipeline(),
    )
    model_params = ModelParams(
        name="logistic-regression",
        task_type="classification",
        tuning=TuningParams(
            grid={"C": [0.1, 1.0]},
            scoring="accuracy",
            cv=CVParams(n_splits=2, random_state=1),
        ),
    )

    class _Spec:
        def tuning_candidates(self, search_space, overrides):
            return [{"C": value} for value in overrides["C"]]

    def _fit_model(model_params, spec, params, X_train, y_train):
        return _ReleasableFoldModel(), 0.0

    monkeypatch.setattr(trainer, "_fit_model", _fit_model)

    result = trainer._tune_model(model_params, _Spec(), X, y)

    assert result.tuned
    assert _ReleasableFoldModel.peak == 1
    assert _ReleasableFoldModel.releases == 4
    assert _ReleasableFoldModel.active == 1

    release_model(result.trained_model)
    assert _ReleasableFoldModel.active == 0


def test_trainer_can_fit_regression_adapter_behind_same_interface():
    X, y = _regression_data()
    model_params = ModelParams(
        name="xgboost",
        task_type="regression",
    )
    trainer = Trainer(
        params=(model_params,),
        **_preprocess_pipeline(),
    )

    with _trained_result(trainer, model_params, X, y) as result:
        predictions, _ = result.trained_model.predict(X)

        assert result.model_name == "xgboost"
        assert result.training_metrics is None
        assert predictions.shape == (len(X),)
        assert np.isfinite(predictions).all()


def test_trainer_uses_model_specific_preprocessing_override():
    X, y = _classification_data()
    X.loc[0, "lab"] = np.nan
    model_params = ModelParams(
        name="logistic-regression",
        task_type="classification",
        params={"max_iter": 200},
        preprocessing={
            "imputer": {"imputation_method": "mean"},
            "scaler_encoder": {"type": "none"},
        },
    )
    trainer = Trainer(
        params=(model_params,),
        default_imputer=ImputerParams(imputation_method="none"),
        default_scaler=ScalerEncoderParams(type="none"),
    )

    with _trained_result(trainer, model_params, X, y) as result:
        predictions, _ = result.trained_model.predict(X)

        assert result.model_name == "logistic-regression"
        assert predictions.shape == (len(X), 2)
        assert np.isfinite(predictions).all()


def test_trainer_encodes_categorical_columns_before_xgboost():
    X, y = _classification_data()
    X["Sex"] = ["F", "M", "F", "M", "F", "M", "F", "M"]
    model_params = ModelParams(
        name="xgboost",
        task_type="classification",
        params={"n_estimators": 2, "max_depth": 1, "n_jobs": 1},
    )
    trainer = Trainer(
        params=(model_params,),
        default_imputer=ImputerParams(imputation_method="none"),
        default_scaler=ScalerEncoderParams(type="none"),
    )

    with _trained_result(trainer, model_params, X, y) as result:
        predictions, _ = result.trained_model.predict(X)

        assert result.model_name == "xgboost"
        assert predictions.shape == (len(X), 2)
        assert np.isfinite(predictions).all()


def test_trainer_releases_final_model_when_training_metrics_fail(monkeypatch):
    _ReleasablePredictFailureModel.active = 0
    _ReleasablePredictFailureModel.peak = 0
    _ReleasablePredictFailureModel.releases = 0
    X, y = _classification_data()
    trainer = Trainer(
        params=(),
        **_preprocess_pipeline(),
    )
    model_params = ModelParams(
        name="logistic-regression",
        task_type="classification",
    )

    class _Spec:
        pass

    def _fit_model(model_params, spec, params, X_train, y_train):
        return _ReleasablePredictFailureModel(), 0.0

    monkeypatch.setattr(trainer, "_fit_model", _fit_model)

    with pytest.raises(RuntimeError, match="training metric prediction failed"):
        trainer._train_without_tuning(model_params, _Spec(), X, y)

    assert _ReleasablePredictFailureModel.peak == 1
    assert _ReleasablePredictFailureModel.releases == 1
    assert _ReleasablePredictFailureModel.active == 0


def test_trainer_preflight_rejects_bad_model_params():
    trainer = Trainer(
        params=(
            ModelParams(
                name="logistic-regression",
                task_type="classification",
                params={"bad_param": 1},
            ),
        ),
        **_preprocess_pipeline(),
    )

    with pytest.raises(ValueError, match="Model preflight validation failed"):
        trainer.validate_model_configs()
