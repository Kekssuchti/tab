from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from src.classes import trainer as trainer_module
from src.classes.trainer import Trainer
from src.interfaces.model_interface import TimedPrediction
from src.schemas.dataset_schemas import DatasetBundle, XYDataset
from src.schemas.metrics import BootstrapClassificationMetrics, BootstrapFinalTestMetrics
from src.schemas.preprocessing_schemas import ImputerConfig, ScalerEncoderConfig
from src.schemas.training_schemas import CrossValidationConfig, ModelConfig, OptunaConfig, TuningConfig
from src.utils import model_registry
from src.utils.tuning_distributions import LogUniform
from tests.factories import bootstrap_classification_metrics
from tests.toy_data import load_toy_classification_data


def _classification_data():
    X, y = load_toy_classification_data()
    return X, y


def _preprocess_pipeline():
    return {
        "default_imputer": ImputerConfig(imputation_method="none"),
        "default_scaler": ScalerEncoderConfig(type="none"),
    }


def _bundle(X, y):
    xy = XYDataset(X=X, y=pd.Series(y))
    return DatasetBundle(train_data=xy, test_mimic=xy, test_tudd=xy)


def _model_config(
    name: str = "logistic-regression",
    *,
    method: str = "grid",
    grid: dict | None = None,
    scoring: str = "accuracy",
    n_splits: int = 2,
    search_space: str | None = None,
    optuna: OptunaConfig | None = None,
    preprocessing: dict | None = None,
) -> ModelConfig:
    tuning_kwargs = {
        "method": method,
        "grid": grid,
        "scoring": scoring,
        "cv": CrossValidationConfig(n_splits=n_splits, random_state=1),
    }
    if search_space is not None:
        tuning_kwargs["search_space"] = search_space
    if optuna is not None:
        tuning_kwargs["optuna"] = optuna
    return ModelConfig(name=name, tuning=TuningConfig(**tuning_kwargs), preprocessing=preprocessing)


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
        return TimedPrediction(probabilities, 0.0)

    def release(self):
        if self.released:
            return
        self.released = True
        type(self).active -= 1
        type(self).releases += 1


class _ReleasablePredictFailureModel(_ReleasableFoldModel):
    def predict(self, X_test):
        raise RuntimeError("training metric prediction failed")


class _FitFailureAdapter:
    task_type = "classification"
    kwargs: ClassVar[dict] = {}
    model = object()
    releases = 0

    def fit(self, X_train, y_train):
        raise RuntimeError("fit failed")

    def release(self):
        type(self).releases += 1


def _bootstrap_final_metrics() -> BootstrapFinalTestMetrics:
    bootstrap_metrics = bootstrap_classification_metrics(0.5, lower=0.4, upper=0.6)
    return BootstrapFinalTestMetrics(
        mimic_test=bootstrap_metrics,
        mimic_prediction_time=0.1,
        tudd_test=bootstrap_metrics,
        tudd_prediction_time=0.2,
    )


def test_trainer_records_final_metrics_after_training():
    X, y = _classification_data()
    trainer = Trainer(task_type="classification", **_preprocess_pipeline())

    result = trainer.train_evaluate_model(_model_config(grid={"C": [1.0]}), _bundle(X, y))

    assert result.model_name == "logistic-regression"
    assert result.task_type == "classification"
    assert result.tuned
    assert result.fit_time >= 0
    assert not hasattr(result, "trained_model")
    assert result.tuning_result is not None
    assert result.tuning_result.final_test_metrics.mimic_test.accuracy >= 0.0
    assert result.tuning_result.final_test_metrics.tudd_test.accuracy >= 0.0


def test_trainer_uses_one_full_training_fit_for_bootstrap_evaluation(monkeypatch):
    _ReleasableFoldModel.active = 0
    _ReleasableFoldModel.peak = 0
    _ReleasableFoldModel.releases = 0
    X, y = _classification_data()
    trainer = Trainer(task_type="classification", **_preprocess_pipeline())
    model_config = _model_config(grid={"C": [1.0]})
    model_spec = model_registry.get_model_spec(model_config, "classification")
    fit_calls = 0

    def _fit_model(model_params, spec, params, X_train, y_train):
        nonlocal fit_calls
        fit_calls += 1
        return _ReleasableFoldModel(), 0.25

    monkeypatch.setattr(trainer, "_fit_model", _fit_model)
    monkeypatch.setattr(
        trainer_module,
        "evaluate_trained_model_bootstrap",
        lambda trained_model, task_type, data: _bootstrap_final_metrics(),
    )

    result = trainer._tune_model(model_config, model_spec, _bundle(X, y))

    assert fit_calls == 1
    assert result.fit_time == pytest.approx(0.25)
    assert result.tuning_result is not None
    assert isinstance(result.tuning_result.final_test_metrics.mimic_test, BootstrapClassificationMetrics)
    assert result.tuning_result.final_test_metrics.mimic_test.metrics.accuracy == pytest.approx(0.5)
    assert _ReleasableFoldModel.releases == 1
    assert _ReleasableFoldModel.active == 0


def test_single_candidate_grid_does_not_construct_cv(monkeypatch):
    X, y = _classification_data()
    trainer = Trainer(task_type="classification", **_preprocess_pipeline())
    model_config = _model_config(grid={"C": [1.0]}, scoring="roc_auc", n_splits=5)

    def fail_if_cv_is_constructed(tuning):
        pytest.fail("CV must not be constructed for a single tuning candidate")

    monkeypatch.setattr(trainer, "_build_cv", fail_if_cv_is_constructed)

    result = trainer._tune_model(
        model_config,
        model_registry.get_model_spec(model_config, "classification"),
        _bundle(X, y),
    )

    assert result.tuning_result is not None
    assert result.tuning_result.fold_results == []


@pytest.mark.parametrize(
    ("method", "grid"),
    [
        ("grid", {"C": [0.1, 1.0]}),
        ("optuna", {"C": [0.1, 1.0]}),
    ],
)
def test_tuning_rejects_more_cv_splits_than_minority_samples(monkeypatch, method, grid):
    X, _ = _classification_data()
    X = X.iloc[:10]
    y = np.array([0] * 7 + [1] * 3)
    trainer = Trainer(task_type="classification", **_preprocess_pipeline())
    model_config = _model_config(
        method=method,
        grid=grid,
        scoring="roc_auc",
        n_splits=5,
        optuna=OptunaConfig(n_trials=1, sampler="random"),
    )

    def fail_if_candidate_is_evaluated(*args, **kwargs):
        pytest.fail("CV candidate evaluation must not start with invalid folds")

    monkeypatch.setattr(trainer, "_evaluate_cv_candidate", fail_if_candidate_is_evaluated)

    with pytest.raises(
        ValueError,
        match=r"5-fold cross-validation.*minority class has only 3 samples",
    ):
        trainer._tune_model(
            model_config,
            model_registry.get_model_spec(model_config, "classification"),
            _bundle(X, y),
        )


def test_trainer_uses_tuning_grid_and_returns_best_params():
    X, y = _classification_data()
    trainer = Trainer(task_type="classification", **_preprocess_pipeline())

    result = trainer.train_evaluate_model(_model_config(grid={"C": [0.1, 1.0]}), _bundle(X, y))

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
    assert all("accuracy" in fold.metrics.scores for fold in result.tuning_result.fold_results)
    assert result.tuning_result.final_test_metrics.mimic_test.accuracy >= 0.0


def test_trainer_can_tune_with_optuna_categorical_grid():
    X, y = _classification_data()
    trainer = Trainer(task_type="classification", **_preprocess_pipeline())

    result = trainer.train_evaluate_model(
        _model_config(
            method="optuna",
            grid={"C": [0.1, 1.0]},
            optuna=OptunaConfig(n_trials=2, sampler="random"),
        ),
        _bundle(X, y),
    )

    assert result.tuned
    assert result.tuning_result is not None
    assert result.tuning_result.method == "optuna"
    assert {fold.candidate_index for fold in result.tuning_result.fold_results} == {
        0,
        1,
    }
    assert len(result.tuning_result.fold_results) == 4
    assert set(result.tuning_result.best_params) == {"C"}
    assert result.tuning_result.final_test_metrics.tudd_test.accuracy >= 0.0


def test_trainer_can_tune_with_mixed_optuna_search_space():
    X, y = _classification_data()
    model_params = _model_config(
        method="optuna",
        search_space="mixed",
        optuna=OptunaConfig(n_trials=2, sampler="random"),
    )
    registered_spec = model_registry.get_model_spec(model_params, "classification")
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
    trainer = Trainer(task_type="classification", **_preprocess_pipeline())

    result = trainer._tune_model(model_params, mixed_spec, _bundle(X, y))

    assert result.tuning_result is not None
    assert result.tuning_result.method == "optuna"
    assert len(result.tuning_result.fold_results) == 4
    assert 0.1 <= result.tuning_result.best_params["C"] <= 1.0
    assert result.tuning_result.best_params["fit_intercept"] in (True, False)


def test_trainer_releases_models_between_tuning_folds(monkeypatch):
    _ReleasableFoldModel.active = 0
    _ReleasableFoldModel.peak = 0
    _ReleasableFoldModel.releases = 0
    X, y = _classification_data()
    trainer = Trainer(task_type="classification", **_preprocess_pipeline())
    model_params = _model_config(grid={"C": [0.1, 1.0]})
    model_spec = model_registry.get_model_spec(model_params, "classification")

    def _fit_model(model_params, spec, params, X_train, y_train):
        return _ReleasableFoldModel(), 0.0

    monkeypatch.setattr(trainer, "_fit_model", _fit_model)

    result = trainer._tune_model(model_params, model_spec, _bundle(X, y))

    assert result.tuned
    assert _ReleasableFoldModel.peak == 1
    assert _ReleasableFoldModel.releases == 5
    assert _ReleasableFoldModel.active == 0


def test_trainer_uses_model_specific_preprocessing_override():
    X, y = _classification_data()
    X.loc[3, "regulatory_score"] = np.nan
    trainer = Trainer(
        task_type="classification",
        default_imputer=ImputerConfig(imputation_method="none"),
        default_scaler=ScalerEncoderConfig(type="none"),
    )

    result = trainer.train_evaluate_model(
        _model_config(
            grid={"C": [1.0]},
            preprocessing={"imputer": {"imputation_method": "mean"}, "scaler_encoder": {"type": "none"}},
        ),
        _bundle(X, y),
    )

    assert result.model_name == "logistic-regression"
    assert result.tuning_result is not None
    assert result.tuning_result.final_test_metrics.mimic_test.accuracy >= 0.0


def test_trainer_encodes_categorical_columns_before_xgboost():
    X, y = _classification_data()
    rng = np.random.default_rng(0)
    X["Sex"] = rng.choice(["F", "M"], len(X))
    trainer = Trainer(
        task_type="classification",
        default_imputer=ImputerConfig(imputation_method="none"),
        default_scaler=ScalerEncoderConfig(type="none"),
    )

    result = trainer.train_evaluate_model(
        _model_config(name="xgboost", grid={"n_estimators": [2], "max_depth": [1], "n_jobs": [1]}),
        _bundle(X, y),
    )

    assert result.model_name == "xgboost"
    assert result.tuning_result is not None
    assert result.tuning_result.final_test_metrics.mimic_test.accuracy >= 0.0


def test_trainer_releases_final_model_when_training_metrics_fail(monkeypatch):
    _ReleasableFoldModel.active = 0
    _ReleasableFoldModel.peak = 0
    _ReleasableFoldModel.releases = 0
    _ReleasablePredictFailureModel.active = 0
    _ReleasablePredictFailureModel.peak = 0
    _ReleasablePredictFailureModel.releases = 0
    X, y = _classification_data()
    trainer = Trainer(task_type="classification", **_preprocess_pipeline())
    model_params = _model_config(grid={"C": [1.0, 2.0]})

    fit_calls = 0

    def _fit_model(model_params, spec, params, X_train, y_train):
        nonlocal fit_calls
        fit_calls += 1
        if fit_calls <= 4:
            return _ReleasableFoldModel(), 0.0
        return _ReleasablePredictFailureModel(), 0.0

    monkeypatch.setattr(trainer, "_fit_model", _fit_model)

    with pytest.raises(RuntimeError, match="training metric prediction failed"):
        trainer._tune_model(
            model_params,
            model_registry.get_model_spec(model_params, "classification"),
            _bundle(X, y),
        )

    assert _ReleasablePredictFailureModel.peak == 1
    assert _ReleasablePredictFailureModel.releases == 1
    assert _ReleasablePredictFailureModel.active == 0
    assert _ReleasableFoldModel.releases == 4


def test_trainer_releases_adapter_when_fit_fails():
    X, y = _classification_data()
    trainer = Trainer(task_type="classification", **_preprocess_pipeline())
    model_config = ModelConfig(name="failing")

    class _Spec:
        def create(self, task_type, params):
            return _FitFailureAdapter()

    _FitFailureAdapter.releases = 0
    with pytest.raises(RuntimeError, match="fit failed"):
        trainer._fit_model(model_config, _Spec(), {}, X, y)

    assert _FitFailureAdapter.releases == 1
