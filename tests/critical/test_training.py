"""Controlled predictions expose selection/leakage bugs that workflow metrics cannot."""

import builtins
import gc
import sys
import weakref
from dataclasses import dataclass, field

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.classes.trainer import Trainer
from src.interfaces.model_interface import ModelAdapter, TimedPrediction
from src.schemas.dataset_schemas import DatasetBundle, XYDataset
from src.schemas.pipeline_schemas import RandomStates
from src.schemas.preprocessing_schemas import ImputerConfig, ScalerEncoderConfig
from src.schemas.training_schemas import ModelConfig
from src.utils import model_registry


@dataclass
class ModelAudit:
    """Only values and weak references escape the adapter's lifetime."""

    events: list = field(default_factory=list)
    fits: list = field(default_factory=list)
    references: list = field(default_factory=list)
    live: set = field(default_factory=set)
    peak: int = 0

    def assert_released(self):
        assert self.peak == 1
        assert not self.live
        created = [identifier for event, identifier in self.events if event == "create"]
        released = [identifier for event, identifier in self.events if event == "release"]
        assert released == created
        # Every resource is released before the next model is even constructed.
        boundaries = [event for event, _ in self.events if event in {"create", "release"}]
        assert boundaries == ["create", "release"] * len(created)
        gc.collect()
        assert all(reference() is None for reference in self.references)


class _Estimator:
    pass


class ObservedAdapter(ModelAdapter):
    """Learn a signal threshold; optionally reverse it or fail at a real boundary."""

    audit: ModelAudit

    def __init__(
        self,
        task_type,
        random_state=None,
        inference_state=None,
        quality=1,
        failure=None,
        failure_rows=None,
    ):
        self.task_type = task_type
        self.kwargs = {"quality": quality}
        self.quality = quality
        self.failure = failure
        self.failure_rows = failure_rows
        self.random_state = random_state
        self.inference_state = inference_state
        self.model = _Estimator()
        self.identifier = len(self.audit.references) // 2
        self.audit.references.extend((weakref.ref(self), weakref.ref(self.model)))
        self.audit.live.add(self.identifier)
        self.audit.peak = max(self.audit.peak, len(self.audit.live))
        self.audit.events.append(("create", self.identifier))

    def fit(self, X_train, y_train):
        self.rows = len(y_train)
        X = np.asarray(X_train)
        y = np.asarray(y_train)
        self.observation = {"quality": self.quality, "X": X.copy(), "y": y.copy(), "predictions": []}
        self.audit.fits.append(self.observation)
        self.audit.events.append(("fit", self.identifier))
        if self.rows == self.failure_rows and self.failure == "fit":
            raise RuntimeError("injected fit failure")
        if self.task_type == "classification":
            self.model.threshold = (X[y == 0, 0].mean() + X[y == 1, 0].mean()) / 2
        return 0.0

    def predict(self, X_test):
        X = np.asarray(X_test)
        self.observation["predictions"].append(X.copy())
        self.audit.events.append(("predict", self.identifier))
        if self.rows == self.failure_rows and self.failure == "predict":
            raise RuntimeError("injected predict failure")
        if self.task_type == "regression":
            return TimedPrediction(X[:, 0] + (1 - self.quality) * 2, 0.0)
        positive = np.where(self.quality * (X[:, 0] - self.model.threshold) > 0, 0.9, 0.1)
        values = np.column_stack((1 - positive, positive))
        if self.rows == self.failure_rows and self.failure == "evaluation":
            values[0] = np.nan  # Real probability validation, not a mocked evaluator.
        return TimedPrediction(values, 0.0)

    def release(self):
        self.audit.events.append(("release", self.identifier))
        self.audit.live.remove(self.identifier)
        super().release()


@pytest.fixture
def model_audit(monkeypatch):
    audit = ModelAudit()
    monkeypatch.setattr(ObservedAdapter, "audit", audit, raising=False)
    spec = model_registry.ModelSpec(adapter_path=f"{__name__}:ObservedAdapter")
    monkeypatch.setitem(model_registry.MODEL_REGISTRY_CLS, "observed", spec)
    monkeypatch.setitem(model_registry.MODEL_REGISTRY_REG, "observed", spec)
    return audit


def model_config(*, method="grid", qualities=(-1, 1), scoring="accuracy", **fixed_params):
    return ModelConfig.model_validate(
        {
            "name": "observed",
            "tuning": {
                "method": method,
                "grid": {"quality": list(qualities), **{key: [value] for key, value in fixed_params.items()}},
                "scoring": scoring,
                "cv": {"n_splits": 2},
                "optuna": {"n_trials": 2, "patience": 10},
            },
        }
    )


def training_bundle():
    y = np.tile([0, 1], 12)
    nuisance = np.arange(24, dtype=float) ** 2
    nuisance[[2, 15]] = np.nan
    nuisance[-1] = 10_000
    train = XYDataset(pd.DataFrame({"signal": 2 * y - 1, "shift": nuisance}), pd.Series(y))
    tests = []
    for start, shift in [(100, 100_000), (200, -100_000)]:
        labels = pd.Series([0, 1, 0, 1], index=np.arange(start, start + 4))
        tests.append(XYDataset(pd.DataFrame({"signal": 2 * labels - 1, "shift": shift}), labels))
    return DatasetBundle(train_data=train, test_mimic=tests[0], test_tudd=tests[1])


def make_trainer(task_type="classification", *, preprocess=True):
    return Trainer(
        task_type=task_type,
        default_imputer=ImputerConfig(imputation_method="mean" if preprocess else "none"),
        default_scaler=ScalerEncoderConfig(type="standardization" if preprocess else "none"),
        random_states=RandomStates.from_seed(43),
    )


def _expected_transform(training, other):
    """Independent arithmetic oracle; never read fitted sklearn internals."""
    mean = np.nanmean(training, axis=0)
    filled = np.where(np.isnan(training), mean, training)
    scale = np.std(filled, axis=0)
    scale[scale == 0] = 1
    return (np.where(np.isnan(other), mean, other) - filled.mean(axis=0)) / scale


@pytest.mark.parametrize("method", ["grid", "optuna"])
def test_tuning_selects_maximum_on_common_fold_local_transforms_then_refits_once(monkeypatch, model_audit, method):
    """A shifted validation/holdout cannot set imputation or scaling statistics."""
    trainer = make_trainer()
    studies = []
    if method == "optuna":
        # Exhaustive scheduling makes both controlled candidates inevitable;
        # objective execution, scoring, study direction and refit remain real.
        monkeypatch.setattr(
            trainer,
            "_build_optuna_sampler",
            lambda tuning: optuna.samplers.GridSampler({"quality": [-1, 1]}, seed=3),
        )
        create_study = optuna.create_study

        def observe_study(**kwargs):
            study = create_study(**kwargs)
            studies.append(study)
            return study

        monkeypatch.setattr(optuna, "create_study", observe_study)
    data = training_bundle()
    outcome = trainer.train_evaluate_model(model_config(method=method), data)
    tuning = outcome.result.tuning_result
    assert tuning.best_params == {"quality": 1}
    if studies:
        assert studies[0].direction == optuna.study.StudyDirection.MAXIMIZE
        assert studies[0].best_params == {"quality": 1}
        assert sorted(trial.value for trial in studies[0].trials) == [0.0, 1.0]
    assert {(fold.model_params["quality"], fold.metrics.accuracy) for fold in tuning.fold_results} == {
        (-1, 0.0),
        (1, 1.0),
    }
    folds = list(StratifiedKFold(n_splits=2, shuffle=True, random_state=43).split(data.train_data.X, data.train_data.y))
    X = data.train_data.X.to_numpy()
    assert len(model_audit.fits) == 5  # Four fold fits plus exactly one full-data refit.
    for candidate in (model_audit.fits[:2], model_audit.fits[2:4]):
        for observation, (train_rows, valid_rows) in zip(candidate, folds, strict=True):
            np.testing.assert_array_equal(observation["y"], data.train_data.y.iloc[train_rows])
            np.testing.assert_allclose(observation["X"], _expected_transform(X[train_rows], X[train_rows]))
            assert len(observation["predictions"]) == 1
            expected_validation = _expected_transform(X[train_rows], X[valid_rows])
            np.testing.assert_allclose(observation["predictions"][0], expected_validation)
            assert not np.allclose(expected_validation, _expected_transform(X, X[valid_rows]))
    final = model_audit.fits[-1]
    assert final["quality"] == 1
    np.testing.assert_array_equal(final["y"], data.train_data.y)
    np.testing.assert_allclose(final["X"], _expected_transform(X, X))
    assert len(final["predictions"]) == 2
    for actual, held_out in zip(final["predictions"], (data.test_mimic, data.test_tudd), strict=True):
        np.testing.assert_allclose(actual, _expected_transform(X, held_out.X.to_numpy()))
    # Keep the outcome, tuning/fold records and audit arrays alive during GC.
    model_audit.assert_released()


def test_shared_tuning_minimizes_error_before_unsupported_regression_reporting(model_audit):
    """One cheap regression case protects argmin; no regression E2E is claimed."""
    data = training_bundle()
    for part in (data.train_data, data.test_mimic, data.test_tudd):
        part.X = part.X[["signal"]]
        part.y = part.X["signal"].copy()
    # Production's final evaluator currently supports classification only.
    with pytest.raises(NotImplementedError, match="Only Classification supported"):
        make_trainer("regression", preprocess=False).train_evaluate_model(model_config(scoring="rmse"), data)
    assert [fit["quality"] for fit in model_audit.fits] == [-1, -1, 1, 1, 1]
    assert [len(fit["y"]) for fit in model_audit.fits] == [12, 12, 12, 12, 24]
    model_audit.assert_released()


def test_failed_cpu_fit_does_not_import_gpu_backend_during_cleanup(monkeypatch, model_audit):
    """A first Torch import under an active exception can cache its model traceback."""
    original_import = builtins.__import__

    def no_torch_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("CPU model cleanup must not import Torch")
        return original_import(name, *args, **kwargs)

    # Model the cold-backend state even when a preceding test has used Torch.
    # Block the import as well, so no duplicate Torch module can be initialized.
    with monkeypatch.context() as cold_backend:
        cold_backend.delitem(sys.modules, "torch", raising=False)
        cold_backend.setattr(builtins, "__import__", no_torch_import)
        with pytest.raises(RuntimeError, match="injected fit failure"):
            make_trainer().train_evaluate_model(
                model_config(qualities=(1,), failure="fit", failure_rows=24), training_bundle()
            )
        model_audit.assert_released()


@pytest.mark.parametrize("invalid", ["minority", "scoring"])
def test_invalid_scientific_tuning_contract_rejected_before_allocating_models(model_audit, invalid):
    data = training_bundle()
    config = model_config()
    if invalid == "minority":
        data.train_data.y[:] = 0
        data.train_data.y.iloc[0] = 1
        message = "minority class has only 1 samples"
    else:
        config = model_config(scoring="rmse")
        message = "is for regression, not classification"
    with pytest.raises(ValueError, match=message):
        make_trainer().train_evaluate_model(config, data)
    assert not model_audit.events
