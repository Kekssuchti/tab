import importlib
import sys

import numpy as np
import optuna
import pytest
from pydantic import ValidationError

from src.schemas.training_schemas import ModelConfig
from src.utils import model_registry
from src.utils.model_lifecycle import release_model
from src.utils.tuning_distributions import (
    DiscreteUniform,
    LogUniform,
    Uniform,
    UniformChoice,
)
from tests.toy_data import load_toy_classification_data, load_toy_regression_data

ADAPTER_MODULES = {
    "src.adapter.sklearn_adapter",
    "src.adapter.tabpfn_adapter",
    "src.adapter.tabicl_adapter",
    "src.adapter.limix_adapter",
    "src.adapter.mitra_adapter",
    "src.adapter.orion_msp_adapter",
    "src.adapter.orion_bix_adapter",
    "src.adapter.tabfm_adapter",
    "src.adapter.tabswift_adapter",
}
CLASSIFICATION_MODELS = [name for name in model_registry.MODEL_REGISTRY_CLS]
LIGHTWEIGHT_REGRESSION_MODELS = ["xgboost", "tabswift"]


@pytest.mark.parametrize("removed_field", ["task_type", "params"])
def test_model_config_rejects_removed_fields(removed_field):
    with pytest.raises(ValidationError, match=removed_field):
        ModelConfig.model_validate({"name": "logistic-regression", removed_field: {}})


def test_regression_catalog_excludes_classification_only_adapters():
    regression_models = set(model_registry.MODEL_CATALOG.available_models("regression"))

    assert {"orion-msp", "orion-bix", "tabfm"}.isdisjoint(regression_models)
    assert model_registry.MODEL_REGISTRY_REG["limix-2m"].search_spaces


def _make_model(model_name, task_type):
    spec = model_registry.get_model_spec(ModelConfig(name=model_name), task_type)
    return spec.create(task_type=task_type, params={})


def _assert_valid_fit_and_predict(model_name, model, X, y, task_type):
    fit_time = model.fit(X, y)
    prediction = model.predict(X)

    predictions = prediction.values

    assert fit_time >= 0, f"{model_name} returned invalid fit time"
    assert prediction.seconds >= 0, f"{model_name} returned invalid predict time"
    assert isinstance(predictions, np.ndarray), f"{model_name} must normalize predictions to numpy"
    assert len(predictions) == len(X), f"{model_name} returned wrong prediction count"
    if task_type == "classification":
        assert predictions.ndim == 2, f"{model_name} must return class probabilities"
    assert np.isfinite(predictions).all(), f"{model_name} returned non-finite predictions"


def _load_regression_data_for_model_smoke_test():
    X, y = load_toy_regression_data()
    y = ((y - y.mean()) / y.std()).astype(np.float32)

    return X, y


def test_model_catalog_lookup_and_search_spaces_are_lazy():
    for module_name in ADAPTER_MODULES:
        sys.modules.pop(module_name, None)

    importlib.reload(model_registry)

    spec = model_registry.get_model_spec(ModelConfig(name="logistic-regression"), "classification")
    candidates = spec.tuning_candidates(search_space=None, overrides=None)

    assert ADAPTER_MODULES.isdisjoint(sys.modules)
    assert spec.adapter_path == "src.adapter.sklearn_adapter:LinearModelAdapter"
    # dont want exact candidates / counts but should have some
    assert len(candidates) > 5


def test_model_catalog_reports_task_specific_unknown_models():
    with pytest.raises(
        ValueError,
        match="Unknown classification model 'linear-regression'.*logistic-regression",
    ):
        model_registry.get_model_spec(ModelConfig(name="linear-regression"), "classification")


def test_model_spec_uses_explicit_tuning_grid_as_the_search_space():
    spec = model_registry.get_model_spec(ModelConfig(name="logistic-regression"), "classification")

    assert spec.tuning_candidates(None, {"C": [0.1, 1.0]}) == [
        {"C": 0.1},
        {"C": 1.0},
    ]
    with pytest.raises(ValueError, match="non-empty grid"):
        spec.tuning_candidates(None, {})
    with pytest.raises(ValueError, match="values must be non-empty"):
        spec.tuning_candidates(None, {"C": []})


def test_model_spec_expands_nested_tuning_grid_keys():
    spec = model_registry.get_model_spec(ModelConfig(name="tabpfn-3"), "classification")

    assert spec.tuning_candidates(
        None,
        {
            "inference_config.SUBSAMPLE_SAMPLES": [128, 256],
            "inference_config.FINGERPRINT_FEATURE": [True],
            "inference_config.PREPROCESS_TRANSFORMS.name": ["power"],
            "inference_config.PREPROCESS_TRANSFORMS.categorical_name": ["none"],
        },
    ) == [
        {
            "inference_config": {
                "SUBSAMPLE_SAMPLES": 128,
                "FINGERPRINT_FEATURE": True,
                "PREPROCESS_TRANSFORMS": {
                    "name": "power",
                    "categorical_name": "none",
                },
            }
        },
        {
            "inference_config": {
                "SUBSAMPLE_SAMPLES": 256,
                "FINGERPRINT_FEATURE": True,
                "PREPROCESS_TRANSFORMS": {
                    "name": "power",
                    "categorical_name": "none",
                },
            }
        },
    ]


def test_model_spec_samples_mixed_optuna_search_space():
    spec = model_registry.ModelSpec(
        adapter_path="unused:Adapter",
        search_spaces={
            "mixed": {
                "uniform": Uniform(0.0, 1.0),
                "discrete": DiscreteUniform(0.0, 1.0, step=0.25),
                "log_uniform": LogUniform(1e-3, 1.0),
                "nested.category": ["first", "second"],
            }
        },
    )
    search_space = spec.tuning_search_space("mixed", overrides=None)
    trial = optuna.trial.FixedTrial(
        {
            "uniform": 0.4,
            "discrete": 0.5,
            "log_uniform": 0.1,
            "nested.category": "second",
        }
    )

    candidate = spec.sample_tuning_candidate(trial, search_space)

    assert candidate == {
        "uniform": 0.4,
        "discrete": 0.5,
        "log_uniform": 0.1,
        "nested": {"category": "second"},
    }
    assert trial.distributions["uniform"] == optuna.distributions.FloatDistribution(0.0, 1.0)
    assert trial.distributions["discrete"] == optuna.distributions.FloatDistribution(0.0, 1.0, step=0.25)
    assert trial.distributions["log_uniform"] == optuna.distributions.FloatDistribution(1e-3, 1.0, log=True)
    assert isinstance(
        trial.distributions["nested.category"],
        optuna.distributions.CategoricalDistribution,
    )


def test_uniform_choice_returns_constant_without_sampling_nested_distribution():
    distribution = UniformChoice(0.0, LogUniform(1e-16, 1e2))
    trial = optuna.trial.FixedTrial({"regularization.__uniform_choice": 0})

    value = distribution.suggest(trial, "regularization")

    assert value == 0.0
    assert trial.params == {"regularization.__uniform_choice": 0}


def test_uniform_choice_samples_selected_nested_distribution():
    distribution = UniformChoice(0.0, LogUniform(1e-16, 1e2))
    trial = optuna.trial.FixedTrial(
        {
            "regularization.__uniform_choice": 1,
            "regularization.__uniform_choice_1": 0.25,
        }
    )

    value = distribution.suggest(trial, "regularization")

    assert value == 0.25
    assert trial.distributions["regularization.__uniform_choice_1"] == optuna.distributions.FloatDistribution(
        1e-16, 1e2, log=True
    )


def test_uniform_choice_supports_recursive_choices():
    distribution = UniformChoice(
        0.0,
        UniformChoice(Uniform(0.1, 1.0), LogUniform(1e-3, 1.0)),
    )
    trial = optuna.trial.FixedTrial(
        {
            "value.__uniform_choice": 1,
            "value.__uniform_choice_1.__uniform_choice": 1,
            "value.__uniform_choice_1.__uniform_choice_1": 0.1,
        }
    )

    value = distribution.suggest(trial, "value")

    assert value == 0.1


def test_grid_tuning_rejects_distribution_search_space():
    spec = model_registry.ModelSpec(
        adapter_path="unused:Adapter",
        search_spaces={"continuous": {"learning_rate": LogUniform(1e-3, 1.0)}},
    )

    with pytest.raises(
        ValueError,
        match="Grid tuning cannot expand distribution domains: learning_rate",
    ):
        spec.tuning_candidates("continuous", overrides=None)


@pytest.mark.parametrize(
    ("distribution", "message"),
    [
        (lambda: Uniform(1.0, 1.0), "low must be less than high"),
        (lambda: LogUniform(0.0, 1.0), "bounds must be greater than zero"),
        (lambda: DiscreteUniform(0.0, 1.0, 0.3), "evenly divisible"),
        (lambda: DiscreteUniform(0.0, 1.0, 0.0), "greater than zero"),
        (lambda: UniformChoice(), "at least one choice"),
    ],
)
def test_tuning_distributions_reject_invalid_domains(distribution, message):
    with pytest.raises(ValueError, match=message):
        distribution()


@pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
def test_registered_classification_models_fit_and_predict(model_name):
    X, y = load_toy_classification_data()
    model = _make_model(model_name, "classification")

    try:
        _assert_valid_fit_and_predict(model_name, model, X, y, "classification")
    finally:
        release_model(model)


# keep regression lightweight since its not really used yet
@pytest.mark.parametrize("model_name", LIGHTWEIGHT_REGRESSION_MODELS)
def test_lightweight_registered_regression_models_fit_and_predict(model_name):
    X, y = _load_regression_data_for_model_smoke_test()
    model = _make_model(model_name, "regression")

    try:
        _assert_valid_fit_and_predict(model_name, model, X, y, "regression")
    finally:
        release_model(model)
