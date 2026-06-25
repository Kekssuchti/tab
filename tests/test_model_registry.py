import importlib
import sys

import numpy as np
import pytest

from src.utils.load_data import load_toy_data_cls, load_toy_data_reg
from src.utils.model_lifecycle import release_model
from src.utils import model_registry

LIGHTWEIGHT_CLASSIFICATION_MODELS = ("logistic-regression", "xgboost")
LIGHTWEIGHT_REGRESSION_MODELS = ("linear-regression", "xgboost")


def _make_model(model_name, registry, task_type):
    return registry[model_name].create(task_type=task_type, params={})


def _as_numpy(predictions):
    if hasattr(predictions, "detach"):
        predictions = predictions.detach().cpu()
    return np.asarray(predictions)


def _assert_valid_fit_and_predict(model_name, model, X, y, task_type):
    fit_time = model.fit(X, y)
    predictions, predict_time = model.predict(X)

    predictions = _as_numpy(predictions)

    assert fit_time >= 0, f"{model_name} returned invalid fit time"
    assert predict_time >= 0, f"{model_name} returned invalid predict time"
    assert len(predictions) == len(X), f"{model_name} returned wrong prediction count"
    if task_type == "classification":
        assert predictions.ndim == 2, f"{model_name} must return class probabilities"
    assert np.isfinite(predictions).all(), (
        f"{model_name} returned non-finite predictions"
    )


def _load_regression_data_for_model_smoke_test():
    X, y = load_toy_data_reg()
    y = ((y - y.mean()) / y.std()).astype(np.float32)

    return X, y


def test_model_registry_import_does_not_load_adapter_modules():
    adapter_modules = {
        "src.adapter.sklearn_adapter",
        "src.adapter.tabpfn_adapter",
        "src.adapter.tabicl_adapter",
        "src.adapter.limix_adapter",
        "src.adapter.mitra_adapter",
        "src.adapter.orion_msp_adapter",
        "src.adapter.orion_bix_adapter",
    }
    for module_name in adapter_modules:
        sys.modules.pop(module_name, None)

    importlib.reload(model_registry)

    assert adapter_modules.isdisjoint(sys.modules)


def test_classification_registry_entries_are_lazy_model_specs():
    assert all(
        spec.adapter_path.startswith("src.adapter.")
        for spec in model_registry.MODEL_REGISTRY_CLS.values()
    )


def test_regression_registry_entries_are_lazy_model_specs():
    assert all(
        spec.adapter_path.startswith("src.adapter.")
        for spec in model_registry.MODEL_REGISTRY_REG.values()
    )


@pytest.mark.parametrize("model_name", LIGHTWEIGHT_CLASSIFICATION_MODELS)
def test_lightweight_registered_classification_models_fit_and_predict(model_name):
    X, y = load_toy_data_cls()
    model = _make_model(model_name, model_registry.MODEL_REGISTRY_CLS, "classification")

    try:
        _assert_valid_fit_and_predict(model_name, model, X, y, "classification")
    finally:
        release_model(model)


@pytest.mark.parametrize("model_name", LIGHTWEIGHT_REGRESSION_MODELS)
def test_lightweight_registered_regression_models_fit_and_predict(model_name):
    X, y = _load_regression_data_for_model_smoke_test()
    model = _make_model(model_name, model_registry.MODEL_REGISTRY_REG, "regression")

    try:
        _assert_valid_fit_and_predict(model_name, model, X, y, "regression")
    finally:
        release_model(model)
