import numpy as np
import pytest

from src.utils.load_data import load_toy_data_cls, load_toy_data_reg
from src.utils.model_registry import MODEL_REGISTRY_CLS, MODEL_REGISTRY_REG


LIGHTWEIGHT_CLASSIFICATION_MODELS = ("logistic-regression", "xgboost")
LIGHTWEIGHT_REGRESSION_MODELS = ("linear-regression", "xgboost")


def _make_model(model_name, registry, task_type):
    return registry[model_name].create(task_type=task_type, params={})


def _as_numpy(predictions):
    if hasattr(predictions, "detach"):
        predictions = predictions.detach().cpu()
    return np.asarray(predictions)


def _assert_valid_fit_and_predict(model_name, model, X, y):
    fit_time = model.fit(X, y)
    predictions, predict_time = model.predict(X)

    predictions = _as_numpy(predictions)

    assert fit_time >= 0, f"{model_name} returned invalid fit time"
    assert predict_time >= 0, f"{model_name} returned invalid predict time"
    assert len(predictions) == len(X), f"{model_name} returned wrong prediction count"
    assert np.isfinite(predictions).all(), (
        f"{model_name} returned non-finite predictions"
    )


def _load_regression_data_for_model_smoke_test():
    X, y = load_toy_data_reg()
    y = ((y - y.mean()) / y.std()).astype(np.float32)

    return X, y


def test_classification_registry_entries_are_model_specs():
    assert all(spec.adapter_cls is not None for spec in MODEL_REGISTRY_CLS.values())


def test_regression_registry_entries_are_model_specs():
    assert all(spec.adapter_cls is not None for spec in MODEL_REGISTRY_REG.values())


@pytest.mark.parametrize("model_name", LIGHTWEIGHT_CLASSIFICATION_MODELS)
def test_lightweight_registered_classification_models_fit_and_predict(model_name):
    X, y = load_toy_data_cls()
    model = _make_model(model_name, MODEL_REGISTRY_CLS, "classification")

    _assert_valid_fit_and_predict(model_name, model, X, y)


@pytest.mark.parametrize("model_name", LIGHTWEIGHT_REGRESSION_MODELS)
def test_lightweight_registered_regression_models_fit_and_predict(model_name):
    X, y = _load_regression_data_for_model_smoke_test()
    model = _make_model(model_name, MODEL_REGISTRY_REG, "regression")

    _assert_valid_fit_and_predict(model_name, model, X, y)
