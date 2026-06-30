import importlib
import sys

import numpy as np
import pytest

from src.schemas.training_schemas import ModelParams
from src.utils import model_registry
from src.utils.load_data import load_toy_data_cls, load_toy_data_reg
from src.utils.model_lifecycle import release_model

ADAPTER_MODULES = {
    "src.adapter.sklearn_adapter",
    "src.adapter.tabpfn_adapter",
    "src.adapter.tabicl_adapter",
    "src.adapter.limix_adapter",
    "src.adapter.mitra_adapter",
    "src.adapter.orion_msp_adapter",
    "src.adapter.orion_bix_adapter",
}
CLASSIFICATION_MODELS = [name for name in model_registry.MODEL_REGISTRY_CLS]
LIGHTWEIGHT_REGRESSION_MODELS = ["xgboost"]


def _make_model(model_name, task_type):
    spec = model_registry.get_model_spec(
        ModelParams(name=model_name, task_type=task_type)
    )
    return spec.create(task_type=task_type, params={})


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


def test_model_catalog_lookup_and_search_spaces_are_lazy():
    for module_name in ADAPTER_MODULES:
        sys.modules.pop(module_name, None)

    importlib.reload(model_registry)

    spec = model_registry.get_model_spec(
        ModelParams(name="logistic-regression", task_type="classification")
    )
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
        model_registry.get_model_spec(
            ModelParams(name="linear-regression", task_type="classification")
        )


def test_model_spec_uses_explicit_tuning_grid_as_the_search_space():
    spec = model_registry.get_model_spec(
        ModelParams(name="logistic-regression", task_type="classification")
    )

    assert spec.tuning_candidates(None, {"C": [0.1, 1.0]}) == [
        {"C": 0.1},
        {"C": 1.0},
    ]
    with pytest.raises(ValueError, match="non-empty grid"):
        spec.tuning_candidates(None, {})
    with pytest.raises(ValueError, match="values must be non-empty"):
        spec.tuning_candidates(None, {"C": []})


def test_model_spec_expands_nested_tuning_grid_keys():
    spec = model_registry.get_model_spec(
        ModelParams(name="tabpfn-3", task_type="classification")
    )

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


@pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
def test_registered_classification_models_fit_and_predict(model_name):
    X, y = load_toy_data_cls()
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
