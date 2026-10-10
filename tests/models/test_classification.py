"""Real adapters, not stand-ins. Run sequentially with ``pytest tests --models -m model``.

TFM profiles intentionally use CUDA (also PyTorch's ROCm device spelling). Missing
hardware skips only that profile; dependency, weight-access and inference failures
are failures. No backend is imported until its individual case executes.
"""

from copy import deepcopy

import pytest

# Explicit profiles force every new registered classifier to receive a considered
# budget rather than accidentally running an expensive estimator's defaults.
SMOKE_PARAMS = {
    "logistic-regression": {"max_iter": 100, "solver": "lbfgs"},
    "xgboost": {"n_estimators": 20, "max_depth": 2, "n_jobs": 1, "device": "cpu"},
    "ebm": {"max_rounds": 4, "outer_bags": 1, "interactions": 0, "n_jobs": 1},
    "tabpfn-3": {"n_estimators": 1, "device": "cuda", "n_preprocessing_jobs": 1},
    "tabpfn-3.5-fast": {"n_estimators": 1, "device": "cuda", "n_preprocessing_jobs": 1},
    "tabpfn-3.5": {"n_estimators": 1, "device": "cuda", "n_preprocessing_jobs": 1},
    "tabicl-2": {"n_estimators": 1, "device": "cuda", "n_jobs": 1},
    "tabswift": {"n_estimators": 1, "batch_size": 1, "device": "cuda"},
    "exaone": {"ensemble_count": 1, "device": "cuda"},
    "causilo": {"n_estimators": 1, "device": "cuda"},
    "tabdpt": {"n_ensembles": 1, "device": "cuda"},
    "kumo_tabular": {"n_estimators": 1, "size": "small", "device": "cuda"},
    "tabfm": {"n_estimators": 1, "predict_batch_size": 12},
}
CPU_MODELS = {"logistic-regression", "xgboost", "ebm"}


def test_every_registered_classifier_has_an_explicit_smoke_budget():
    """Default CI catches an omitted model without importing any of its backends."""
    from src.utils.model_registry import MODEL_CATALOG

    registered = set(MODEL_CATALOG.available_models("classification"))
    configured = set(SMOKE_PARAMS)
    assert configured == registered, f"Missing profiles: {registered - configured}; obsolete: {configured - registered}"


@pytest.mark.model
@pytest.mark.parametrize("model_name", SMOKE_PARAMS)
def test_real_classifier_returns_binary_probabilities_on_unseen_rows(model_name, request):
    if model_name not in CPU_MODELS:
        request.getfixturevalue("model_gpu")

    import numpy as np
    from threadpoolctl import threadpool_limits

    from src.schemas.training_schemas import ModelConfig
    from src.utils.model_lifecycle import release_model
    from src.utils.model_registry import get_model_spec
    from tests.toy_data import model_smoke_data

    X_train, y_train, X_test = model_smoke_data()
    assert X_train.index.intersection(X_test.index).empty
    assert not set(map(tuple, X_train.to_numpy())) & set(map(tuple, X_test.to_numpy()))
    spec = get_model_spec(ModelConfig(name=model_name), "classification")
    model = None
    with threadpool_limits(limits=1):
        try:
            model = spec.create(
                "classification", deepcopy(SMOKE_PARAMS[model_name]), random_state=17, inference_state=19
            )
            model.fit(X_train, y_train)
            probabilities = model.predict(X_test).values
            assert isinstance(probabilities, np.ndarray)
            assert probabilities.shape == (len(X_test), 2)
            assert np.isfinite(probabilities).all()
            assert ((0 <= probabilities) & (probabilities <= 1)).all()
            np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-6)
        finally:
            release_model(model)
