"""Kumo tensor API integration without checkpoints or GPU allocations."""

import numpy as np
import pytest


@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_kumo_converts_arrays_at_real_fit_boundary(monkeypatch, task_type):
    import sdm
    import torch
    from sdm.models.base import ICLModel
    from sdm.processing.execution import RecipeExecution

    from src.adapter.kumo_tabular_adapter import KumoTabularAdapter

    # Use Kumo's real fit and recipe; stop before executing the neural network.
    model = sdm.models.KumoTabular.__new__(sdm.models.KumoTabular)
    ICLModel.__init__(model, task=task_type)
    model.eval()
    monkeypatch.setattr(sdm.models, "KumoTabular", lambda **kwargs: model)
    original_transform = RecipeExecution.fit_transform
    observations = []

    class ContextReady(Exception):
        pass

    def inspect_context(self, x, y, related_tables, **kwargs):
        assert isinstance(x, torch.Tensor)
        assert x.device == torch.device("cpu")
        assert x.is_floating_point()
        assert y.shape == (4, 1)
        assert y.is_floating_point() == (task_type == "regression")
        assert kwargs["num_members"] == 2
        assert kwargs["generator"].initial_seed() == 17
        contexts = original_transform(self, x, y, related_tables, **kwargs)
        assert len(contexts) == 2
        observations.append(True)
        raise ContextReady

    monkeypatch.setattr(RecipeExecution, "fit_transform", inspect_context)
    adapter = KumoTabularAdapter(task_type=task_type, device="cpu", n_estimators=2, random_state=17)
    X = np.arange(8, dtype=float).reshape(4, 2)
    with pytest.raises(ContextReady):
        adapter.fit(X, np.array([0, 1, 0, 1]))
    assert observations == [True]

    def predict(x):
        assert isinstance(x, torch.Tensor)
        assert x.device == torch.device("cpu")
        if task_type == "classification":
            return sdm.TableTensor(columns={"numerical": ["0", "1"]}, numerical=torch.tensor([[0.2, 0.8]]))
        return sdm.TableTensor(columns={"numerical": ["q001", "q999"]}, numerical=torch.tensor([[2.0, 4.0]]))

    monkeypatch.setattr(model, "predict", predict)
    prediction = adapter.predict(X[:1])
    np.testing.assert_allclose(prediction.values, [[0.2, 0.8]] if task_type == "classification" else [3.0])
    assert prediction.seconds >= 0
    adapter.release()
    assert adapter.model is None
    assert model._cache is None
