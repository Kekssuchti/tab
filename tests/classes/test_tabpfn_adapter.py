import pytest

from src.adapter.tabpfn_adapter import TabPFNAdapter


@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_tabpfn_adapter_passes_configured_params_to_default_version(task_type):
    adapter = TabPFNAdapter(
        task_type=task_type,
        random_state=1337,
        fit_mode="fit_with_cache",
        predict_batch_size=256,
        show_progress_bar=False,
    )

    model_params = adapter.model.get_params()

    assert adapter.predict_batch_size == 256
    assert model_params["random_state"] == 1337
    assert model_params["fit_mode"] == "fit_with_cache"
