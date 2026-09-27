"""Standalone real-model runs are opt-in; collection never probes a GPU."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--models",
        action="store_true",
        default=False,
        help="Run real registered-model fit/predict smokes (may download weights and require a GPU)",
    )


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="Real model smoke tests require explicit --models")
    for item in items:
        if item.get_closest_marker("model") is not None:
            if not config.getoption("--models"):
                item.add_marker(skip)
            elif getattr(config.option, "numprocesses", None):
                raise pytest.UsageError("Real model smokes must run sequentially; remove pytest-xdist -n")


@pytest.fixture
def model_gpu(request):
    """Probe hardware only inside a selected GPU smoke; broken dependencies fail."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip(f"{request.node.callspec.id}: GPU smoke profile requires a CUDA/HIP device")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old_threads)
