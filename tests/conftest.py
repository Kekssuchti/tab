"""Pytest configuration: shared markers and skip hooks."""

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip tests marked `gpu` when no CUDA/HIP device is available."""
    if _cuda_available():
        return
    skip = pytest.mark.skip(reason="GPU required but no CUDA/HIP device is available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()
