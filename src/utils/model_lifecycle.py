import gc
from typing import Any


def release_model(model: Any) -> None:
    """Release model-owned resources and clear cached CUDA allocations."""
    if model is None:
        return

    try:
        release = getattr(model, "release", None)
        if callable(release):
            release()
        else:
            _release_model_attr(model)
    finally:
        gc.collect()
        _empty_cuda_cache()


def _release_model_attr(owner: Any) -> None:
    estimator = getattr(owner, "model", None)
    if estimator is None:
        return

    close = getattr(estimator, "close", None)
    if callable(close):
        close()

    cpu = getattr(estimator, "cpu", None)
    if callable(cpu):
        cpu()

    try:
        owner.model = None
    except AttributeError:
        pass


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except (ImportError, RuntimeError):
        return
