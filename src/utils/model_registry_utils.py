from collections.abc import Mapping, Sequence
from functools import lru_cache
from importlib import import_module
from itertools import product
from typing import Any

from src.utils.logger import logger
from src.utils.tuning_distributions import (
    OptunaDistribution,
)

SearchDomain = Sequence[Any] | OptunaDistribution


def _copy_search_space(
    search_space: Mapping[str, SearchDomain],
) -> dict[str, SearchDomain]:
    copied = {}
    for key, domain in search_space.items():
        copied[key] = domain if isinstance(domain, OptunaDistribution) else list(domain)
    return copied


def _expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        raise ValueError("Tuning requires a non-empty grid")

    keys = list(grid)
    values = [grid[key] for key in keys]
    if any(not value for value in values):
        raise ValueError("Tuning grid values must be non-empty")

    return [
        _expand_candidate(dict(zip(keys, combination)))
        for combination in product(*values)
    ]


def _expand_candidate(values: Mapping[str, Any]) -> dict[str, Any]:
    candidate: dict[str, Any] = {}
    for key, value in values.items():
        _set_nested_value(candidate, key, value)
    return candidate


def _set_nested_value(candidate: dict[str, Any], key: str, value: Any) -> None:
    # we define nested splits via '.'
    # e.g. TabPFN accepts inference_config{nested_param=123}
    # to iterate and have multiple values use: inference_config.nested_param: [123, 124, ...]

    parts = key.split(".")
    if any(part == "" for part in parts):
        raise ValueError(f"Tuning grid key '{key}' contains an empty path segment")

    current = candidate
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        existing = current[part]
        if not isinstance(existing, dict):
            raise ValueError(
                f"Tuning grid key '{key}' conflicts with non-nested parameter '{part}'"
            )
        current = existing

    final_key = parts[-1]
    if final_key in current:
        raise ValueError(f"Tuning grid key '{key}' is duplicated")
    current[final_key] = value


@lru_cache
def _load_adapter_cls(adapter_path: str):
    # really complicated looking for what it does
    # basically just lazy load all adapters only when we need them to reduce startup time

    module_name, class_name = adapter_path.split(":", maxsplit=1)
    logger.info(f"Loading model adapter: {adapter_path}")
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Could not import model adapter module '{module_name}'"
        ) from exc
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Model adapter '{class_name}' not found in '{module_name}'"
        ) from exc
