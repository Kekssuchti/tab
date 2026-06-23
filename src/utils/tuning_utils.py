from itertools import product
from typing import Any

from sklearn.model_selection import KFold, StratifiedKFold

from src.schemas.training_schemas import ModelParams, TuningParams


def get_candidate_params(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        raise ValueError("Tuning requires a non-empty grid")

    keys = list(grid)
    values = [grid[key] for key in keys]
    if any(not value for value in values):
        raise ValueError("Tuning grid values must be non-empty")

    return [dict(zip(keys, combination)) for combination in product(*values)]


def build_cv(model_params: ModelParams, tuning: TuningParams):
    cv_cls = StratifiedKFold if model_params.task_type == "classification" else KFold
    return cv_cls(
        n_splits=tuning.cv.n_splits,
        shuffle=tuning.cv.shuffle,
        random_state=tuning.cv.random_state,
    )
