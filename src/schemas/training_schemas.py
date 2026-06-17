from typing import Any, Literal

from pydantic import Field

from src.schemas.base_schemas import StrictParams


class HPOParams(StrictParams):
    search_grid: dict[str, Any]


class ModelParams(StrictParams):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    task_type: Literal["classification", "regression"] = "classification"
    optimize_hyperparameters: bool = False
    hyperparameter_optimization_params: HPOParams | None = None


class TrainingParams(StrictParams):
    models: tuple[ModelParams, ...] = (ModelParams(name="tabpfn-3"),)
