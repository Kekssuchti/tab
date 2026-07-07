from typing import Literal

from pydantic import Field

from src.schemas.base_schemas import StrictConfig


class ScalerEncoderConfig(StrictConfig):
    type: Literal["standardization", "none"] = "standardization"


class ImputerConfig(StrictConfig):
    imputation_method: Literal["knn", "mean", "median", "none"] = "knn"
    flag_missing: bool = False
    knn_neighbors: int = Field(default=5, ge=1)
