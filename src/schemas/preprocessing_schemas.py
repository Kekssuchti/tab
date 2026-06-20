from typing import Literal

from pydantic import Field

from src.schemas.base_schemas import StrictParams


class ScalerEncoderParams(StrictParams):
    type: Literal["standardization", "none"] = "standardization"


class ImputerParams(StrictParams):
    imputation_method: Literal["knn", "mean", "median", "none"] = "knn"
    flag_missing: bool = False
    knn_neighbors: int = Field(default=5, ge=1)
