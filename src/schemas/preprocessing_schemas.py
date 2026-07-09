from typing import Literal

from pydantic import Field

from src.schemas.base_schemas import StrictConfig


class ScalerEncoderConfig(StrictConfig):
    """
    Configuration for numeric scaling and categorical encoding.

    ---
    Attributes:
        type: {"standardization", "none"}, default="standardization"
            Scaling strategy for numeric features.
    """

    type: Literal["standardization", "none"] = "standardization"


class ImputerConfig(StrictConfig):
    """
    Configuration for missing-value imputation.

    ---
    Attributes:
        imputation_method: {"knn", "mean", "median", "none"}, default="knn"
            Imputation strategy for missing values.

        flag_missing: bool, default=False
            Whether to add indicators for missing values.

        knn_neighbors: int, default=5
            Number of neighbors used by KNN imputation.
    """

    imputation_method: Literal["knn", "mean", "median", "none"] = "knn"
    flag_missing: bool = False
    knn_neighbors: int = Field(default=5, ge=1)
