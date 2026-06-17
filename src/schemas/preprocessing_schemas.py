from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from src.schemas.base_schemas import StrictParams


class PreprocessorParams(StrictParams):
    steps: tuple[str, ...] = (
        "remove_minors",
        "adapt_max_age",
        "outlier_removal",
    )
    bmi_limit: float = Field(default=100, gt=0)
    min_age: int = Field(default=18, ge=0)
    max_age: int = Field(default=90, ge=0)
    outlier_limits: Path | None = Path("new_extended_limits.json")
    missing_rate: float = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="after")
    def validate_age_range(self) -> "PreprocessorParams":
        if self.min_age > self.max_age:
            raise ValueError("min_age must be less than or equal to max_age")
        return self


class ScalerEncoderParams(StrictParams):
    pass


class ImputerParams(StrictParams):
    imputation_method: Literal["knn", "mean", "median"] = "knn"
    flag_missing: bool = False
