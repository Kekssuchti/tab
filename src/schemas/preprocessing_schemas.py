from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from src.config import config
from src.schemas.base_schemas import StrictParams


class PreprocessorParams(StrictParams):
    outlier_limits_path: Path = Path(config.dir_configs / "data_limits.json")
    missing_threshold_row: float = Field(default=0.5, ge=0, le=1)
    missing_threshold_col: float = Field(default=0.5, ge=0, le=1)


class ScalerEncoderParams(StrictParams):
    pass


class ImputerParams(StrictParams):
    imputation_method: Literal["knn", "mean", "median"] = "knn"
    flag_missing: bool = False
