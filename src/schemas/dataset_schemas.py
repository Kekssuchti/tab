from typing import Literal

from pydantic import Field, field_validator

from src.config import config
from src.schemas.base_schemas import StrictParams
from src.schemas.preprocessing_schemas import (
    ImputerParams,
    PreprocessorParams,
    ScalerEncoderParams,
)

DatasetName = Literal["mimic", "tudd", "mimic_readmission", "tudd_readmission"]
Target = Literal["mortality", "LOS", "hours_to_readmit"]


class DataSplitParams(StrictParams):
    dataset: DatasetName
    fraction: float | int = Field(default=1.0, gt=0)

    @field_validator("fraction")
    @classmethod
    def valid_fraction(cls, v: float | int) -> float | int:
        if isinstance(v, float) and v > 1.0:
            raise ValueError("fraction must be <= 1.0 when float")
        return v


class DatasetParams(StrictParams):
    target: Target
    random_state: int = Field(default=config.seed)
    train_size: float = Field(default=0.8, gt=0, lt=1)
    train_on: tuple[DataSplitParams, ...]
    classification: bool = Field(default=True)
    preprocessor: PreprocessorParams = Field(default_factory=PreprocessorParams)
    force_repreprocess: bool = Field(
        default=False,
        description="Forces reprocessing from extracted to filtered if true",
    )

    # WIP!
    scaler_encoder: ScalerEncoderParams = Field(default_factory=ScalerEncoderParams)
    imputer: ImputerParams = Field(default_factory=ImputerParams)
