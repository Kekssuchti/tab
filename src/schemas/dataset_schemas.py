from typing import Literal

from pydantic import Field

from src.config import config
from src.schemas.base_schemas import StrictParams
from src.schemas.preprocessing_schemas import (
    ImputerParams,
    PreprocessorParams,
    ScalerEncoderParams,
)

DatasetName = Literal["mimic", "tudd"]


class DataSplitParams(StrictParams):
    dataset: DatasetName
    fraction: float = Field(default=1.0, gt=0)


class DatasetParams(StrictParams):
    random_state: int = Field(default=config.seed)
    train_size: float = Field(default=0.8, gt=0, lt=1)
    train_on: tuple[DataSplitParams, ...]
    classification: bool = Field(default=True)

    # WIP!
    preprocessor: PreprocessorParams = Field(default_factory=PreprocessorParams)
    scaler_encoder: ScalerEncoderParams = Field(default_factory=ScalerEncoderParams)
    imputer: ImputerParams = Field(default_factory=ImputerParams)
