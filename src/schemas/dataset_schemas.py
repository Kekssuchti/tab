from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import Field, field_validator

from src.config import config
from src.schemas.base_schemas import StrictParams
from src.schemas.preprocessing_schemas import (
    ImputerParams,
    ScalerEncoderParams,
)

DatasetName = Literal["mimic", "tudd", "mimic_readmission", "tudd_readmission"]
Target = Literal["mortality", "LOS", "hours_to_readmit"]


@dataclass
class XYDataset:
    X: pd.DataFrame
    y: pd.Series


@dataclass
class DatasetBundle:
    train_data: XYDataset
    test_mimic: XYDataset
    test_tudd: XYDataset


@dataclass(frozen=True)
class DatasetPartSummary:
    row_count: int
    class_balance: dict[str, int]


@dataclass(frozen=True)
class DatasetFileSummary:
    dataset_name: str
    data_origin: str
    file_name: str
    path: str
    sha256: str | None


@dataclass(frozen=True)
class DatasetSummary:
    target: Target
    train: DatasetPartSummary
    test_mimic: DatasetPartSummary
    test_tudd: DatasetPartSummary
    data_files: tuple[DatasetFileSummary, ...]


class DataCleanerParams(StrictParams):
    outlier_limits_path: Path = Path(config.dir_configs / "data_limits.json")
    missing_threshold_row: float = Field(default=0.5, ge=0, le=1)


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
    data_cleaner: DataCleanerParams = Field(default_factory=DataCleanerParams)
    force_repreprocess: bool = Field(
        default=False,
        description="Forces reprocessing from extracted to filtered if true",
    )

    # WIP!
    scaler_encoder: ScalerEncoderParams = Field(default_factory=ScalerEncoderParams)
    imputer: ImputerParams = Field(default_factory=ImputerParams)
