from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import Field, field_validator, model_validator

from src.config import config
from src.schemas.base_schemas import StrictConfig
from src.schemas.preprocessing_schemas import (
    ImputerConfig,
    ScalerEncoderConfig,
)

DatasetName = Literal["mimic", "tudd", "mimic_readmission", "tudd_readmission"]
DatasetOrigin = Literal["mimic", "tudd"]
DatasetKind = Literal["normal", "readmission"]
Target = Literal["mortality", "LOS7", "hours_to_readmit", "LOS", "hours_to_readmit_72"]


@dataclass(frozen=True)
class ClassificationTargetSummary:
    """Label counts for a classification target."""

    class_balance: dict[str, int]


@dataclass(frozen=True)
class RegressionTargetSummary:
    """Finite descriptive statistics for a regression target."""

    count: int
    mean: float
    std: float
    min: float
    max: float


TargetSummary = ClassificationTargetSummary | RegressionTargetSummary


@dataclass(frozen=True)
class DatasetPartSummary:
    """Row and target summary for one dataset split."""

    row_count: int
    target_summary: TargetSummary


@dataclass(frozen=True)
class DatasetFileSummary:
    """
    Provenance summary for one filtered data file.

    ---
    Attributes:
        dataset_name: str
            Config dataset name that used the file.

        data_origin: str
            Source system, such as mimic or tudd.

        file_name: str
            File name of the filtered dataset.

        path: str
            Full path to the filtered dataset.

        sha256: str or None
            File content hash, or None when unavailable.
    """

    dataset_name: str
    data_origin: str
    file_name: str
    path: str
    sha256: str | None


@dataclass(frozen=True)
class DatasetSummary:
    """
    Dataset summary attached to a completed pipeline run.

    ---
    Attributes:
        target: {"mortality", "LOS7", "hours_to_readmit", "LOS"}
            Prediction target used by the run.

        train: DatasetPartSummary
            Summary of the combined training data.

        test_mimic: DatasetPartSummary
            Summary of the held-out MIMIC test data.

        test_tudd: DatasetPartSummary
            Summary of the held-out TUDD test data.

        data_files: tuple of DatasetFileSummary
            Source files used to build the dataset.
    """

    target: Target
    train: DatasetPartSummary
    test_mimic: DatasetPartSummary
    test_tudd: DatasetPartSummary
    data_files: tuple[DatasetFileSummary, ...]


class DataCleanerConfig(StrictConfig):
    """
    Configuration for extracted-to-filtered data cleaning.

    ---
    Attributes:
        outlier_limits_path: Path, default=configs/data_limits.json
            JSON file defining valid ranges for clinical variables.

        missing_threshold_row: float, default=0.5
            Maximum allowed missing-value fraction per row.
    """

    outlier_limits_path: Path = Path(config.dir_configs / "data_limits.json")
    missing_threshold_row: float = Field(default=0.5, ge=0, le=1)


class DataSplitConfig(StrictConfig):
    """
    Training contribution from one dataset.

    ---
    Attributes:
        dataset: {"mimic", "tudd"}
            Dataset used as a training source.

        fraction: float or int, default=1.0
            Fraction of that training split, or absolute sample count when int.
    """

    dataset: DatasetOrigin
    fraction: float | int = Field(default=1.0, gt=0)

    @field_validator("fraction")
    @classmethod
    def valid_fraction(cls, v: float | int) -> float | int:
        if isinstance(v, float) and v > 1.0:
            raise ValueError("fraction must be <= 1.0 when float")
        return v


class DatasetConfig(StrictConfig):
    """
    Configuration for dataset construction.

    ---
    Attributes:
        target: {"mortality", "LOS7", "hours_to_readmit", "LOS"}
            Prediction target.

        random_state: int, default=config.seed
            Seed used for train-test splitting and sampling.

        train_size: float, default=0.8
            Fraction reserved for each source's train split.

        train_on: tuple of DataSplitConfig
            Dataset sources combined into the training set.

        data_cleaner: DataCleanerConfig, default=DataCleanerConfig()
            Cleaning settings for filtered data generation.

        force_repreprocess: bool, default=False
            Whether to rebuild filtered data from extracted data.

        log_transform_target: bool, default=False
            Whether to train LOS regression models on the natural log of LOS.

        scaler_encoder: ScalerEncoderConfig, default=ScalerEncoderConfig()
            Default scaling and encoding settings.

        imputer: ImputerConfig, default=ImputerConfig()
            Default missing-value imputation settings.
    """

    target: Target
    random_state: int = Field(default=config.seed)
    train_size: float = Field(default=0.8, gt=0, lt=1)
    train_on: tuple[DataSplitConfig, ...]
    data_cleaner: DataCleanerConfig = Field(default_factory=DataCleanerConfig)
    force_repreprocess: bool = Field(
        default=False,
        description="Forces reprocessing from extracted to filtered if true",
    )
    log_transform_target: bool = Field(
        default=False,
        description="Train LOS regression models in log space and evaluate in hours",
    )
    scaler_encoder: ScalerEncoderConfig = Field(default_factory=ScalerEncoderConfig)
    imputer: ImputerConfig = Field(default_factory=ImputerConfig)

    @field_validator("train_on")
    @classmethod
    def unique_origins(cls, train_on: tuple[DataSplitConfig, ...]) -> tuple[DataSplitConfig, ...]:
        origins = [split.dataset for split in train_on]
        if len(origins) != len(set(origins)):
            raise ValueError("Duplicate dataset origins detected")
        return train_on

    @model_validator(mode="after")
    def log_transform_requires_los(self):
        if self.log_transform_target and self.target != "LOS":
            raise ValueError("log_transform_target is only supported for the LOS target")
        return self


@dataclass
class XYDataset:
    """
    Feature matrix and target vector.

    ---
    Attributes:
        X: pandas.DataFrame
            Feature matrix.

        y: pandas.Series
            Target vector.
    """

    X: pd.DataFrame
    y: pd.Series


@dataclass
class DatasetBundle:
    """
    Aligned train and test datasets for a pipeline run.

    ---
    Attributes:
        train_data: XYDataset
            Combined training dataset.

        test_mimic: XYDataset
            Held-out MIMIC test dataset.

        test_tudd: XYDataset
            Held-out TUDD test dataset.
    """

    train_data: XYDataset
    test_mimic: XYDataset
    test_tudd: XYDataset
