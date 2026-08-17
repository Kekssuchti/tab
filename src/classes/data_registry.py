from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import pandas as pd

from src.schemas.base_schemas import TaskType
from src.schemas.dataset_schemas import DatasetKind, DatasetName, DatasetOrigin, Target

TARGET_LIKE_COLUMNS = (
    "mortality",
    "LOS",  # regression
    "LOS3",  # binary, true if LOS <= 3 days
    "LOS7",  # binary, true if LOS > 7 days
    "hours_to_readmit",  # binary, did or didnt
    "hours_to_readmit_72",  # binary, true if readmitted within <= 72 hours
)


@dataclass(frozen=True)
class DataFile:
    """
    Registered filtered data file.

    ---
    Attributes:
        data_origin: {"mimic", "tudd"}
            Source system for the file.

        file_name: str
            File name under the filtered data directory.

        dataset_kind: {"normal", "readmission"}
            Dataset variant containing the file.
    """

    data_origin: DatasetOrigin
    dataset_kind: DatasetKind
    file_name: str

    @property
    def dataset_name(self) -> DatasetName:
        if self.dataset_kind == "readmission":
            return f"{self.data_origin}_readmission"
        return self.data_origin


@dataclass(frozen=True)
class DatasetTask:
    target: Target

    @property
    def task_type(self) -> TaskType:
        return TARGET_DEFINITIONS[self.target].task_type

    @property
    def dataset_kind(self) -> DatasetKind:
        return TARGET_DEFINITIONS[self.target].dataset_kind

    @property
    def data_files(self) -> dict[DatasetOrigin, DataFile]:
        return {data_file.data_origin: data_file for data_file in data_files_for_kind(self.dataset_kind)}

    def labels_from(self, df: pd.DataFrame) -> pd.Series:
        return TARGET_DEFINITIONS[self.target].label_deriver(df)

    def features_from(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [
            column for column in TARGET_LIKE_COLUMNS if not (self.dataset_kind == "readmission" and column == "LOS")
        ]
        return df.drop(columns=cols_to_drop, errors="ignore")


def _mortality_labels(df: pd.DataFrame) -> pd.Series:
    return df["mortality"].astype(int)


def _los7_labels(df: pd.DataFrame) -> pd.Series:
    return (df["LOS"] > 7 * 24).astype(int)


def _readmission_labels(df: pd.DataFrame) -> pd.Series:
    return df["hours_to_readmit"].notna().astype(int)


def _readmission_72_labels(df: pd.DataFrame) -> pd.Series:
    return (df["hours_to_readmit"] <= 72).astype(int)


def _los_labels(df: pd.DataFrame) -> pd.Series:
    return df["LOS"].copy()


DATA_FILES = (
    DataFile("mimic", "normal", "mimic4_mean_100_full.csv"),
    DataFile("tudd", "normal", "tudd_mean_100_full.csv"),
    DataFile("mimic", "readmission", "mimic4_readmission.csv"),
    DataFile("tudd", "readmission", "tudd_readmission.csv"),
)


def data_files_for_kind(dataset_kind: DatasetKind | None = None) -> tuple[DataFile, ...]:
    return tuple(
        data_file for data_file in DATA_FILES if dataset_kind is None or data_file.dataset_kind == dataset_kind
    )


class TargetDefinition(NamedTuple):
    task_type: TaskType
    dataset_kind: DatasetKind
    label_deriver: Callable[[pd.DataFrame], pd.Series]


TARGET_DEFINITIONS: dict[Target, TargetDefinition] = {
    "mortality": TargetDefinition("classification", "normal", _mortality_labels),
    "LOS": TargetDefinition("regression", "normal", _los_labels),
    "LOS7": TargetDefinition("classification", "normal", _los7_labels),
    "hours_to_readmit": TargetDefinition("classification", "readmission", _readmission_labels),
    "hours_to_readmit_72": TargetDefinition("classification", "readmission", _readmission_72_labels),
}


def dataset_task_for_target(target: Target) -> DatasetTask:
    return DatasetTask(target)
