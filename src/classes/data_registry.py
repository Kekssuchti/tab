from dataclasses import dataclass

import pandas as pd

from src.schemas.dataset_schemas import DatasetName, DatasetOrigin, Target

TARGET_LIKE_COLUMNS = (
    "mortality",
    "LOS",
    "LOS3",
    "LOS7",
    "hours_to_readmit",
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

        is_readmission: bool, default=False
            Whether the file belongs to a readmission task.
    """

    data_origin: DatasetOrigin
    file_name: str
    is_readmission: bool = False


@dataclass(frozen=True)
class DatasetTask:
    """
    Target-specific dataset registry entry.

    ---
    Attributes:
        target: {"mortality", "LOS7", "hours_to_readmit"}
            Target column or derived label for the task.

        data_files: dict
            Mapping from dataset names to registered files.
    """

    target: Target
    data_files: dict[DatasetName, DataFile]

    def labels_from(self, df: pd.DataFrame) -> pd.Series:
        labels = df[self.target]
        if self.target == "hours_to_readmit":
            return labels.notna().astype(int)
        return labels.astype(int)

    def features_from(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=list(TARGET_LIKE_COLUMNS), errors="ignore")


DATA_FILES_NORMAL: dict[DatasetName, DataFile] = {
    "mimic": DataFile("mimic", "mimic4_mean_100_full.csv"),
    "tudd": DataFile("tudd", "tudd_mean_100_full.csv"),
}

DATA_FILES_READMISSION: dict[DatasetName, DataFile] = {
    "mimic_readmission": DataFile(
        "mimic", "mimic4_readmission.csv", is_readmission=True
    ),
    "tudd_readmission": DataFile("tudd", "tudd_readmission.csv", is_readmission=True),
}

DATA_FILES_ALL: dict[DatasetName, DataFile] = DATA_FILES_NORMAL | DATA_FILES_READMISSION

DATA_FILES_BY_TARGET: dict[Target, dict[DatasetName, DataFile]] = {
    "mortality": DATA_FILES_NORMAL,
    "LOS7": DATA_FILES_NORMAL,
    "hours_to_readmit": DATA_FILES_READMISSION,
}


def dataset_task_for_target(target: Target) -> DatasetTask:
    return DatasetTask(target=target, data_files=DATA_FILES_BY_TARGET[target])


def origin_for_dataset_name(dataset_name: DatasetName) -> DatasetOrigin:
    return DATA_FILES_ALL[dataset_name].data_origin
