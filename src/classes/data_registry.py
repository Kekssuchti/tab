from dataclasses import dataclass
from typing import Literal

from src.schemas.dataset_schemas import DatasetName

DatasetOrigin = Literal["mimic", "tudd"]


@dataclass(frozen=True)
class DataFile:
    data_origin: DatasetOrigin
    file_name: str


DATA_FILES_NORMAL: dict[DatasetName, DataFile] = {
    "mimic": DataFile("mimic", "mimic4_mean_100_full.csv"),
    "tudd": DataFile("tudd", "tudd_mean_100_full.csv"),
}

DATA_FILES_READMISSION: dict[DatasetName, DataFile] = {
    "mimic_readmission": DataFile("mimic", "mimic4_readmission.csv"),
    "tudd_readmission": DataFile("tudd", "tudd_readmission.csv"),
}

DATA_FILES_ALL: dict[DatasetName, DataFile] = DATA_FILES_NORMAL | DATA_FILES_READMISSION
