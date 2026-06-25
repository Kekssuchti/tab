from pathlib import Path

import pandas as pd

from src.classes.data_registry import DATA_FILES_ALL
from src.config import config
from src.schemas.dataset_schemas import DataCleanerParams
from src.utils.dataset_utils import standard_preprocessing


class DataCleaner:
    def __init__(self, params: DataCleanerParams) -> None:
        self.params = params

    def preprocess_extracted_to_filtered(self) -> None:
        """load and return both extracted dataframes"""
        # we skip the raw data step since I dont have access yet

        extracted_path = Path(config.dir_data / "extracted")
        filtered_path = Path(config.dir_data / "filtered")

        for dataset in DATA_FILES_ALL.values():
            file_name = dataset.file_name
            df = pd.read_csv(Path(extracted_path / file_name))
            df_filtered = standard_preprocessing(
                df,
                dataset.is_readmission,
                self.params.missing_threshold_row,
                self.params.outlier_limits_path,
            )

            df_filtered.to_csv(Path(filtered_path / file_name), index=False)
