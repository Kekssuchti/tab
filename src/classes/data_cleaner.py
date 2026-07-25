from pathlib import Path

import pandas as pd

from src.classes.data_registry import DATA_FILES_ALL
from src.config import config
from src.schemas.dataset_schemas import DataCleanerConfig
from src.utils.dataset_utils import standard_preprocessing
from src.utils.logger import logger


class DataCleaner:
    """Clean extracted clinical CSVs into filtered pipeline-ready CSVs."""

    def __init__(self, data_cleaner_config: DataCleanerConfig) -> None:
        self.config = data_cleaner_config

    def preprocess_extracted_to_filtered(self) -> None:
        """load and return both extracted dataframes"""
        # we skip the raw data step since I dont have access yet

        extracted_path = Path(config.dir_data / "extracted")
        filtered_path = Path(config.dir_data / "filtered")
        logger.info("Preprocessing extracted clinical CSVs into filtered CSVs")

        for dataset in DATA_FILES_ALL.values():
            file_name = dataset.file_name
            df = pd.read_csv(Path(extracted_path / file_name))

            df_filtered = standard_preprocessing(
                df,
                dataset.data_origin,
                dataset.is_readmission,
                self.config.missing_threshold_row,
                self.config.outlier_limits_path,
            )

            df_filtered.to_csv(Path(filtered_path / file_name), index=False)
            logger.info(f"Filtered {file_name}: rows {len(df)} -> {len(df_filtered)}")
