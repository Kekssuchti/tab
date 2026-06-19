from pathlib import Path

import pandas as pd

from src.config import config
from src.schemas.preprocessing_schemas import PreprocessorParams
from src.utils.datset_utils import standard_preprocessing


class Preprocessor:
    def __init__(self, params: PreprocessorParams) -> None:
        self.params = params

    def preprocess_extracted_to_filtered(self):
        """load and return both extracted dataframes"""
        # we skip the raw data step since I dont have access yet

        extracted_path = Path(config.dir_data / "extracted")
        filtered_path = Path(config.dir_data / "filtered")

        file_names = [
            "mimic4_mean_100_full.csv",
            "mimic4_readmission.csv",
            "tudd_mean_100_full.csv",
            "tudd_readmission.csv",
        ]

        for file_name in file_names:
            df = pd.read_csv(Path(extracted_path / file_name))
            readmission = True if "readmission" in file_name else False
            df_filtered = standard_preprocessing(
                df,
                readmission,
                self.params.missing_threshold_row,
                self.params.missing_threshold_col,
                self.params.outlier_limits_path,
            )

            df_filtered.to_csv(Path(filtered_path / file_name))
