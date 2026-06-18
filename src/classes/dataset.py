import math
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from src.schemas.dataset_schemas import DatasetParams


@dataclass(frozen=True)
class PreparedTestSet:
    X: pd.DataFrame
    y: pd.Series


@dataclass(frozen=True)
class PreparedDataset:
    X_train: pd.DataFrame
    y_train: pd.Series
    test_sets: dict[str, PreparedTestSet]


class Dataset:
    """
    Custom logic explanaition:
        we always have source and target df. they can be from the same data source of different.
    """

    def __init__(
        self,
        params: DatasetParams,
    ) -> None:
        self.params = params

    def create_dataset(self):
        # read in data
        # put data into df
        # preprocess data
        # combine data given params
        # save to self.data: PreparedDataset
        df_mimic, df_tudd = self._load_data()

        test_sets = {
            "mimic": PreparedTestSet(pd.DataFrame([1, 3, 2]), pd.Series([1, 6, 1]))
        }
        self.data: PreparedDataset = PreparedDataset(
            pd.DataFrame(123), pd.Series(123), test_sets=test_sets
        )

    def get_dataset(self):
        if not self.data:
            self.create_dataset()
        return self.data

    def _load_data(self):
        """load and return both extracted dataframes"""
        # we skip the raw data step since I dont have access yet

        pass
