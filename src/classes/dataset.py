from dataclasses import dataclass
from os.path import exists
from pathlib import Path
from typing import TypedDict

import pandas as pd
from sklearn.model_selection import train_test_split

from src.classes.data_cleaner import DataCleaner
from src.classes.data_registry import (
    DATA_FILES_ALL,
    DATA_FILES_NORMAL,
    DATA_FILES_READMISSION,
    DatasetOrigin,
)
from src.config import config
from src.schemas.dataset_schemas import (
    DatasetBundle,
    DatasetParams,
    DataSplitParams,
    XYDataset,
)
from src.utils.logger import logger


class _SplitResult(TypedDict):
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


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
        self.seed = self.params.random_state
        self.data_cleaner = DataCleaner(params.data_cleaner)
        self._target = self.params.target
        self._data_files = (
            DATA_FILES_READMISSION
            if self._target == "hours_to_readmit"
            else DATA_FILES_NORMAL
        )

    def get_dataset(self) -> DatasetBundle:
        """
        Get ALL dataset parts in DatasetBundle

        Train XYDataset are the specified combination of Datasplits given the DatasetParams
        This can include only 1 of the 2 datasets, both datasets, fractions of any of those datasets or combinations of both
        """
        # combine data given params
        # save to self.data: PreparedDataset
        dfs = self._load_data(self.params.train_on)
        # dfs is always len 2 and has both mimic and tudd datasets
        # either "normal" or readmission

        return self._split_data(dfs)

    def _load_data(
        self,
        df_splits: tuple[DataSplitParams, ...],
    ) -> dict[DatasetOrigin, pd.DataFrame]:
        """
        Args:
            df_names:   List of strings of keys for what df you want returned
                        Available Keys are: 'mimic', 'mimic_readmission', 'tudd', 'tudd_readmission'

        Returns:
            list of pd.DataFrame in the order of df_names
        """

        # ensure filtered csvs exist
        data_preprocessed = True

        for data_file in DATA_FILES_ALL.values():
            path = config.dir_data / "filtered" / data_file.file_name
            if not exists(path):
                logger.debug(f"Path: {path} doesnt exist")
                # if any file is missing we assume something bad happened and reprocess all!
                data_preprocessed = False

        if not data_preprocessed or self.params.force_repreprocess:
            self.data_cleaner.preprocess_extracted_to_filtered()

        dfs = {}
        # load csvs
        for data_file in self._data_files.values():
            name = data_file.file_name
            # normalized mimic or tudd without readmission flag
            data_origin = data_file.data_origin

            path = Path(config.dir_data / "filtered" / name)

            df = pd.read_csv(path)
            df = self._runtime_preprocessing(df)

            dfs[data_origin] = df

        return dfs

    def _runtime_preprocessing(self, df):
        # here we set age to max = 90 (only really needed for tudd data)
        df["Age"] = df["Age"].clip(upper=90)
        return df

    def _split_data(self, dfs: dict[DatasetOrigin, pd.DataFrame]) -> DatasetBundle:
        """
        This function splits our data to the parts we want and need.
        Returns:
            DataBundle
        """

        splits_dict: dict[DatasetOrigin, _SplitResult] = {}

        for key, df in dfs.items():
            X_train, X_test, y_train, y_test = self._split_single_df(df)

            splits_dict[key] = {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            }

        X_train_parts: list[pd.DataFrame] = []
        y_train_parts: list[pd.Series] = []

        for training_data_split in self.params.train_on:
            # each datasplit we train on
            for dataset_origin, split in splits_dict.items():
                # compared against all keys we have (aka mimic and tudd)
                # skip if not in training_data_split.dataset
                if str(dataset_origin) not in str(training_data_split.dataset):
                    continue

                # if we do train on the split -> apply fraction of training data
                # and add it to combined X_train
                if isinstance(training_data_split.fraction, float):
                    n = int(training_data_split.fraction * len(split["X_train"]))
                else:
                    n = training_data_split.fraction

                sampled_indices = (
                    split["X_train"].sample(n=n, random_state=self.seed).index
                )

                X_train_sampled = split["X_train"].loc[sampled_indices]
                y_train_sampled = split["y_train"].loc[sampled_indices]

                X_train_parts.append(X_train_sampled)
                y_train_parts.append(y_train_sampled)

        X_train_combined = pd.concat(X_train_parts, axis=0, ignore_index=True)
        y_train_combined = pd.concat(y_train_parts, axis=0, ignore_index=True)

        shuffled_indices = X_train_combined.sample(
            frac=1, random_state=self.seed * 2
        ).index

        train_data = XYDataset(
            X=X_train_combined.loc[shuffled_indices],
            y=y_train_combined.loc[shuffled_indices],
        )

        test_mimic = XYDataset(
            X=splits_dict["mimic"]["X_test"], y=splits_dict["mimic"]["y_test"]
        )

        test_tudd = XYDataset(
            X=splits_dict["tudd"]["X_test"], y=splits_dict["tudd"]["y_test"]
        )

        data_bundle = DatasetBundle(
            train_data=train_data,
            test_mimic=test_mimic,
            test_tudd=test_tudd,
        )

        return data_bundle

    def _split_single_df(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        y = df[self._target]
        if self._target == "hours_to_readmit":
            y = y.notna().astype(int)

        cols_to_drop = [
            "mortality",
            "LOS",
            "hours_to_readmit",
        ]

        X = df.drop(columns=cols_to_drop, errors="ignore")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=(1 - self.params.train_size),
            random_state=self.seed,
            shuffle=True,
        )

        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        y_train = pd.Series(y_train)
        y_test = pd.Series(y_test)

        return X_train, X_test, y_train, y_test
