from os.path import exists
from pathlib import Path
from typing import TypedDict

import pandas as pd
from sklearn.model_selection import train_test_split

from src.classes.data_cleaner import DataCleaner
from src.classes.data_registry import (
    dataset_task_for_target,
    origin_for_dataset_name,
)
from src.config import config
from src.schemas.dataset_schemas import (
    DatasetBundle,
    DatasetConfig,
    DatasetFileSummary,
    DatasetOrigin,
    DatasetSummary,
    XYDataset,
)
from src.utils.dataset_utils import (
    hash_file_sha256,
    remove_impossible_values,
    summarize_data_part,
)
from src.utils.logger import logger


class _SplitResult(TypedDict):
    """Train-test split for one source dataset."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


class Dataset:
    """Build aligned train and test data for one clinical target."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
    ) -> None:
        self.config = dataset_config
        self.seed = self.config.random_state
        self.data_cleaner = DataCleaner(self.config.data_cleaner)
        self._task = dataset_task_for_target(self.config.target)

    def get_dataset(self) -> DatasetBundle:
        """
        Get ALL dataset parts in DatasetBundle

        Train XYDataset are the specified combination of data splits given the DatasetConfig
        This can include only 1 of the 2 datasets, both datasets, fractions of any of those datasets or combinations of both
        """
        logger.info(
            f"Preparing dataset target={self.config.target} "
            f"files={','.join(file.file_name for file in self._task.data_files.values())}"
        )
        dfs = self._load_data()
        bundle = self._split_data(dfs)
        logger.info(
            "Dataset ready: "
            f"train_rows={len(bundle.train_data.y)} "
            f"test_mimic_rows={len(bundle.test_mimic.y)} "
            f"test_tudd_rows={len(bundle.test_tudd.y)} "
            f"features={bundle.train_data.X.shape[1]}"
        )
        return bundle

    def _load_data(self) -> dict[DatasetOrigin, pd.DataFrame]:
        """Load the filtered files required by the configured clinical target."""

        # ensure filtered csvs exist
        # force repreprocesisng each time to ensure consistency
        data_preprocessed = False

        for data_file in self._task.data_files.values():
            path = config.dir_data / "filtered" / data_file.file_name
            if not exists(path):
                logger.debug(f"Path: {path} doesnt exist")
                # if any required file is missing we assume something bad happened and reprocess all!
                data_preprocessed = False

        if not data_preprocessed or self.config.force_repreprocess:
            logger.info("Required filtered data missing or force_repreprocess=true")
            self.data_cleaner.preprocess_extracted_to_filtered()

        dfs = {}
        # load csvs
        for data_file in self._task.data_files.values():
            name = data_file.file_name
            # normalized mimic or tudd without readmission flag
            data_origin = data_file.data_origin

            path = Path(config.dir_data / "filtered" / name)

            df = pd.read_csv(path)
            df = self._runtime_preprocessing(df)

            dfs[data_origin] = df

        return dfs

    def _runtime_preprocessing(self, df):
        index_columns = [
            column for column in df.columns if column.startswith("Unnamed:")
        ]
        if index_columns:
            df = df.drop(columns=index_columns)
            logger.info(f"Dropped CSV index columns: {index_columns}")

        df, removed_counts = remove_impossible_values(
            df, self.config.data_cleaner.outlier_limits_path
        )
        removed_counts = {
            column: count for column, count in removed_counts.items() if count
        }
        if removed_counts:
            logger.info(f"Runtime removed unreasonable values: {removed_counts}")

        # here we set age to max = 90 (only really needed for tudd data)
        df["Age"] = df["Age"].clip(upper=91)
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

        self._align_split_feature_columns(splits_dict)

        X_train_parts: list[pd.DataFrame] = []
        y_train_parts: list[pd.Series] = []

        for training_data_split in self.config.train_on:
            # each datasplit we train on
            for dataset_origin, split in splits_dict.items():
                # compared against all keys we have (aka mimic and tudd)
                # skip if not in training_data_split.dataset
                if (
                    origin_for_dataset_name(training_data_split.dataset)
                    != dataset_origin
                ):
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

    def _align_split_feature_columns(
        self, splits_dict: dict[DatasetOrigin, _SplitResult]
    ) -> None:
        common_columns = set.intersection(
            *(set(split["X_train"].columns) for split in splits_dict.values())
        )
        ordered_columns = [
            column
            for column in next(iter(splits_dict.values()))["X_train"].columns
            if column in common_columns
        ]

        for split in splits_dict.values():
            split["X_train"] = split["X_train"].loc[:, ordered_columns]
            split["X_test"] = split["X_test"].loc[:, ordered_columns]

    def summarize(self, bundle: DatasetBundle) -> DatasetSummary:
        return DatasetSummary(
            target=self.config.target,
            train=summarize_data_part(bundle.train_data),
            test_mimic=summarize_data_part(bundle.test_mimic),
            test_tudd=summarize_data_part(bundle.test_tudd),
            data_files=tuple(self._summarize_data_files()),
        )

    def _summarize_data_files(self) -> list[DatasetFileSummary]:
        summaries = []
        for dataset_name, data_file in self._task.data_files.items():
            path = config.dir_data / "filtered" / data_file.file_name
            summaries.append(
                DatasetFileSummary(
                    dataset_name=dataset_name,
                    data_origin=data_file.data_origin,
                    file_name=data_file.file_name,
                    path=str(path),
                    sha256=hash_file_sha256(path),
                )
            )
        return summaries

    def _split_single_df(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        y = self._task.labels_from(df)
        X = self._task.features_from(df)

        stratify = None
        if (
            self.config.classification
            and y.nunique() > 1
            and y.value_counts().min() >= 2
        ):
            stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=(1 - self.config.train_size),
            random_state=self.seed,
            shuffle=True,
            stratify=stratify,
        )

        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        y_train = pd.Series(y_train)
        y_test = pd.Series(y_test)

        return X_train, X_test, y_train, y_test
