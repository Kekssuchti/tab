import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import config
from src.schemas.dataset_schemas import DatasetPartSummary, XYDataset
from src.utils.logger import logger


def remove_impossible_values(df, json_file_path):
    """
    Remove entries from a DataFrame based on limits specified in a JSON file.
    Mostly measurement errors or very unrealistic values

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    json_file_path (str): Path to the JSON file containing limits.

    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    dict: Dictionary with the count of removed values for each column.
    """
    # Read the limits from the JSON file
    with open(json_file_path, "r") as file:
        limits = json.load(file)

    removed_counts = {}

    for column, bounds in limits.items():
        if column in df.columns:
            lower_bound = bounds["lower_bound"]
            upper_bound = bounds["upper_bound"]

            before_count = df[column].notna().sum()
            df[column] = df[column].apply(
                lambda x: x if lower_bound <= x <= upper_bound else None
            )
            after_count = df[column].notna().sum()

            removed_counts[column] = before_count - after_count

    return df, removed_counts


def _filter_many_missing(
    df: pd.DataFrame,
    readmission: bool,
    threshold_row=0.5,
):
    logger.debug(f"shape before missing filter {df.shape}")
    # -3 is offset for cols we dont use
    if readmission:
        # mortality, LOS, Bmi+100%mean, hours_to_readmit
        feature_offset = 4

        # subject_id, hadm_id, stay_id dropped
        df = df.drop(columns=["subject_id", "hadm_id", "stay_id"])

        # drop dead patients -> cannot readmit
        df = df.loc[df["mortality"] != 1]
    else:
        # mortality, LOS, Bmi+100%mean
        feature_offset = 3

    row_null: pd.Series = df.isnull().sum(axis=1)
    df = df.loc[row_null < int(threshold_row * (len(df.columns) - feature_offset))]

    # REMOVED FOR NOW:
    # Problem is that both datasets have different cols where they miss a lot of values

    # drop all cols with too many missing but keep readmission for obvious reasons
    # df = df.loc[:, (df.isnull().mean() < 0.5) | df.columns.isin(["hours_to_readmit"])]

    logger.debug(f"shape after missing filter {df.shape}")
    return df


def _filter_reasonable_los(df, min_h, max_h):
    logger.debug(f"shape before LOS filter {df.shape}")
    df = df[df["LOS"] > min_h]
    df = df[df["LOS"] < max_h]
    logger.debug(f"shape before LOS filter {df.shape}")
    return df


def _filter_childs(df, min_age):
    logger.debug(f"shape before min age filter {df.shape}")
    df = df[df["Age"] > min_age]
    logger.debug(f"shape before min age filter {df.shape}")
    return df


def _clean_dtypes(df: pd.DataFrame):
    df["Sex"] = (df["Sex"] == "F").astype(int)
    return df


def standard_preprocessing(
    df,
    readmission: bool,
    threshold_row: float = 0.5,
    data_limit_config_path: Path = config.dir_configs / "data_limits.json",
    min_los_filter=24,
    max_los_filter=24 * 14000,
    min_age_filter=18,
):
    """
    Args:
        df: data
        readmission: bool if this is readmission dataset
        threshold_row: float threshold when missing data removes the row (sample)
        threshold_col: float threshold when missing data removes the col (feature)
        data_limit_config_path: path for json limits enforced

    This is task agnostic preprocessing from extracted to filtered datasets
    It removes unrealistic values based on "data_limits.json" (by making them null)
    It filters for minimum LOS 24h
    It removes rows (samples) with more missing values than threshold_row (default 50%)
    It removes cols (features) with more missing values than threshold_col (default 50%)
    """

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df, rm_count = remove_impossible_values(df, data_limit_config_path)
    logger.info(f"removed {rm_count} unreasonable values:")
    df = _filter_reasonable_los(df, min_los_filter, max_los_filter)
    df = _filter_childs(df, min_age=min_age_filter)
    df = _filter_many_missing(df, readmission, threshold_row)
    df = _clean_dtypes(df)

    # still wip, we have minimally less rows than expected
    return df


def summarize_data_part(part: XYDataset) -> DatasetPartSummary:
    counts = part.y.value_counts(dropna=False).sort_index()
    class_balance = {str(label): int(count) for label, count in counts.items()}
    return DatasetPartSummary(
        row_count=len(part.y),
        class_balance=class_balance,
    )


def hash_file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
