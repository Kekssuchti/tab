import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import config
from src.schemas.dataset_schemas import DatasetOrigin, DatasetPartSummary, XYDataset
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
            df[column] = df[column].apply(lambda x: x if lower_bound <= x <= upper_bound else None)
            after_count = df[column].notna().sum()

            removed_counts[column] = before_count - after_count

    return df, removed_counts


def remove_unused_columns(df, cols):
    # this assumes all cols we dont filter for are not worth keeping
    cols_before = df.columns.tolist()

    for column in cols_before:
        if column not in cols:
            df = df.drop(columns=[column], errors="ignore")

    logger.info(f"dropped columns: {set(cols_before) - set(df.columns.tolist())}")

    return df


def _get_cols_from_json(json_file_path, is_readmission):
    with open(json_file_path, "r") as file:
        cols_json = json.load(file)

    if is_readmission:
        cols = cols_json["readmission"]
    else:
        cols = cols_json["normal"]
    return cols


def _get_feature_cols(cols, is_readmission):
    cols_to_drop = ["mortality", "hours_to_readmit"]

    if not is_readmission:
        cols_to_drop.extend(["LOS"])

    feature_cols = [col for col in cols if col not in cols_to_drop]
    return feature_cols


def _convert_units(df, dataset_origin):
    # most of this is already done from the data I recieve
    # only urea is not correctly converted
    # this is also only applied to tudd datasets to convert TOTAL UREA to BUN
    if dataset_origin == "tudd":
        df["Urea+100%mean"] = df["Urea+100%mean"] / 2.1428
    return df


def _filter_many_missing(
    df: pd.DataFrame,
    readmission: bool,
    feature_cols: list[str],
    threshold_row=0.5,
):
    logger.debug(f"shape before missing filter {df.shape}")
    if readmission:
        # drop dead patients -> cannot readmit
        df = df.loc[df["mortality"] == 0].copy()
        df = df.drop(columns=["mortality"])

    missing_frac = df[feature_cols].isnull().mean(axis=1)

    df = df.loc[missing_frac <= threshold_row].copy()

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
    df = df[df["Age"] >= min_age]
    logger.debug(f"shape before min age filter {df.shape}")
    return df


def _clip_max_age(df, max_age):
    logger.debug(f"shape before max age filter {df.shape}")
    df["Age"] = df["Age"].clip(upper=max_age)
    logger.debug(f"shape after max age filter {df.shape}")
    return df


def _clean_dtypes(df: pd.DataFrame):
    mapping = {"M": 0, "F": 1}
    df["Sex"] = df["Sex"].map(mapping, na_action="ignore")
    return df


def standard_preprocessing(
    df,
    df_origin: DatasetOrigin,
    readmission: bool,
    threshold_row: float = 0.5,
    data_limit_config_path: Path = config.dir_configs / "data_limits.json",
    data_cols_config_path: Path = config.dir_configs / "data_cols.json",
    min_los_filter=24,
    max_los_filter=24 * 100,
    min_age_filter=18,
    max_age_filter=91,
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
    It clips max age to max_age_filter (default 91) - confirms with MIMIC Age tracking
    """
    logger.debug(f"starting df len: {df.shape}")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df = _convert_units(df, df_origin)
    df, rm_count = remove_impossible_values(df, data_limit_config_path)
    logger.debug(f"removed {rm_count} unreasonable values:")
    logger.debug(f"df len after remove_impossible_values: {len(df)}")

    all_cols = _get_cols_from_json(data_cols_config_path, readmission)
    feature_cols = _get_feature_cols(all_cols, readmission)

    df = remove_unused_columns(df, all_cols)
    logger.debug(f"df len after remove_unused_columns: {df.shape}")

    df = _filter_reasonable_los(df, min_los_filter, max_los_filter)
    logger.debug(f"df len after _filter_reasonable_los: {len(df)}")

    df = _filter_childs(df, min_age=min_age_filter)
    logger.debug(f"df len after _filter_childs: {len(df)}")

    df = _clip_max_age(df, max_age=max_age_filter)

    df = _filter_many_missing(df, readmission, feature_cols, threshold_row)
    logger.debug(f"df len after _filter_many_missing: {len(df)}")

    df = _clean_dtypes(df)
    logger.debug(f"df len after _clean_dtypes: {len(df)}")

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
