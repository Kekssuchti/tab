import json

import numpy as np
import pandas as pd

from src.config import config
from src.utils.logger import logger


def _remove_impossible_values(df, json_file_path):
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


def _filter_many_missing(df, threshold_row=0.5, threshold_col=0.5):
    logger.info("shape before missing filter", df.shape)

    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    row_null = df.isnull().sum(axis=1)
    df = df[row_null < (threshold_row * len(df.columns))]

    df = df.loc[:, df.isnull().mean() < 0.5]

    logger.info("shape after missing filter", df.shape)
    return df


def standart_preprocessing(df, threshold_row=0.5, threshold_col=0.5):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    _remove_impossible_values(df, config.dir_configs / "data_limits.json")
    _filter_many_missing(df, threshold_row, threshold_col)

    # still wip, we have way to many rows compared to expected extracted df
    return df
