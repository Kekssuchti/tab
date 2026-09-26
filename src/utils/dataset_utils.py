import hashlib
import json
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from src.config import config
from src.schemas.base_schemas import TaskType
from src.schemas.dataset_schemas import (
    ClassificationTargetSummary,
    CustomRetrieverConfig,
    DatasetOrigin,
    DatasetPartSummary,
    RegressionTargetSummary,
    RetrieverDistanceMetric,
    SplitResult,
    XYDataset,
)
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
            df[column] = df[column].where(df[column].between(lower_bound, upper_bound))
            after_count = df[column].notna().sum()

            removed_counts[column] = before_count - after_count

    return df, removed_counts


def remove_unused_columns(df, cols):
    # this assumes all cols we dont filter for are not worth keeping
    cols_before = df.columns.tolist()
    columns_to_drop = [column for column in cols_before if column not in cols]
    df = df.drop(columns=columns_to_drop)

    logger.info(f"dropped columns: {set(columns_to_drop)}")

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


UREA_TO_BUN = 2.1428  # conversion factor from total urea to blood urea nitrogen (BUN)


def _convert_units(df, dataset_origin):
    # only applied to tudd datasets to convert TOTAL UREA to BUN
    if dataset_origin == "tudd" and "Urea+100%mean" in df.columns:
        df["Urea+100%mean"] = df["Urea+100%mean"] / UREA_TO_BUN
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
    logger.debug(f"shape after LOS filter {df.shape}")
    return df


def _filter_childs(df, min_age):
    logger.debug(f"shape before min age filter {df.shape}")
    df = df[df["Age"] >= min_age]
    logger.debug(f"shape after min age filter {df.shape}")
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
        data_limit_config_path: path for json limits enforced

    This is task agnostic preprocessing from extracted to filtered datasets
    It removes unrealistic values based on "data_limits.json" (by making them null)
    It filters for minimum LOS 24h
    It removes rows (samples) with more missing values than threshold_row (default 50%)
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

    return df


def summarize_data_part(part: XYDataset, task_type: TaskType) -> DatasetPartSummary:
    if task_type == "classification":
        counts = part.y.value_counts(dropna=False).sort_index()
        target_summary = ClassificationTargetSummary(
            class_balance={str(label): int(count) for label, count in counts.items()}
        )
    else:
        numeric_values = pd.to_numeric(part.y, errors="coerce")
        finite_values = numeric_values[numeric_values.notna() & np.isfinite(numeric_values)]
        count = len(finite_values)

        def finite_or_zero(value: float) -> float:
            value = float(value)
            return value if np.isfinite(value) else 0.0

        target_summary = RegressionTargetSummary(
            count=count,
            mean=finite_or_zero(finite_values.mean()) if count else 0.0,
            std=finite_or_zero(finite_values.std()) if count > 1 else 0.0,
            min=finite_or_zero(finite_values.min()) if count else 0.0,
            max=finite_or_zero(finite_values.max()) if count else 0.0,
        )

    return DatasetPartSummary(row_count=len(part.y), target_summary=target_summary)


def hash_file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_retrieval_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    train_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if train_size > len(X_train):
        raise ValueError(f"Retriever train_size={train_size} exceeds the {len(X_train)} available training rows")
    if X_test.empty:
        raise ValueError("KNN retrieval requires at least one test row")

    try:
        train_values = X_train.to_numpy(dtype=float)
        test_values = X_test.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("KNN retrieval only supports numeric feature columns") from error

    preprocessing = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    return preprocessing.fit_transform(train_values), preprocessing.transform(test_values)


def _ranked_knn_positions(
    train_values: np.ndarray,
    test_values: np.ndarray,
    selection_size: int,
    distance_metric: RetrieverDistanceMetric,
) -> np.ndarray:
    """Return unique neighbors rank-by-rank across all test queries."""
    retriever = NearestNeighbors(metric=distance_metric, n_jobs=6).fit(train_values)
    # Start with 2x the no-overlap estimate because queries usually share neighbors.
    n_neighbors = min(
        selection_size,
        max(1, ceil(selection_size / len(test_values))) * 2,
    )

    while True:
        neighbor_positions = retriever.kneighbors(
            test_values,
            n_neighbors=n_neighbors,
            return_distance=False,
        )
        rank_ordered_positions = neighbor_positions.T.reshape(-1)
        unique_positions = pd.unique(rank_ordered_positions)
        if len(unique_positions) >= selection_size:
            return np.asarray(unique_positions[:selection_size], dtype=int)

        n_neighbors = min(selection_size, n_neighbors * 2)


def _knn_train_positions(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    train_size: int,
    distance_metric: RetrieverDistanceMetric,
) -> np.ndarray:
    """Retrieve a fixed budget of training rows nearest to the test cohort."""
    train_values, test_values = _prepare_retrieval_features(X_train, X_test, train_size)
    return _ranked_knn_positions(train_values, test_values, train_size, distance_metric)


def _square_root_cluster_selection(
    candidate_positions: np.ndarray,
    cluster_labels: np.ndarray,
    train_size: int,
) -> np.ndarray:
    """Select candidates by square-root cluster-size quotas in KNN rank order."""
    cluster_ids, cluster_sizes = np.unique(cluster_labels, return_counts=True)
    quotas = np.ones(len(cluster_ids), dtype=int)
    target_quotas = train_size * np.sqrt(cluster_sizes) / np.sqrt(cluster_sizes).sum()

    while quotas.sum() < train_size:
        eligible = quotas < cluster_sizes
        deficits = np.where(eligible, target_quotas - quotas, -np.inf)
        quotas[int(np.argmax(deficits))] += 1

    cluster_index = {cluster_id: index for index, cluster_id in enumerate(cluster_ids)}
    selected_counts = np.zeros(len(cluster_ids), dtype=int)
    selected_positions: list[int] = []
    for position, cluster_label in zip(candidate_positions, cluster_labels, strict=True):
        index = cluster_index[cluster_label]
        if selected_counts[index] >= quotas[index]:
            continue
        selected_positions.append(int(position))
        selected_counts[index] += 1

    return np.asarray(selected_positions, dtype=int)


def _knn_diverse_train_positions(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    train_size: int,
    distance_metric: RetrieverDistanceMetric,
    pool_multiplier: float,
    n_clusters: int,
    random_state: int,
) -> np.ndarray:
    """Retrieve candidates with the configured metric, then diversify with Euclidean clusters."""
    train_values, test_values = _prepare_retrieval_features(X_train, X_test, train_size)
    candidate_pool_size = min(len(train_values), ceil(train_size * pool_multiplier))
    candidate_positions = _ranked_knn_positions(
        train_values,
        test_values,
        candidate_pool_size,
        distance_metric,
    )

    effective_clusters = min(n_clusters, train_size, candidate_pool_size)
    cluster_labels = MiniBatchKMeans(
        n_clusters=effective_clusters,
        random_state=random_state,
        batch_size=min(1024, candidate_pool_size),
        n_init="auto",
    ).fit_predict(train_values[candidate_positions])

    return _square_root_cluster_selection(candidate_positions, cluster_labels, train_size)


def retriever_resample(
    retriever_config: CustomRetrieverConfig,
    data: dict[DatasetOrigin, SplitResult],
    train_data: XYDataset,
    train_sample_seed: int,
    test_sample_seed: int,
) -> tuple[XYDataset, XYDataset]:
    # create custom test set and sample indecies according to strategy
    # returns train_data and test_data as XYDatasets

    # loop through to create test set
    X_test_parts: list[pd.DataFrame] = []
    y_test_parts: list[pd.Series] = []

    for origin, split in data.items():
        for test_data_split in retriever_config.test_on:
            if origin != test_data_split.dataset:
                continue

            new_test_sample_idx = resample(
                split["X_test"].index,
                n_samples=test_data_split.fraction,
                random_state=test_sample_seed,
                replace=False,
                stratify=split["y_test"],
            )
            X_test_sampled = split["X_test"].loc[new_test_sample_idx].copy()
            y_test_sampled = split["y_test"].loc[new_test_sample_idx].copy()

            # Preserve stable source-row identity while preventing collisions when
            # the retriever cohort combines MIMIC and TUDD rows.
            origin_code = 0 if origin == "mimic" else 1
            retriever_index = pd.Index(2 * np.asarray(new_test_sample_idx, dtype=int) + origin_code)
            X_test_sampled.index = retriever_index
            y_test_sampled.index = retriever_index

            X_test_parts.append(X_test_sampled)
            y_test_parts.append(y_test_sampled)

    X_test = pd.concat(X_test_parts)
    y_test = pd.concat(y_test_parts)

    test_set = XYDataset(X=X_test, y=y_test)

    # Sample the fixed training budget according to the configured strategy.
    if retriever_config.selection_strategy == "random":
        train_indices = resample(
            train_data.X.index,
            replace=False,
            n_samples=retriever_config.train_size,
            random_state=train_sample_seed,
            stratify=train_data.y,
        )
        train_set = XYDataset(X=train_data.X.loc[train_indices], y=train_data.y.loc[train_indices])
    elif retriever_config.selection_strategy == "knn":
        train_positions = _knn_train_positions(
            train_data.X,
            test_set.X,
            retriever_config.train_size,
            retriever_config.distance_metric,
        )
        train_set = XYDataset(X=train_data.X.iloc[train_positions], y=train_data.y.iloc[train_positions])
    elif retriever_config.selection_strategy == "knn-diverse":
        train_positions = _knn_diverse_train_positions(
            train_data.X,
            test_set.X,
            retriever_config.train_size,
            retriever_config.distance_metric,
            retriever_config.diversity_pool_multiplier,
            retriever_config.diversity_clusters,
            train_sample_seed,
        )
        train_set = XYDataset(X=train_data.X.iloc[train_positions], y=train_data.y.iloc[train_positions])
    else:
        raise NotImplementedError(f"Retriever strategy {retriever_config.selection_strategy!r} is not implemented")

    return train_set, test_set
