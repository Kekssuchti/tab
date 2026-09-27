"""Selection contracts across seeds and geometries; no private cluster-label assertions."""

import numpy as np
import pandas as pd
import pytest

from src.schemas.dataset_schemas import CustomRetrieverConfig, DataSplitConfig, XYDataset
from src.utils.dataset_utils import retriever_resample


@pytest.mark.parametrize("strategy", ["random", "knn", "knn-diverse"])
def test_retrieval_has_unique_budget_aligned_source_identity_and_independent_seeds(strategy):
    """A changed training draw must never redraw the query cohort or conflate source IDs."""
    train_X = pd.DataFrame({"source": np.repeat([0, 1], 40), "local_id": np.tile(np.arange(40), 2)})
    train = XYDataset(X=train_X, y=(train_X["local_id"] + train_X["source"]) % 2)
    splits = {}
    for source, name in enumerate(("mimic", "tudd")):
        # Matching pandas indices in different sources must become distinct query identities.
        query_X = pd.DataFrame({"source": source, "local_id": np.arange(100, 130)}, index=np.arange(100, 130))
        query_y = (query_X["local_id"] + query_X["source"]) % 2
        splits[name] = {"X_train": train.X, "y_train": train.y, "X_test": query_X, "y_test": query_y}
    config = CustomRetrieverConfig(
        selection_strategy=strategy,
        train_size=24,
        test_on=tuple(DataSplitConfig(dataset=name, fraction=8) for name in splits),
        diversity_clusters=4,
    )

    def retrieve(train_seed=7, test_seed=11):
        return retriever_resample(config, splits, train, train_sample_seed=train_seed, test_sample_seed=test_seed)

    first = retrieve()
    repeated = retrieve()
    changed_train = retrieve(train_seed=8)
    changed_query = retrieve(test_seed=12)

    def identities(part):
        return set(map(tuple, part.X[["source", "local_id"]].to_numpy()))

    for selected, query in (first, repeated, changed_train, changed_query):
        assert len(selected.y) == len(identities(selected)) == 24
        assert len(query.y) == len(identities(query)) == 16
        assert selected.X.index.is_unique
        assert query.X.index.is_unique
        assert identities(selected) <= identities(train)
        assert identities(selected).isdisjoint(identities(query))
        assert query.X["source"].value_counts().to_dict() == {0: 8, 1: 8}
        for part in (selected, query):
            assert part.X.index.equals(part.y.index)
            np.testing.assert_array_equal(part.y, (part.X["local_id"] + part.X["source"]) % 2)
        for source, name in enumerate(("mimic", "tudd")):
            query_ids = set(query.X.loc[query.X["source"] == source, "local_id"])
            assert query_ids <= set(splits[name]["X_test"]["local_id"])

    for original, repeat in zip(first, repeated, strict=True):
        pd.testing.assert_frame_equal(original.X, repeat.X)
        pd.testing.assert_series_equal(original.y, repeat.y)
    pd.testing.assert_frame_equal(first[1].X, changed_train[1].X)
    pd.testing.assert_series_equal(first[1].y, changed_train[1].y)
    assert identities(first[1]) != identities(changed_query[1])
    # Stable query identity is meaningful across draws, not just unique within each draw.
    original_query, redrawn_query = first[1], changed_query[1]
    shared = original_query.X.index.intersection(redrawn_query.X.index)
    assert len(shared) > 0
    pd.testing.assert_frame_equal(original_query.X.loc[shared], redrawn_query.X.loc[shared])
    pd.testing.assert_series_equal(original_query.y.loc[shared], redrawn_query.y.loc[shared])
    if strategy == "random":
        assert identities(first[0]) != identities(changed_train[0])
        pd.testing.assert_frame_equal(first[0].X, changed_query[0].X)
        pd.testing.assert_series_equal(first[0].y, changed_query[0].y)
    elif strategy == "knn":
        # Deterministic neighbors on a fixed pool should not invent training-seed randomness.
        pd.testing.assert_frame_equal(first[0].X, changed_train[0].X)


def _single_source(train_X, query_X):
    train = XYDataset(X=train_X, y=pd.Series(np.arange(len(train_X)) % 2, index=train_X.index))
    query_y = pd.Series(np.arange(len(query_X)) % 2, index=query_X.index)
    splits = {"mimic": {"X_train": train.X, "y_train": train.y, "X_test": query_X, "y_test": query_y}}
    return train, splits


def test_knn_deduplicates_overlapping_neighborhoods_without_losing_budget():
    """Multiple almost-identical queries must not spend the budget on duplicate rows."""
    train_X = pd.DataFrame({"position": [0.0, 1.0, 2.0, 8.0, 9.0, 10.0], "lab": [0.0, np.nan, 0.0, 0.0, 0.0, 0.0]})
    query_X = pd.DataFrame(
        {"position": [0.1, 0.2, 9.8, 9.9], "lab": [0.0, np.nan, 0.0, 0.0]}, index=[100, 101, 102, 103]
    )
    train, splits = _single_source(train_X, query_X)
    config = CustomRetrieverConfig(
        selection_strategy="knn", train_size=4, test_on=(DataSplitConfig(dataset="mimic", fraction=4),)
    )
    selected, query = retriever_resample(config, splits, train, train_sample_seed=7, test_sample_seed=11)
    assert set(selected.X["position"]) == {0.0, 1.0, 9.0, 10.0}
    assert len(selected.y) == 4
    assert selected.X.index.is_unique
    pd.testing.assert_frame_equal(selected.X, train.X.loc[selected.X.index])
    pd.testing.assert_series_equal(selected.y, train.y.loc[selected.X.index])
    assert set(selected.X["position"]).isdisjoint(query.X["position"])


def test_knn_distance_metric_changes_geometry_not_just_configuration():
    """Equal axis variances make the Euclidean/Manhattan nearest-neighbor oracle explicit."""
    train_X = pd.DataFrame({"a": [2.0, 1.1, -2.0, 0.0, 0.0, -1.1], "b": [0.0, 1.1, 0.0, 2.0, -2.0, -1.1]})
    query_X = pd.DataFrame({"a": [0.0, 0.0], "b": [0.0, 0.0]}, index=[100, 101])
    train, splits = _single_source(train_X, query_X)
    selections = {}
    for metric in ("euclidean", "manhattan"):
        config = CustomRetrieverConfig(
            selection_strategy="knn",
            distance_metric=metric,
            train_size=1,
            test_on=(DataSplitConfig(dataset="mimic", fraction=2),),
        )
        selected, _ = retriever_resample(config, splits, train, train_sample_seed=7, test_sample_seed=11)
        selections[metric] = sorted(abs(selected.X.iloc[0].to_numpy()))
    assert selections["euclidean"] == pytest.approx([1.1, 1.1])
    assert selections["manhattan"] == pytest.approx([0.0, 2.0])


def test_diverse_retrieval_reproducibly_covers_separated_regions_within_neighbor_pool():
    """Diversity must change selected geometry, not merely invoke a clustering helper."""
    train_X = pd.DataFrame({"position": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 3.0, 3.1, 3.2, 10.0]})
    query_X = pd.DataFrame({"position": [0.05, 0.15]}, index=[100, 101])
    train, splits = _single_source(train_X, query_X)

    def retrieve(strategy):
        config = CustomRetrieverConfig(
            selection_strategy=strategy,
            train_size=4,
            test_on=(DataSplitConfig(dataset="mimic", fraction=2),),
            diversity_pool_multiplier=2.0,
            diversity_clusters=2,
        )
        return retriever_resample(config, splits, train, train_sample_seed=7, test_sample_seed=11)[0]

    nearest, diverse, repeated = retrieve("knn"), retrieve("knn-diverse"), retrieve("knn-diverse")
    assert nearest.X["position"].max() < 1
    assert (diverse.X["position"] < 1).any()
    assert (diverse.X["position"] >= 3).any()
    # The eight nearest candidates include 3.0 and 3.1, not the remote points.
    assert set(diverse.X["position"]) <= {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 3.0, 3.1}
    assert len(diverse.y) == 4
    assert diverse.X.index.is_unique
    pd.testing.assert_frame_equal(diverse.X, repeated.X)
    pd.testing.assert_series_equal(diverse.y, train.y.loc[diverse.X.index])
    pd.testing.assert_series_equal(diverse.y, repeated.y)
