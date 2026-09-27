"""Scientific target/split policies and real cleaning, not dataset assembly snapshots."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.classes import data_cleaner as cleaner_module
from src.classes import dataset as dataset_module
from src.classes.data_cleaner import DataCleaner
from src.classes.data_registry import dataset_task_for_target
from src.classes.dataset import Dataset
from src.schemas.dataset_schemas import DataCleanerConfig, DatasetConfig, DataSplitConfig


@pytest.mark.parametrize(
    ("target", "kind", "expected"),
    [
        ("mortality", "normal", [0, 1, 0, 1, 0, 1]),
        ("LOS7", "normal", [0, 0, 1, 0, 0, 1]),
        ("hours_to_readmit", "readmission", [0, 1, 1, 1, 1, 1]),
        ("hours_to_readmit_72", "readmission", [0, 1, 1, 1, 0, 0]),
    ],
)
def test_classification_boundaries_missing_readmission_and_target_exclusion(target, kind, expected):
    """E2Es cannot establish exact clinical boundaries or every target's leakage policy."""
    rows = pd.DataFrame(
        {
            "Age": [40, 50, 60, 70, 80, 90],
            "mortality": [0, 1, 0, 1, 0, 1],
            "LOS": [167.99, 168, 168.01, 12, 48, 200],
            "hours_to_readmit": [np.nan, 0, 71.99, 72, 72.01, 200],
            # Stale derived columns must neither supply labels nor leak into X.
            "LOS3": [99] * 6,
            "LOS7": [99] * 6,
            "hours_to_readmit_72": [99] * 6,
        },
        index=[20, 2, 18, 7, 13, 6],
    )
    original = rows.copy(deep=True)
    task = dataset_task_for_target(target)

    labels = task.labels_from(rows)
    features = task.features_from(rows)

    assert labels.tolist() == expected
    assert labels.index.equals(rows.index)
    assert task.task_type == "classification"
    assert task.dataset_kind == kind
    expected_files = (
        {"mimic4_readmission.csv", "tudd_readmission.csv"}
        if kind == "readmission"
        else {"mimic4_mean_100_full.csv", "tudd_mean_100_full.csv"}
    )
    assert {file.file_name for file in task.data_files.values()} == expected_files
    # Completed-stay LOS is deliberately available for readmission prediction.
    expected_features = ["Age", "LOS"] if kind == "readmission" else ["Age"]
    pd.testing.assert_frame_equal(features, original[expected_features])
    pd.testing.assert_frame_equal(rows, original)


def test_holdouts_ignore_training_seed_and_budget_with_source_aware_disjointness(tmp_path, monkeypatch):
    """Repetitions must redraw training rows, not the evaluation population or labels."""
    filtered = tmp_path / "filtered"
    filtered.mkdir()
    for source, file_name in (("mimic", "mimic4_mean_100_full.csv"), ("tudd", "tudd_mean_100_full.csv")):
        record_ids = np.arange(80)  # The two sources intentionally reuse local IDs.
        pd.DataFrame(
            {
                "source": source,
                "record_id": record_ids,
                "Age": 40 + record_ids % 30,
                "mortality": (record_ids % 5 == (source == "tudd")).astype(int),
            }
        ).to_csv(filtered / file_name, index=False)
    limits = tmp_path / "limits.json"
    limits.write_text("{}")
    monkeypatch.setattr(dataset_module, "config", SimpleNamespace(dir_data=tmp_path))

    def build(sample_seed=7, budget=10, split_seed=19):
        return Dataset(
            DatasetConfig(
                target="mortality",
                random_state=split_seed,
                train_size=0.75,
                train_on=tuple(DataSplitConfig(dataset=source, fraction=budget) for source in ("mimic", "tudd")),
                data_cleaner=DataCleanerConfig(outlier_limits_path=limits),
            ),
            sample_seed=sample_seed,
        ).get_dataset()

    first = build()
    repeated = build()
    resampled = build(sample_seed=8)
    enlarged = build(budget=0.5)
    resplit = build(split_seed=20)

    def identities(part):
        return set(zip(part.X["source"], part.X["record_id"], strict=True))

    for bundle in (first, repeated, resampled, enlarged, resplit):
        for part in (bundle.train_data, bundle.test_mimic, bundle.test_tudd):
            assert part.X.index.equals(part.y.index)
            assert len(identities(part)) == len(part.y)
            expected = (part.X["record_id"] % 5 == (part.X["source"] == "tudd")).astype(int)
            np.testing.assert_array_equal(part.y, expected)
        assert identities(bundle.train_data).isdisjoint(identities(bundle.test_mimic))
        assert identities(bundle.train_data).isdisjoint(identities(bundle.test_tudd))
        assert identities(bundle.test_mimic).isdisjoint(identities(bundle.test_tudd))
        # Stratification protects the deliberately imbalanced 80:20 population.
        for source in ("mimic", "tudd"):
            source_labels = bundle.train_data.y[bundle.train_data.X["source"] == source]
            assert source_labels.mean() == pytest.approx(0.2)

    pd.testing.assert_frame_equal(first.train_data.X, repeated.train_data.X)
    pd.testing.assert_series_equal(first.train_data.y, repeated.train_data.y)
    assert identities(first.train_data) != identities(resampled.train_data)
    assert len(first.train_data.y) == 20
    assert len(enlarged.train_data.y) == 60
    for name in ("test_mimic", "test_tudd"):
        reference = getattr(first, name)
        assert len(reference.y) == 20
        for bundle in (repeated, resampled, enlarged):
            pd.testing.assert_frame_equal(reference.X, getattr(bundle, name).X)
            pd.testing.assert_series_equal(reference.y, getattr(bundle, name).y)
        assert identities(reference) != identities(getattr(resplit, name))


@pytest.mark.parametrize("kind", ["normal", "readmission"])
def test_real_cleaning_filters_and_converts_only_requested_dataset_kind(tmp_path, monkeypatch, kind):
    """Wrong units, survivor rules or kind routing silently change the study population."""
    extracted, filtered = tmp_path / "extracted", tmp_path / "filtered"
    extracted.mkdir()
    filtered.mkdir()
    columns = ["record_id", "Age", "Sex", "LOS", "mortality", "lab", "Urea+100%mean"]
    column_policy = tmp_path / "columns.json"
    column_policy.write_text(json.dumps({"normal": columns, "readmission": [*columns, "hours_to_readmit"]}))
    limits = tmp_path / "limits.json"
    limits.write_text(json.dumps({"lab": {"lower_bound": 0, "upper_bound": 100}}))
    rows = pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4, 5, 6, 7],
            "Age": [95, 17, 55, 65, 65, 60, 18],
            "Sex": ["F", "M", "F", "M", "M", None, "F"],
            "LOS": [48, 48, 24, 2400, 72, 60, 25],
            "mortality": [0, 0, 0, 0, 1, 0, 0],
            "hours_to_readmit": [72, 24, 12, 48, 36, 24, np.nan],
            "lab": [999, 20, 30, 40, 50, np.nan, 70],
            "Urea+100%mean": [21.428, 21.428, 21.428, 21.428, 21.428, np.inf, 42.856],
            "subject_id": range(7),
        }
    )
    selected = (
        ("mimic4_mean_100_full.csv", "tudd_mean_100_full.csv")
        if kind == "normal"
        else ("mimic4_readmission.csv", "tudd_readmission.csv")
    )
    unselected = (
        ("mimic4_readmission.csv", "tudd_readmission.csv")
        if kind == "normal"
        else ("mimic4_mean_100_full.csv", "tudd_mean_100_full.csv")
    )
    for file_name in selected:
        rows.to_csv(extracted / file_name, index=False)
    # No unselected extracted files exist; existing other-kind outputs must not be touched.
    for file_name in unselected:
        (filtered / file_name).write_text("unrelated dataset sentinel\n")
    monkeypatch.setattr(cleaner_module, "config", SimpleNamespace(dir_data=tmp_path))
    real_preprocessing = cleaner_module.standard_preprocessing

    def with_toy_column_policy(*args, **kwargs):
        return real_preprocessing(*args, **kwargs, data_cols_config_path=column_policy)

    monkeypatch.setattr(cleaner_module, "standard_preprocessing", with_toy_column_policy)
    cleaner = DataCleaner(DataCleanerConfig(outlier_limits_path=limits, missing_threshold_row=0.4))
    cleaner.preprocess_extracted_to_filtered(kind)

    for origin, file_name in zip(("mimic", "tudd"), selected, strict=True):
        result = pd.read_csv(filtered / file_name).set_index("record_id")
        assert set(result.index) == ({1, 5, 7} if kind == "normal" else {1, 7})
        assert result.loc[1, "Age"] == 91
        assert result.loc[7, "Age"] == 18
        assert result.loc[1, "Sex"] == result.loc[7, "Sex"] == 1
        assert pd.isna(result.loc[1, "lab"])
        assert result.loc[7, "lab"] == 70
        assert result.loc[1, "Urea+100%mean"] == pytest.approx(10 if origin == "tudd" else 21.428)
        assert result.loc[7, "Urea+100%mean"] == pytest.approx(20 if origin == "tudd" else 42.856)
        assert "subject_id" not in result
        assert not any(column.startswith("Unnamed:") for column in result)
        if kind == "readmission":
            assert "mortality" not in result
            assert result.loc[1, "hours_to_readmit"] == 72
            assert pd.isna(result.loc[7, "hours_to_readmit"])
        else:
            assert result.loc[5, "mortality"] == 1
            assert result.loc[5, "Sex"] == 0
    for file_name in unselected:
        assert (filtered / file_name).read_text() == "unrelated dataset sentinel\n"
