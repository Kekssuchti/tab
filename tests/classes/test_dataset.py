from types import SimpleNamespace

import pandas as pd

from src.classes import dataset as dataset_module
from src.classes.data_registry import TARGET_LIKE_COLUMNS
from src.classes.dataset import Dataset
from src.schemas.dataset_schemas import DatasetParams, DataSplitParams


def _make_rows(source: str, start_id: int, n_rows: int) -> pd.DataFrame:
    row_numbers = list(range(n_rows))
    record_ids = list(range(start_id, start_id + n_rows))
    return pd.DataFrame(
        {
            "record_id": record_ids,
            "source": [source] * n_rows,
            "Age": [70 + i * 4 for i in row_numbers],
            "shared_feature": [float(i * 10) for i in row_numbers],
            "mortality": [i % 2 for i in row_numbers],
            "LOS": [48.0 + i for i in row_numbers],
            "LOS3": [int(i % 2 == 0) for i in row_numbers],
            "LOS7": [int(i % 2 == 1) for i in row_numbers],
            "hours_to_readmit": [12.0 if i % 3 else None for i in row_numbers],
        }
    )


def _write_filtered_files(tmp_path, files: dict[str, pd.DataFrame]) -> None:
    filtered_path = tmp_path / "filtered"
    filtered_path.mkdir()
    for file_name, df in files.items():
        df.to_csv(filtered_path / file_name, index=False)


def _dataset_params(
    target: str,
    train_on: tuple[DataSplitParams, ...],
    train_size: float = 0.5,
    random_state: int = 7,
) -> DatasetParams:
    return DatasetParams(
        target=target,
        train_on=train_on,
        train_size=train_size,
        random_state=random_state,
    )


def _labels_by_record_id(df: pd.DataFrame, target: str) -> dict[int, int | float]:
    if target == "hours_to_readmit":
        return df.set_index("record_id")[target].notna().astype(int).to_dict()
    return df.set_index("record_id")[target].to_dict()


def _assert_labels_match_rows(
    X: pd.DataFrame, y: pd.Series, labels_by_id: dict[int, int | float]
) -> None:
    actual = dict(zip(X["record_id"], y, strict=True))
    expected = {record_id: labels_by_id[record_id] for record_id in X["record_id"]}
    assert actual == expected


def _assert_common_feature_shape(bundle) -> None:
    expected_columns = {"record_id", "source", "Age", "shared_feature"}
    assert set(bundle.train_data.X.columns) == expected_columns
    assert set(bundle.test_mimic.X.columns) == expected_columns
    assert set(bundle.test_tudd.X.columns) == expected_columns
    assert set(TARGET_LIKE_COLUMNS).isdisjoint(bundle.train_data.X.columns)
    assert set(TARGET_LIKE_COLUMNS).isdisjoint(bundle.test_mimic.X.columns)
    assert set(TARGET_LIKE_COLUMNS).isdisjoint(bundle.test_tudd.X.columns)


def test_mortality_dataset_uses_only_normal_task_files_and_assembles_splits(
    tmp_path, monkeypatch
):
    mimic = _make_rows("mimic", 100, 12).assign(mimic_only_feature=1.0)
    tudd = _make_rows("tudd", 200, 12).assign(tudd_only_feature=2.0)
    _write_filtered_files(
        tmp_path,
        {
            "mimic4_mean_100_full.csv": mimic,
            "tudd_mean_100_full.csv": tudd,
        },
    )
    monkeypatch.setattr(dataset_module, "config", SimpleNamespace(dir_data=tmp_path))

    dataset = Dataset(
        _dataset_params(
            target="mortality",
            train_on=(
                DataSplitParams(dataset="mimic", fraction=1.0),
                DataSplitParams(dataset="tudd", fraction=2),
            ),
        )
    )

    bundle = dataset.get_dataset()

    assert len(bundle.train_data.X) == 8
    assert len(bundle.test_mimic.X) == 6
    assert len(bundle.test_tudd.X) == 6
    assert (bundle.train_data.X["source"] == "mimic").sum() == 6
    assert (bundle.train_data.X["source"] == "tudd").sum() == 2
    assert bundle.train_data.X["Age"].max() <= 90
    assert bundle.test_tudd.X["Age"].max() <= 90
    _assert_common_feature_shape(bundle)

    labels = _labels_by_record_id(pd.concat([mimic, tudd]), "mortality")
    _assert_labels_match_rows(bundle.train_data.X, bundle.train_data.y, labels)
    _assert_labels_match_rows(bundle.test_mimic.X, bundle.test_mimic.y, labels)
    _assert_labels_match_rows(bundle.test_tudd.X, bundle.test_tudd.y, labels)

    train_ids = set(bundle.train_data.X["record_id"])
    assert train_ids.isdisjoint(bundle.test_mimic.X["record_id"])
    assert train_ids.isdisjoint(bundle.test_tudd.X["record_id"])

    summary = dataset.summarize(bundle)
    assert summary.target == "mortality"
    assert summary.train.row_count == 8
    assert {data_file.dataset_name for data_file in summary.data_files} == {
        "mimic",
        "tudd",
    }
    assert {data_file.file_name for data_file in summary.data_files} == {
        "mimic4_mean_100_full.csv",
        "tudd_mean_100_full.csv",
    }
    assert all(
        data_file.sha256 and len(data_file.sha256) == 64
        for data_file in summary.data_files
    )
    assert not (tmp_path / "filtered" / "mimic4_readmission.csv").exists()
    assert not (tmp_path / "filtered" / "tudd_readmission.csv").exists()


def test_readmission_dataset_uses_readmission_task_policy_without_normal_files(
    tmp_path, monkeypatch
):
    mimic_readmission = _make_rows("mimic", 300, 10).assign(mimic_only_feature=1.0)
    tudd_readmission = _make_rows("tudd", 400, 10).assign(tudd_only_feature=2.0)
    _write_filtered_files(
        tmp_path,
        {
            "mimic4_readmission.csv": mimic_readmission,
            "tudd_readmission.csv": tudd_readmission,
        },
    )
    monkeypatch.setattr(dataset_module, "config", SimpleNamespace(dir_data=tmp_path))

    dataset = Dataset(
        _dataset_params(
            target="hours_to_readmit",
            train_on=(
                DataSplitParams(dataset="mimic_readmission", fraction=1.0),
                DataSplitParams(dataset="tudd_readmission", fraction=1.0),
            ),
        )
    )

    bundle = dataset.get_dataset()

    assert len(bundle.train_data.X) == 10
    assert len(bundle.test_mimic.X) == 5
    assert len(bundle.test_tudd.X) == 5
    assert set(bundle.train_data.y.unique()) <= {0, 1}
    assert set(bundle.test_mimic.y.unique()) <= {0, 1}
    assert set(bundle.test_tudd.y.unique()) <= {0, 1}
    _assert_common_feature_shape(bundle)

    labels = _labels_by_record_id(
        pd.concat([mimic_readmission, tudd_readmission]), "hours_to_readmit"
    )
    _assert_labels_match_rows(bundle.train_data.X, bundle.train_data.y, labels)
    _assert_labels_match_rows(bundle.test_mimic.X, bundle.test_mimic.y, labels)
    _assert_labels_match_rows(bundle.test_tudd.X, bundle.test_tudd.y, labels)

    summary = dataset.summarize(bundle)
    assert summary.target == "hours_to_readmit"
    assert {data_file.dataset_name for data_file in summary.data_files} == {
        "mimic_readmission",
        "tudd_readmission",
    }
    assert {data_file.file_name for data_file in summary.data_files} == {
        "mimic4_readmission.csv",
        "tudd_readmission.csv",
    }
    assert not (tmp_path / "filtered" / "mimic4_mean_100_full.csv").exists()
    assert not (tmp_path / "filtered" / "tudd_mean_100_full.csv").exists()
