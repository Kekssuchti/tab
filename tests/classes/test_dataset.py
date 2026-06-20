from types import SimpleNamespace

import pandas as pd

from src.classes import dataset as dataset_module
from src.classes.dataset import Dataset
from src.schemas.dataset_schemas import DatasetParams, DataSplitParams

TARGET_COLUMNS = {"mortality", "LOS", "hours_to_readmit"}


def _make_rows(source: str, start_id: int, n_rows: int) -> pd.DataFrame:
    record_ids = list(range(start_id, start_id + n_rows))
    return pd.DataFrame(
        {
            "record_id": record_ids,
            "source": [source] * n_rows,
            "Age": [30 + (i % 70) for i in range(n_rows)],
            "feature_value": [float(i * 10) for i in range(n_rows)],
            "mortality": [record_id % 2 for record_id in record_ids],
            "LOS": [48.0 + i for i in range(n_rows)],
            "hours_to_readmit": [12.0 if i % 3 else None for i in range(n_rows)],
        }
    )


def _write_filtered_data(
    tmp_path,
    mimic: pd.DataFrame,
    tudd: pd.DataFrame,
    mimic_readmission: pd.DataFrame | None = None,
    tudd_readmission: pd.DataFrame | None = None,
) -> None:
    filtered_path = tmp_path / "filtered"
    filtered_path.mkdir()

    mimic.to_csv(filtered_path / "mimic4_mean_100_full.csv", index=False)
    tudd.to_csv(filtered_path / "tudd_mean_100_full.csv", index=False)
    (mimic_readmission if mimic_readmission is not None else mimic).to_csv(
        filtered_path / "mimic4_readmission.csv", index=False
    )
    (tudd_readmission if tudd_readmission is not None else tudd).to_csv(
        filtered_path / "tudd_readmission.csv", index=False
    )


def _dataset_params(
    target: str,
    train_on: tuple[DataSplitParams, ...],
    train_size: float = 0.6,
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


def test_dataset_splits_selected_origin_without_train_test_overlap(
    tmp_path, monkeypatch
):
    mimic = _make_rows("mimic", 100, 10)
    tudd = _make_rows("tudd", 200, 10)
    _write_filtered_data(tmp_path, mimic, tudd)
    monkeypatch.setattr(dataset_module, "config", SimpleNamespace(dir_data=tmp_path))

    dataset = Dataset(
        _dataset_params(
            target="mortality",
            train_on=(DataSplitParams(dataset="mimic", fraction=1.0),),
        )
    )

    bundle = dataset.get_dataset()

    assert len(bundle.train_data.X) == 6
    assert len(bundle.test_mimic.X) == 4
    assert len(bundle.test_tudd.X) == 4
    assert set(bundle.train_data.X["source"]) == {"mimic"}

    train_ids = set(bundle.train_data.X["record_id"])
    mimic_test_ids = set(bundle.test_mimic.X["record_id"])
    assert train_ids.isdisjoint(mimic_test_ids)
    assert train_ids | mimic_test_ids == set(mimic["record_id"])

    assert TARGET_COLUMNS.isdisjoint(bundle.train_data.X.columns)
    labels = _labels_by_record_id(mimic, "mortality")
    _assert_labels_match_rows(bundle.train_data.X, bundle.train_data.y, labels)
    _assert_labels_match_rows(bundle.test_mimic.X, bundle.test_mimic.y, labels)


def test_dataset_combines_fractional_and_absolute_training_splits(
    tmp_path, monkeypatch
):
    mimic = _make_rows("mimic", 100, 12)
    tudd = _make_rows("tudd", 200, 12)
    _write_filtered_data(tmp_path, mimic, tudd)
    monkeypatch.setattr(dataset_module, "config", SimpleNamespace(dir_data=tmp_path))

    dataset = Dataset(
        _dataset_params(
            target="LOS",
            train_size=0.5,
            train_on=(
                DataSplitParams(dataset="mimic", fraction=0.5),
                DataSplitParams(dataset="tudd", fraction=2),
            ),
        )
    )

    bundle = dataset.get_dataset()

    assert len(bundle.train_data.X) == 5
    assert set(bundle.train_data.X["source"]) == {"mimic", "tudd"}
    assert (bundle.train_data.X["source"] == "mimic").sum() == 3
    assert (bundle.train_data.X["source"] == "tudd").sum() == 2

    train_by_source = bundle.train_data.X.groupby("source")["record_id"].agg(set)
    assert train_by_source["mimic"].isdisjoint(bundle.test_mimic.X["record_id"])
    assert train_by_source["tudd"].isdisjoint(bundle.test_tudd.X["record_id"])

    labels = _labels_by_record_id(pd.concat([mimic, tudd]), "LOS")
    _assert_labels_match_rows(bundle.train_data.X, bundle.train_data.y, labels)


def test_readmission_dataset_uses_binary_readmission_labels_and_drops_targets(
    tmp_path, monkeypatch
):
    mimic = _make_rows("mimic", 100, 10)
    tudd = _make_rows("tudd", 200, 10)
    mimic_readmission = _make_rows("mimic", 300, 10)
    tudd_readmission = _make_rows("tudd", 400, 10)
    _write_filtered_data(tmp_path, mimic, tudd, mimic_readmission, tudd_readmission)
    monkeypatch.setattr(dataset_module, "config", SimpleNamespace(dir_data=tmp_path))

    dataset = Dataset(
        _dataset_params(
            target="hours_to_readmit",
            train_size=0.5,
            train_on=(
                DataSplitParams(dataset="mimic_readmission", fraction=1.0),
                DataSplitParams(dataset="tudd_readmission", fraction=1.0),
            ),
        )
    )

    bundle = dataset.get_dataset()

    assert len(bundle.train_data.X) == 10
    assert set(bundle.train_data.y.unique()) <= {0, 1}
    assert set(bundle.test_mimic.y.unique()) <= {0, 1}
    assert set(bundle.test_tudd.y.unique()) <= {0, 1}
    assert TARGET_COLUMNS.isdisjoint(bundle.train_data.X.columns)
    assert TARGET_COLUMNS.isdisjoint(bundle.test_mimic.X.columns)
    assert TARGET_COLUMNS.isdisjoint(bundle.test_tudd.X.columns)

    labels = _labels_by_record_id(
        pd.concat([mimic_readmission, tudd_readmission]), "hours_to_readmit"
    )
    _assert_labels_match_rows(bundle.train_data.X, bundle.train_data.y, labels)
    _assert_labels_match_rows(bundle.test_mimic.X, bundle.test_mimic.y, labels)
    _assert_labels_match_rows(bundle.test_tudd.X, bundle.test_tudd.y, labels)
