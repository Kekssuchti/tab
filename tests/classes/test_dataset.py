import math
from dataclasses import asdict, fields
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from src.classes import dataset as dataset_module
from src.classes.data_registry import TARGET_LIKE_COLUMNS, DatasetTask, dataset_task_for_target
from src.classes.dataset import Dataset
from src.schemas.dataset_schemas import (
    ClassificationTargetSummary,
    DatasetBundle,
    DatasetConfig,
    DataSplitConfig,
    RegressionTargetSummary,
    XYDataset,
)
from src.utils.dataset_utils import summarize_data_part


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
    train_on: tuple[DataSplitConfig, ...],
    train_size: float = 0.5,
    random_state: int = 7,
) -> DatasetConfig:
    return DatasetConfig(
        target=target,
        train_on=train_on,
        train_size=train_size,
        random_state=random_state,
    )


def _labels_by_record_id(df: pd.DataFrame, target: str) -> dict[int, int | float]:
    if target == "hours_to_readmit":
        return df.set_index("record_id")[target].notna().astype(int).to_dict()
    return df.set_index("record_id")[target].to_dict()


def _assert_labels_match_rows(X: pd.DataFrame, y: pd.Series, labels_by_id: dict[int, int | float]) -> None:
    actual = dict(zip(X["record_id"], y, strict=True))
    expected = {record_id: labels_by_id[record_id] for record_id in X["record_id"]}
    assert actual == expected


def _assert_common_feature_shape(bundle, *, include_los: bool = False) -> None:
    expected_columns = {"record_id", "source", "Age", "shared_feature"}
    if include_los:
        expected_columns.add("LOS")
    assert set(bundle.train_data.X.columns) == expected_columns
    assert set(bundle.test_mimic.X.columns) == expected_columns
    assert set(bundle.test_tudd.X.columns) == expected_columns
    excluded_targets = set(TARGET_LIKE_COLUMNS) - ({"LOS"} if include_los else set())
    assert excluded_targets.isdisjoint(bundle.train_data.X.columns)
    assert excluded_targets.isdisjoint(bundle.test_mimic.X.columns)
    assert excluded_targets.isdisjoint(bundle.test_tudd.X.columns)


@pytest.mark.parametrize(
    ("target", "task_type", "dataset_kind", "expected_labels"),
    [
        ("mortality", "classification", "normal", [0, 1, 0]),
        ("LOS7", "classification", "normal", [0, 0, 1]),
        ("hours_to_readmit", "classification", "readmission", [0, 1, 0]),
        ("LOS", "regression", "normal", [167.5, 168.0, 168.5]),
    ],
)
def test_target_defines_all_task_metadata(target, task_type, dataset_kind, expected_labels):
    task = dataset_task_for_target(target)
    df = pd.DataFrame(
        {
            "mortality": [0, 1, 0],
            "LOS": [167.5, 168.0, 168.5],
            "hours_to_readmit": [None, 24.0, None],
        }
    )

    assert task.task_type == task_type
    assert task.dataset_kind == dataset_kind
    assert task.labels_from(df).tolist() == expected_labels
    assert set(task.data_files) == {"mimic", "tudd"}
    assert {data_file.data_origin for data_file in task.data_files.values()} == {"mimic", "tudd"}
    assert {data_file.dataset_kind for data_file in task.data_files.values()} == {dataset_kind}


def test_dataset_task_has_only_target_as_dataclass_field():
    assert [field.name for field in fields(DatasetTask)] == ["target"]
    assert dataset_task_for_target("mortality") == DatasetTask("mortality")


def test_feature_selection_does_not_mutate_target_like_columns():
    df = _make_rows("mimic", 1, 2)

    normal_features = dataset_task_for_target("mortality").features_from(df)
    readmission_features = dataset_task_for_target("hours_to_readmit").features_from(df)

    assert isinstance(TARGET_LIKE_COLUMNS, tuple)
    assert "LOS" not in normal_features
    assert "LOS" in readmission_features


def test_dataset_config_rejects_duplicate_origins():
    with pytest.raises(ValidationError, match="Duplicate dataset origins detected"):
        _dataset_params(
            target="mortality",
            train_on=(
                DataSplitConfig(dataset="mimic", fraction=1.0),
                DataSplitConfig(dataset="mimic", fraction=2),
            ),
        )


def test_dataset_config_rejects_independent_classification_flag():
    with pytest.raises(ValidationError, match="classification"):
        DatasetConfig(
            target="mortality",
            train_on=(DataSplitConfig(dataset="mimic", fraction=1.0),),
            classification=True,
        )


def test_dataset_config_only_allows_log_transform_for_los():
    los_config = DatasetConfig(
        target="LOS",
        train_on=(DataSplitConfig(dataset="mimic", fraction=1.0),),
        log_transform_target=True,
    )

    assert los_config.log_transform_target

    with pytest.raises(ValidationError, match="only supported for the LOS target"):
        DatasetConfig(
            target="mortality",
            log_transform_target=True,
            train_on=(DataSplitConfig(dataset="mimic", fraction=1.0),),
        )


def test_los_regression_split_does_not_stratify_continuous_labels():
    df = _make_rows("mimic", 1, 10)
    dataset = Dataset(
        _dataset_params(
            target="LOS",
            train_on=(DataSplitConfig(dataset="mimic", fraction=1.0),),
        )
    )

    X_train, X_test, y_train, y_test = dataset._split_single_df(df)

    labels = df.set_index("record_id")["LOS"].to_dict()
    _assert_labels_match_rows(X_train, y_train, labels)
    _assert_labels_match_rows(X_test, y_test, labels)


def test_existing_target_filtered_files_do_not_invoke_cleaner(tmp_path, monkeypatch):
    _write_filtered_files(
        tmp_path,
        {
            "mimic4_mean_100_full.csv": _make_rows("mimic", 1, 4),
            "tudd_mean_100_full.csv": _make_rows("tudd", 10, 4),
        },
    )
    monkeypatch.setattr(dataset_module, "config", SimpleNamespace(dir_data=tmp_path))
    dataset = Dataset(
        _dataset_params(
            target="mortality",
            train_on=(DataSplitConfig(dataset="mimic", fraction=1.0),),
        )
    )

    def fail_if_called(dataset_kind=None):
        pytest.fail(f"cleaner unexpectedly called for {dataset_kind}")

    monkeypatch.setattr(dataset.data_cleaner, "preprocess_extracted_to_filtered", fail_if_called)

    loaded = dataset._load_data()

    assert set(loaded) == {"mimic", "tudd"}


def test_missing_filtered_file_preprocesses_only_target_kind(tmp_path, monkeypatch):
    (tmp_path / "filtered").mkdir()
    monkeypatch.setattr(dataset_module, "config", SimpleNamespace(dir_data=tmp_path))
    dataset = Dataset(
        _dataset_params(
            target="hours_to_readmit",
            train_on=(DataSplitConfig(dataset="mimic", fraction=1.0),),
        )
    )
    calls = []

    def write_selected_kind(dataset_kind):
        calls.append(dataset_kind)
        for file_name, source, start_id in (
            ("mimic4_readmission.csv", "mimic", 1),
            ("tudd_readmission.csv", "tudd", 10),
        ):
            _make_rows(source, start_id, 4).to_csv(tmp_path / "filtered" / file_name, index=False)

    monkeypatch.setattr(dataset.data_cleaner, "preprocess_extracted_to_filtered", write_selected_kind)

    dataset._load_data()

    assert calls == ["readmission"]


def test_force_repreprocesses_only_target_kind(tmp_path, monkeypatch):
    _write_filtered_files(
        tmp_path,
        {
            "mimic4_mean_100_full.csv": _make_rows("mimic", 1, 4),
            "tudd_mean_100_full.csv": _make_rows("tudd", 10, 4),
        },
    )
    monkeypatch.setattr(dataset_module, "config", SimpleNamespace(dir_data=tmp_path))
    params = _dataset_params(
        target="mortality",
        train_on=(DataSplitConfig(dataset="mimic", fraction=1.0),),
    ).model_copy(update={"force_repreprocess": True})
    dataset = Dataset(params)
    calls = []
    monkeypatch.setattr(dataset.data_cleaner, "preprocess_extracted_to_filtered", calls.append)

    dataset._load_data()

    assert calls == ["normal"]


def test_summarize_data_part_preserves_classification_balance():
    part = XYDataset(X=pd.DataFrame(index=range(3)), y=pd.Series([0, 1, 1]))

    summary = summarize_data_part(part, "classification")

    assert summary.row_count == 3
    assert summary.target_summary == ClassificationTargetSummary(class_balance={"0": 1, "1": 2})


def test_regression_target_summary_is_finite_and_bounded():
    part = XYDataset(
        X=pd.DataFrame(index=range(6)),
        y=pd.Series([1.0, 2.0, 3.0, float("nan"), float("inf"), float("-inf")]),
    )

    summary = summarize_data_part(part, "regression")

    assert summary.row_count == 6
    assert isinstance(summary.target_summary, RegressionTargetSummary)
    assert summary.target_summary.count == 3
    assert summary.target_summary.mean == 2.0
    assert summary.target_summary.std == 1.0
    assert summary.target_summary.min == 1.0
    assert summary.target_summary.max == 3.0
    serialized_target = asdict(summary.target_summary)
    assert set(serialized_target) == {"count", "mean", "std", "min", "max"}
    assert all(math.isfinite(value) for value in serialized_target.values())


def test_dataset_summarize_uses_target_task_type(monkeypatch):
    dataset = Dataset(
        _dataset_params(
            target="LOS",
            train_on=(DataSplitConfig(dataset="mimic", fraction=1.0),),
        )
    )
    part = XYDataset(X=pd.DataFrame(index=range(2)), y=pd.Series([48.0, 72.0]))
    bundle = DatasetBundle(train_data=part, test_mimic=part, test_tudd=part)
    monkeypatch.setattr(dataset, "_summarize_data_files", list)

    summary = dataset.summarize(bundle)

    assert isinstance(summary.train.target_summary, RegressionTargetSummary)
    assert isinstance(summary.test_mimic.target_summary, RegressionTargetSummary)
    assert isinstance(summary.test_tudd.target_summary, RegressionTargetSummary)


def test_mortality_dataset_uses_only_normal_task_files_and_assembles_splits(tmp_path, monkeypatch):
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
                DataSplitConfig(dataset="mimic", fraction=1.0),
                DataSplitConfig(dataset="tudd", fraction=2),
            ),
        )
    )

    bundle = dataset.get_dataset()

    assert len(bundle.train_data.X) == 8
    assert len(bundle.test_mimic.X) == 6
    assert len(bundle.test_tudd.X) == 6
    assert (bundle.train_data.X["source"] == "mimic").sum() == 6
    assert (bundle.train_data.X["source"] == "tudd").sum() == 2
    assert bundle.train_data.X["Age"].max() <= 91
    assert bundle.test_tudd.X["Age"].max() <= 91
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
    assert isinstance(summary.train.target_summary, ClassificationTargetSummary)
    assert {data_file.dataset_name for data_file in summary.data_files} == {
        "mimic",
        "tudd",
    }
    assert {data_file.file_name for data_file in summary.data_files} == {
        "mimic4_mean_100_full.csv",
        "tudd_mean_100_full.csv",
    }
    assert all(data_file.sha256 and len(data_file.sha256) == 64 for data_file in summary.data_files)
    assert not (tmp_path / "filtered" / "mimic4_readmission.csv").exists()
    assert not (tmp_path / "filtered" / "tudd_readmission.csv").exists()


def test_combined_classification_training_subsamples_are_stratified():
    mortality = [0] * 16 + [1] * 4
    mimic = _make_rows("mimic", 100, 20).assign(mortality=mortality)
    tudd = _make_rows("tudd", 200, 20).assign(mortality=mortality)
    dataset = Dataset(
        _dataset_params(
            target="mortality",
            train_on=(
                DataSplitConfig(dataset="mimic", fraction=5),
                DataSplitConfig(dataset="tudd", fraction=5),
            ),
            random_state=7,
        )
    )

    bundle = dataset._split_data({"mimic": mimic, "tudd": tudd})

    assert bundle.train_data.y.value_counts().to_dict() == {0: 8, 1: 2}


def test_readmission_dataset_uses_readmission_task_policy_without_normal_files(tmp_path, monkeypatch):
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
                DataSplitConfig(dataset="mimic", fraction=1.0),
                DataSplitConfig(dataset="tudd", fraction=1.0),
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
    _assert_common_feature_shape(bundle, include_los=True)

    labels = _labels_by_record_id(pd.concat([mimic_readmission, tudd_readmission]), "hours_to_readmit")
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
