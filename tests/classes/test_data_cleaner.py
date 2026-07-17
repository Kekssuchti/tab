import json
from types import SimpleNamespace

import pandas as pd

from src.classes import data_cleaner as data_cleaner_module
from src.classes.data_cleaner import DataCleaner
from src.schemas.dataset_schemas import DataCleanerConfig


def _normal_raw_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4],
            "Age": [70, 17, 55, 65],
            "Sex": ["F", "M", "F", "M"],
            "LOS": [48.0, 48.0, 12.0, 72.0],
            "mortality": [0, 0, 0, 1],
            "lab_value": [999.0, 20.0, 30.0, 40.0],
        }
    )


def _readmission_raw_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": [10, 11, 12],
            "subject_id": [100, 101, 102],
            "hadm_id": [200, 201, 202],
            "stay_id": [300, 301, 302],
            "Age": [80, 70, 16],
            "Sex": ["F", "M", "F"],
            "LOS": [60.0, 60.0, 60.0],
            "mortality": [0, 1, 0],
            "hours_to_readmit": [36.0, 48.0, None],
            "lab_value": [25.0, 35.0, 45.0],
        }
    )


def _write_extracted_data(tmp_path) -> None:
    extracted_path = tmp_path / "extracted"
    filtered_path = tmp_path / "filtered"
    extracted_path.mkdir()
    filtered_path.mkdir()

    normal = _normal_raw_data()
    readmission = _readmission_raw_data()
    normal.to_csv(extracted_path / "mimic4_mean_100_full.csv", index=False)
    normal.to_csv(extracted_path / "tudd_mean_100_full.csv", index=False)
    readmission.to_csv(extracted_path / "mimic4_readmission.csv", index=False)
    readmission.to_csv(extracted_path / "tudd_readmission.csv", index=False)


def test_data_cleaner_writes_filtered_files_with_preprocessed_content(
    tmp_path, monkeypatch
):
    _write_extracted_data(tmp_path)
    limits_path = tmp_path / "limits.json"
    limits_path.write_text(
        json.dumps({"lab_value": {"lower_bound": 0, "upper_bound": 100}})
    )
    monkeypatch.setattr(
        data_cleaner_module, "config", SimpleNamespace(dir_data=tmp_path)
    )

    cleaner = DataCleaner(
        DataCleanerConfig(outlier_limits_path=limits_path, missing_threshold_row=1.0)
    )

    cleaner.preprocess_extracted_to_filtered()

    normal = pd.read_csv(tmp_path / "filtered" / "mimic4_mean_100_full.csv")
    readmission = pd.read_csv(tmp_path / "filtered" / "mimic4_readmission.csv")

    assert not any(column.startswith("Unnamed") for column in normal.columns)
    assert set(normal["record_id"]) == {1, 4}
    assert normal.set_index("record_id").loc[1, "Sex"] == 1
    assert normal.set_index("record_id").loc[4, "Sex"] == 0
    assert pd.isna(normal.set_index("record_id").loc[1, "lab_value"])

    assert not {"subject_id", "hadm_id", "stay_id"} & set(readmission.columns)
    assert set(readmission["record_id"]) == {10}
    assert readmission.set_index("record_id").loc[10, "Sex"] == 1
