from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.schemas.dataset_schemas import DatasetSummary

PREDICTION_DATASETS = ("mimic", "tudd")
PREDICTION_SCHEMA_VERSION = "1"
PREDICTION_MANIFEST_FILENAME = "manifest.json"
TEST_SET_ID_COLUMN = "test_set_id"
Y_TRUE_COLUMN = "y_true"
PREDICTION_COLUMN_PREFIX = "y_pred_"
_ID_STRATEGY = "sha256-source-file-and-row-index-v1"
_BASE_COLUMNS = (TEST_SET_ID_COLUMN, Y_TRUE_COLUMN)


@dataclass(frozen=True)
class BinaryTestPredictions:
    """Positive-class probabilities and labels for one held-out test set."""

    test_set_id: np.ndarray
    y_true: np.ndarray
    positive_class_probability: np.ndarray

    def __post_init__(self) -> None:
        test_set_id = np.asarray(self.test_set_id)
        y_true = np.asarray(self.y_true)
        probability = np.asarray(self.positive_class_probability, dtype=float)

        if test_set_id.ndim != 1 or y_true.ndim != 1 or probability.ndim != 1:
            raise ValueError("Prediction artifact values must be one-dimensional")
        if not (test_set_id.size == y_true.size == probability.size):
            raise ValueError("Prediction artifact IDs, labels, and probabilities must have equal lengths")
        if pd.isna(test_set_id).any():
            raise ValueError("test_set_id values must not be missing")
        if not pd.Index(test_set_id).is_unique:
            raise ValueError("test_set_id values must be unique within a test set")
        if not np.isin(y_true, (0, 1)).all():
            raise ValueError("Prediction artifact y_true values must be encoded as 0 or 1")
        if not np.isfinite(probability).all():
            raise ValueError("Prediction artifact probabilities must be finite")
        if ((probability < 0.0) | (probability > 1.0)).any():
            raise ValueError("Prediction artifact probabilities must lie between 0 and 1")

        test_set_id = test_set_id.copy()
        y_true = y_true.astype(np.int8, copy=True)
        probability = probability.copy()
        test_set_id.setflags(write=False)
        y_true.setflags(write=False)
        probability.setflags(write=False)
        object.__setattr__(self, "test_set_id", test_set_id)
        object.__setattr__(self, "y_true", y_true)
        object.__setattr__(self, "positive_class_probability", probability)


@dataclass(frozen=True)
class FinalTestPredictions:
    """Binary predictions produced by one model on both held-out sources."""

    mimic: BinaryTestPredictions
    tudd: BinaryTestPredictions


@dataclass(frozen=True)
class PredictionSnapshot:
    """Validated prediction tables and their integrity metadata."""

    tables: dict[str, pd.DataFrame]
    generation_id: str
    model_instance_ids: tuple[str, ...]
    cohort_fingerprints: dict[str, str]


class PredictionTableAccumulator:
    """Accumulate one positive-class probability column per model instance."""

    def __init__(self, cohort_fingerprints: Mapping[str, str] | None = None) -> None:
        supplied = (
            {dataset: "unversioned" for dataset in PREDICTION_DATASETS}
            if cohort_fingerprints is None
            else cohort_fingerprints
        )
        if set(supplied) != set(PREDICTION_DATASETS):
            raise ValueError("Cohort fingerprints must contain exactly the MIMIC and TUDD datasets")
        if any(not isinstance(value, str) or not value for value in supplied.values()):
            raise ValueError("Cohort fingerprints must be non-empty strings")
        self._cohort_fingerprints = {dataset: supplied[dataset] for dataset in PREDICTION_DATASETS}
        self._tables: dict[str, pd.DataFrame] = {}

    def __bool__(self) -> bool:
        return bool(self._tables)

    @property
    def model_instance_ids(self) -> tuple[str, ...]:
        if not self._tables:
            return ()
        first = self._tables[PREDICTION_DATASETS[0]]
        return tuple(
            column.removeprefix(PREDICTION_COLUMN_PREFIX) for column in first.columns if is_prediction_column(column)
        )

    def add(self, model_instance_id: str, predictions: FinalTestPredictions) -> None:
        """Atomically add a model column to both dataset tables."""

        column = prediction_column(model_instance_id)
        incoming = {
            "mimic": _prediction_frame(
                "mimic",
                predictions.mimic,
                column,
                self._cohort_fingerprints["mimic"],
            ),
            "tudd": _prediction_frame(
                "tudd",
                predictions.tudd,
                column,
                self._cohort_fingerprints["tudd"],
            ),
        }
        updated: dict[str, pd.DataFrame] = {}

        for dataset in PREDICTION_DATASETS:
            current = self._tables.get(dataset)
            model_frame = incoming[dataset]
            if current is None:
                updated[dataset] = model_frame
                continue
            if column in current.columns:
                raise ValueError(f"Prediction table already contains model instance {model_instance_id!r}")
            if not current.loc[:, list(_BASE_COLUMNS)].equals(model_frame.loc[:, list(_BASE_COLUMNS)]):
                raise ValueError(f"Prediction IDs or labels changed for the {dataset} test set")

            next_frame = current.copy()
            next_frame[column] = model_frame[column].to_numpy(copy=True)
            updated[dataset] = next_frame

        self._tables.update(updated)

    def frames(self) -> dict[str, pd.DataFrame]:
        """Return defensive copies of the current per-dataset tables."""

        return {dataset: self._tables[dataset].copy() for dataset in PREDICTION_DATASETS if dataset in self._tables}

    def write_csvs(self, output_dir: str | Path) -> tuple[Path, ...]:
        """Write two CSVs and an integrity manifest for the current snapshot."""

        if not self._tables:
            return ()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        files: dict[str, dict[str, object]] = {}
        for dataset in PREDICTION_DATASETS:
            path = output_dir / f"{dataset}.csv"
            temporary_path = path.with_suffix(".csv.tmp")
            self._tables[dataset].to_csv(temporary_path, index=False)
            temporary_path.replace(path)
            paths.append(path)
            files[dataset] = {
                "path": path.name,
                "sha256": _file_sha256(path),
                "row_count": len(self._tables[dataset]),
            }

        manifest_without_generation: dict[str, object] = {
            "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
            "id_strategy": _ID_STRATEGY,
            "positive_class_label": 1,
            "model_instance_ids": list(self.model_instance_ids),
            "cohort_fingerprints": self._cohort_fingerprints,
            "files": files,
        }
        manifest = {
            **manifest_without_generation,
            "generation_id": _content_sha256(manifest_without_generation),
        }
        manifest_path = output_dir / PREDICTION_MANIFEST_FILENAME
        temporary_manifest_path = manifest_path.with_suffix(".json.tmp")
        temporary_manifest_path.write_text(_canonical_json(manifest), encoding="utf-8")
        temporary_manifest_path.replace(manifest_path)
        return tuple(paths)


def cohort_fingerprints_from_summary(summary: DatasetSummary) -> dict[str, str]:
    """Return source-file fingerprints used to namespace non-PHI test IDs."""

    fingerprints: dict[str, str] = {}
    for dataset in PREDICTION_DATASETS:
        matching = [data_file for data_file in summary.data_files if data_file.data_origin == dataset]
        if len(matching) != 1:
            raise ValueError(f"Expected one source file for {dataset!r}, found {len(matching)}")
        if matching[0].sha256 is None:
            raise ValueError(f"Source file hash is unavailable for {dataset!r}")
        fingerprints[dataset] = "cohort_v1:" + _content_sha256(
            {
                "dataset": dataset,
                "target": summary.target,
                "source_file_sha256": matching[0].sha256,
            }
        )
    return fingerprints


def load_prediction_snapshot(snapshot_dir: str | Path) -> PredictionSnapshot:
    """Load prediction CSVs only after schema, generation, and hash validation."""

    snapshot_dir = Path(snapshot_dir)
    manifest_path = snapshot_dir / PREDICTION_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_keys = {
        "prediction_schema_version",
        "generation_id",
        "id_strategy",
        "positive_class_label",
        "model_instance_ids",
        "cohort_fingerprints",
        "files",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise ValueError("Prediction manifest has an invalid set of fields")
    if manifest["prediction_schema_version"] != PREDICTION_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported prediction schema version "
            f"{manifest['prediction_schema_version']!r}; expected {PREDICTION_SCHEMA_VERSION!r}"
        )
    if manifest["id_strategy"] != _ID_STRATEGY or manifest["positive_class_label"] != 1:
        raise ValueError("Prediction manifest class or test-set ID contract is unsupported")

    generation_payload = {key: value for key, value in manifest.items() if key != "generation_id"}
    expected_generation = _content_sha256(generation_payload)
    if manifest["generation_id"] != expected_generation:
        raise ValueError("Prediction manifest generation ID does not match its contents")

    model_instance_ids = manifest["model_instance_ids"]
    if (
        not isinstance(model_instance_ids, list)
        or not model_instance_ids
        or any(not isinstance(value, str) or not value for value in model_instance_ids)
        or len(model_instance_ids) != len(set(model_instance_ids))
    ):
        raise ValueError("Prediction manifest model_instance_ids must be unique non-empty strings")

    cohort_fingerprints = manifest["cohort_fingerprints"]
    if (
        not isinstance(cohort_fingerprints, dict)
        or set(cohort_fingerprints) != set(PREDICTION_DATASETS)
        or any(not isinstance(value, str) or not value for value in cohort_fingerprints.values())
    ):
        raise ValueError("Prediction manifest cohort fingerprints are invalid")

    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != set(PREDICTION_DATASETS):
        raise ValueError("Prediction manifest must describe MIMIC and TUDD files")

    expected_columns = [
        TEST_SET_ID_COLUMN,
        Y_TRUE_COLUMN,
        *(prediction_column(model_instance_id) for model_instance_id in model_instance_ids),
    ]
    tables: dict[str, pd.DataFrame] = {}
    for dataset in PREDICTION_DATASETS:
        metadata = files[dataset]
        if not isinstance(metadata, dict) or set(metadata) != {"path", "sha256", "row_count"}:
            raise ValueError(f"Prediction manifest metadata for {dataset!r} is invalid")
        if metadata["path"] != f"{dataset}.csv":
            raise ValueError(f"Prediction manifest path for {dataset!r} is invalid")
        path = snapshot_dir / metadata["path"]
        if _file_sha256(path) != metadata["sha256"]:
            raise ValueError(f"Prediction CSV hash mismatch for {dataset!r}")

        table = pd.read_csv(path)
        if len(table) != metadata["row_count"]:
            raise ValueError(f"Prediction CSV row count mismatch for {dataset!r}")
        if list(table.columns) != expected_columns:
            raise ValueError(f"Prediction CSV columns do not match the manifest for {dataset!r}")
        for column in expected_columns[2:]:
            BinaryTestPredictions(
                test_set_id=table[TEST_SET_ID_COLUMN].to_numpy(),
                y_true=table[Y_TRUE_COLUMN].to_numpy(),
                positive_class_probability=table[column].to_numpy(),
            )
        tables[dataset] = table

    return PredictionSnapshot(
        tables=tables,
        generation_id=manifest["generation_id"],
        model_instance_ids=tuple(model_instance_ids),
        cohort_fingerprints={dataset: cohort_fingerprints[dataset] for dataset in PREDICTION_DATASETS},
    )


def prediction_column(model_instance_id: str) -> str:
    if not isinstance(model_instance_id, str) or not model_instance_id:
        raise ValueError("model_instance_id must be a non-empty string")
    return f"{PREDICTION_COLUMN_PREFIX}{model_instance_id}"


def is_prediction_column(column: object) -> bool:
    return (
        isinstance(column, str)
        and column.startswith(PREDICTION_COLUMN_PREFIX)
        and len(column) > len(PREDICTION_COLUMN_PREFIX)
    )


def _prediction_frame(
    dataset: str,
    predictions: BinaryTestPredictions,
    column: str,
    cohort_fingerprint: str,
) -> pd.DataFrame:
    stable_ids = [_stable_test_set_id(dataset, cohort_fingerprint, source_id) for source_id in predictions.test_set_id]
    return pd.DataFrame(
        {
            TEST_SET_ID_COLUMN: stable_ids,
            Y_TRUE_COLUMN: predictions.y_true,
            column: predictions.positive_class_probability,
        }
    )


def _stable_test_set_id(dataset: str, cohort_fingerprint: str, source_id: object) -> str:
    if isinstance(source_id, np.generic):
        source_id = source_id.item()
    if isinstance(source_id, bool) or not isinstance(source_id, int):
        raise TypeError("Source test indices must be integer row positions, not patient identifiers")
    payload = {
        "dataset": dataset,
        "cohort_fingerprint": cohort_fingerprint,
        "source_index_type": type(source_id).__name__,
        "source_index": str(source_id),
    }
    return f"ts_{_content_sha256(payload)}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
