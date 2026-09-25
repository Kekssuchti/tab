from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.schemas.dataset_schemas import DatasetOrigin, DatasetSummary

PREDICTION_DATASETS: tuple[DatasetOrigin, ...] = ("mimic", "tudd")
RETRIEVER_DATASET = "retriever"
PREDICTION_SCHEMA_VERSION = "2"
_SUPPORTED_PREDICTION_SCHEMA_VERSIONS = {"1", PREDICTION_SCHEMA_VERSION}
PREDICTION_MANIFEST_FILENAME = "manifest.json"
TEST_SET_ID_COLUMN = "test_set_id"
Y_TRUE_COLUMN = "y_true"
PREDICTION_COLUMN_PREFIX = "y_pred_"
_BASE_COLUMNS = [TEST_SET_ID_COLUMN, Y_TRUE_COLUMN]


@dataclass(frozen=True)
class BinaryTestPredictions:
    """Labels and positive-class probabilities for one held-out dataset."""

    test_set_id: np.ndarray
    y_true: np.ndarray
    positive_class_probability: np.ndarray

    def __post_init__(self) -> None:
        ids = np.asarray(self.test_set_id).copy()
        labels = np.asarray(self.y_true, dtype=np.int8).copy()
        probabilities = np.asarray(self.positive_class_probability, dtype=float).copy()
        if ids.ndim != 1 or labels.ndim != 1 or probabilities.ndim != 1:
            raise ValueError("Prediction values must be one-dimensional")
        if len(ids) != len(labels) or len(ids) != len(probabilities):
            raise ValueError("Prediction IDs, labels, and probabilities must have equal lengths")
        if not np.isin(labels, (0, 1)).all():
            raise ValueError("Prediction labels must be binary")
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("Predictions must be probabilities between 0 and 1")

        for value in (ids, labels, probabilities):
            value.setflags(write=False)
        object.__setattr__(self, "test_set_id", ids)
        object.__setattr__(self, "y_true", labels)
        object.__setattr__(self, "positive_class_probability", probabilities)


@dataclass(frozen=True)
class FinalTestPredictions:
    mimic: BinaryTestPredictions
    tudd: BinaryTestPredictions
    retriever: BinaryTestPredictions | None = None


@dataclass(frozen=True)
class PredictionSnapshot:
    tables: dict[str, pd.DataFrame]
    generation_id: str

    @property
    def model_instance_ids(self) -> tuple[str, ...]:
        return _model_instance_ids(next(iter(self.tables.values())))


class PredictionTableAccumulator:
    """Build cumulative prediction tables for all evaluated cohorts."""

    def __init__(self, cohort_fingerprints: Mapping[str, str] | None = None) -> None:
        self._cohort_fingerprints = dict(cohort_fingerprints or {})
        self._tables: dict[str, pd.DataFrame] = {}

    def __bool__(self) -> bool:
        return bool(self._tables)

    @property
    def model_instance_ids(self) -> tuple[str, ...]:
        return _model_instance_ids(next(iter(self._tables.values()))) if self._tables else ()

    def add(self, model_instance_id: str, predictions: FinalTestPredictions) -> None:
        column = f"{PREDICTION_COLUMN_PREFIX}{model_instance_id}"
        prediction_sets = {dataset: getattr(predictions, dataset) for dataset in PREDICTION_DATASETS}
        if predictions.retriever is not None:
            prediction_sets[RETRIEVER_DATASET] = predictions.retriever
        # Retriever-only runs retain required MIMIC/TUDD fields as empty
        # structural placeholders; only populated cohorts become artifacts.
        prediction_sets = {
            dataset: values for dataset, values in prediction_sets.items() if len(values.test_set_id) > 0
        }
        incoming = {
            dataset: _prediction_frame(
                values,
                column,
                self._cohort_fingerprints[dataset],
            )
            for dataset, values in prediction_sets.items()
        }

        if not self._tables:
            self._tables = incoming
            return

        if incoming.keys() != self._tables.keys():
            raise ValueError("The evaluated test sets changed between models")

        # add the new col to existing table
        updated = {}
        for dataset, current in self._tables.items():
            if not current[_BASE_COLUMNS].equals(incoming[dataset][_BASE_COLUMNS]):
                raise ValueError(f"The {dataset} test set changed between models")
            updated[dataset] = current.assign(**{column: incoming[dataset][column]})
        self._tables = updated

    def frames(self) -> dict[str, pd.DataFrame]:
        return {dataset: table.copy() for dataset, table in self._tables.items()}

    def write(self, output_dir: str | Path) -> None:
        if not self._tables:
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_hashes = {}
        for dataset, table in self._tables.items():
            path = output_dir / f"{dataset}.csv"
            table.to_csv(path, index=False)
            file_hashes[dataset] = _file_sha256(path)

        datasets = tuple(self._tables)
        generation_id = _sha256("|".join(file_hashes[dataset] for dataset in datasets))
        manifest = {
            "version": PREDICTION_SCHEMA_VERSION,
            "generation_id": generation_id,
            "datasets": list(datasets),
            "cohort_fingerprints": {
                dataset: self._cohort_fingerprints[dataset] for dataset in datasets
            },
            "files": {dataset: {"sha256": file_hashes[dataset]} for dataset in datasets},
        }
        (output_dir / PREDICTION_MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def cohort_fingerprints_from_summary(summary: DatasetSummary) -> dict[str, str]:
    """Fingerprint each evaluation cohort without exposing clinical identifiers."""

    source_hashes = {data_file.data_origin: data_file.sha256 for data_file in summary.data_files}
    fingerprints = {
        dataset: _sha256(f"{summary.target}:{dataset}:{source_hashes[dataset]}")
        for dataset in PREDICTION_DATASETS
    }
    retriever_sources = ":".join(f"{dataset}:{source_hashes[dataset]}" for dataset in PREDICTION_DATASETS)
    fingerprints[RETRIEVER_DATASET] = _sha256(f"{summary.target}:{RETRIEVER_DATASET}:{retriever_sources}")
    return fingerprints


def load_prediction_snapshot(snapshot_dir: str | Path) -> PredictionSnapshot:
    """Load a complete prediction snapshot and reject partial MLflow uploads."""

    snapshot_dir = Path(snapshot_dir)
    manifest = json.loads((snapshot_dir / PREDICTION_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    version = manifest["version"]
    if version not in _SUPPORTED_PREDICTION_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported prediction snapshot version: {version}")

    datasets = PREDICTION_DATASETS if version == "1" else tuple(manifest["datasets"])
    tables = {}
    hashes = {}
    for dataset in datasets:
        path = snapshot_dir / f"{dataset}.csv"
        hashes[dataset] = _file_sha256(path)
        if hashes[dataset] != manifest["files"][dataset]["sha256"]:
            raise ValueError(f"Prediction CSV hash mismatch for {dataset}")
        tables[dataset] = pd.read_csv(path)

    generation_id = _sha256("|".join(hashes[dataset] for dataset in datasets))
    if generation_id != manifest["generation_id"]:
        raise ValueError("Prediction snapshot generation does not match its files")

    first_table = next(iter(tables.values()))
    model_columns = [column for column in first_table if is_prediction_column(column)]
    for dataset, table in tables.items():
        if list(table.columns) != [*_BASE_COLUMNS, *model_columns]:
            raise ValueError(f"Prediction columns differ for {dataset}")

    return PredictionSnapshot(tables=tables, generation_id=generation_id)


def is_prediction_column(column: object) -> bool:
    return isinstance(column, str) and column.startswith(PREDICTION_COLUMN_PREFIX)


def _model_instance_ids(table: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        column.removeprefix(PREDICTION_COLUMN_PREFIX) for column in table.columns if is_prediction_column(column)
    )


def _prediction_frame(
    predictions: BinaryTestPredictions,
    column: str,
    cohort_fingerprint: str,
) -> pd.DataFrame:
    ids = [_stable_test_set_id(cohort_fingerprint, source_id) for source_id in predictions.test_set_id]
    return pd.DataFrame(
        {
            TEST_SET_ID_COLUMN: ids,
            Y_TRUE_COLUMN: predictions.y_true,
            column: predictions.positive_class_probability,
        }
    )


def _stable_test_set_id(cohort_fingerprint: str, source_id: object) -> str:
    if isinstance(source_id, np.generic):
        source_id = source_id.item()
    if isinstance(source_id, bool) or not isinstance(source_id, int):
        raise TypeError("Test-set indexes must be integer source-row positions")
    return f"ts_{_sha256(f'{cohort_fingerprint}:{source_id}')}"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
