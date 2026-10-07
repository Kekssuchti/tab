"""Shared readers for one run's recorded training design and evaluation artifacts.

Every figure family that compares differently composed training sets reads the
same two authoritative sources: the run's `config.json` for the configured
training contribution of each source, and the run's `prediction_metrics.csv` and
`bootstrap_metrics.csv` for the realized training count and the aligned bootstrap
draws. Keeping that reading here means composition, augmentation, and retrieval
validation cannot drift apart.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException

from mlflow import MlflowClient
from src.mlflow.tracking_contract import ARTIFACT_CONFIG
from src.plotting.utils.artifacts import PlotArtifacts
from src.utils.prediction_tables import PREDICTION_DATASETS

FULL_POOL = "full_pool"
SAMPLE_COUNT = "sample_count"


class IncompleteExperimentError(ValueError):
    """Raised when a registered experiment does not yet measure a required design.

    A missing design is reported so its figure can be skipped without
    substituting another experiment; it is not malformed data and must not abort
    a full figure-recreation run.
    """


@dataclass(frozen=True)
class TrainOnEntry:
    """One configured training contribution: the whole pool or an absolute count."""

    source: str
    kind: str
    value: int | float

    @property
    def is_full_pool(self) -> bool:
        """Whether this entry contributes the source's complete training pool."""
        return self.kind == FULL_POOL

    def describe(self) -> str:
        return f"{self.source}=1.0" if self.is_full_pool else f"{self.source}={int(self.value):,}"


@dataclass(frozen=True)
class TrainOnDesign:
    """The training-source design recorded in one run's `config.json`."""

    run_id: str
    target: str
    entries: tuple[TrainOnEntry, ...]
    retriever: Mapping[str, object] | None

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(entry.source for entry in self.entries)

    @property
    def uses_retriever(self) -> bool:
        return self.retriever is not None

    def entry(self, source: str) -> TrainOnEntry | None:
        return next((entry for entry in self.entries if entry.source == source), None)

    def is_full_pool(self, source: str) -> bool:
        entry = self.entry(source)
        return entry is not None and entry.is_full_pool

    def sample_count(self, source: str) -> int | None:
        """Return the absolute configured count, or None when absent or a full pool."""
        entry = self.entry(source)
        if entry is None or entry.is_full_pool:
            return None
        return int(entry.value)

    @property
    def sample_total(self) -> int | None:
        """Return the configured total, or None when any entry is a full pool."""
        if any(entry.is_full_pool for entry in self.entries):
            return None
        return int(sum(int(entry.value) for entry in self.entries))

    def describe(self) -> str:
        return ", ".join(entry.describe() for entry in self.entries)


@dataclass(frozen=True)
class RunPointData:
    """Validated point metrics of one figure's selected runs."""

    points: pd.DataFrame
    target: str
    training_sizes: dict[str, int]


def read_train_on_design(
    client: MlflowClient,
    run_id: str,
    *,
    allowed_sources: Sequence[str] = PREDICTION_DATASETS,
) -> TrainOnDesign:
    """Read the authoritative training-source design from one run's config artifact.

    A fraction is either the JSON float 1.0, meaning the complete training pool of
    that source, or a positive JSON integer, meaning that many observations.
    Fractional subsampling in between is not a design this figure set compares, so
    it is rejected instead of being silently rounded.
    """
    payload = read_json_artifact(client, run_id, ARTIFACT_CONFIG)
    dataset = payload.get("dataset")
    if not isinstance(dataset, Mapping):
        raise TypeError(f"Run {run_id} has an invalid {ARTIFACT_CONFIG}; dataset must be an object")
    target = dataset.get("target")
    if not isinstance(target, str) or not target:
        raise ValueError(f"Run {run_id} has an invalid {ARTIFACT_CONFIG}; dataset.target must be a string")
    train_on = dataset.get("train_on")
    if not isinstance(train_on, list) or not 1 <= len(train_on) <= len(allowed_sources):
        count = len(train_on) if isinstance(train_on, list) else "non-list"
        raise ValueError(
            f"Run {run_id} dataset.train_on must contain between one and {len(allowed_sources)} entries; "
            f"found {count}"
        )

    entries = []
    seen: set[str] = set()
    for index, split in enumerate(train_on):
        if not isinstance(split, Mapping) or "dataset" not in split or "fraction" not in split:
            raise ValueError(f"Run {run_id} dataset.train_on[{index}] must contain dataset and fraction")
        source = split["dataset"]
        if source not in allowed_sources:
            raise ValueError(
                f"Run {run_id} dataset.train_on[{index}] has unknown source {source!r}; "
                f"expected one of {tuple(allowed_sources)}"
            )
        if source in seen:
            raise ValueError(f"Run {run_id} dataset.train_on contains duplicate source {source!r}")
        seen.add(source)
        entries.append(_train_on_entry(str(source), split["fraction"], run_id))

    entries.sort(key=lambda entry: allowed_sources.index(entry.source))
    retriever = dataset.get("custom_retriever")
    if retriever is not None and not isinstance(retriever, Mapping):
        raise TypeError(f"Run {run_id} has an invalid {ARTIFACT_CONFIG}; custom_retriever must be an object")
    return TrainOnDesign(run_id=str(run_id), target=target, entries=tuple(entries), retriever=retriever)


def _train_on_entry(source: str, fraction: object, run_id: str) -> TrainOnEntry:
    if isinstance(fraction, bool):
        raise TypeError(f"Run {run_id} dataset.train_on fraction for {source!r} must be a number; found bool")
    if isinstance(fraction, int):
        if fraction <= 0:
            raise ValueError(
                f"Run {run_id} dataset.train_on count for {source!r} must be positive; found {fraction}"
            )
        return TrainOnEntry(source=source, kind=SAMPLE_COUNT, value=fraction)
    if isinstance(fraction, float) and fraction == 1.0:
        return TrainOnEntry(source=source, kind=FULL_POOL, value=1.0)
    raise TypeError(
        f"Run {run_id} dataset.train_on fraction for {source!r} must be a positive JSON integer or the "
        f"JSON float 1.0; found {type(fraction).__name__} {fraction!r}"
    )


def read_json_artifact(client: MlflowClient, run_id: str, artifact_path: str) -> dict[str, object]:
    """Load one JSON artifact, reporting missing or invalid artifacts by run."""
    try:
        path = Path(client.download_artifacts(run_id, artifact_path))
    except MlflowException as error:
        raise ValueError(f"Run {run_id} has no readable {artifact_path} artifact") from error
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Run {run_id} has invalid JSON in {artifact_path}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"Run {run_id} {artifact_path} must contain a JSON object")
    return payload


def validated_point_rows(
    artifacts: PlotArtifacts,
    *,
    metrics: Sequence[str],
    datasets: Sequence[str],
    description: str,
) -> RunPointData:
    """Return the point rows of the selected runs, checked against the declared cohorts."""
    required = {
        "pipeline_mlflow_run_id",
        "scope",
        "statistic",
        "dataset",
        "target",
        "training_size",
        "model_name",
        "model_instance",
        *metrics,
    }
    require_columns(artifacts.metrics, required, f"{description} metrics")
    points = artifacts.metrics.loc[
        artifacts.metrics["scope"].eq("test") & artifacts.metrics["statistic"].eq("point")
    ].copy()
    if points.empty:
        raise ValueError("No scope='test', statistic='point' rows are available")
    points["pipeline_mlflow_run_id"] = points["pipeline_mlflow_run_id"].astype(str)
    expected_runs = {str(run_id) for run_id in artifacts.run_ids}
    observed_runs = set(points["pipeline_mlflow_run_id"])
    if observed_runs != expected_runs:
        raise ValueError(
            f"{description} data does not cover exactly the selected pipeline runs; "
            f"missing={sorted(expected_runs - observed_runs)}, extra={sorted(observed_runs - expected_runs)}"
        )

    target = single_value(points, "target", f"{description} point data")
    found = set(points["dataset"].dropna().astype(str))
    if found != set(datasets):
        raise ValueError(
            f"{description} runs must evaluate exactly the declared evaluation cohorts {sorted(datasets)}; "
            f"found {sorted(found)}"
        )

    training_sizes = {}
    for run_id, rows in points.groupby("pipeline_mlflow_run_id", sort=False):
        values = pd.to_numeric(rows["training_size"], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(f"Run {run_id} training_size must be a finite positive integer")
        unique = values.unique()
        if len(unique) != 1 or unique[0] <= 0 or float(unique[0]).is_integer() is False:
            raise ValueError(f"Run {run_id} training_size must record one positive integer; found {unique.tolist()}")
        training_sizes[str(run_id)] = int(unique[0])
    return RunPointData(points=points, target=target, training_sizes=training_sizes)


def validated_bootstrap_ids(
    artifacts: PlotArtifacts,
    points: pd.DataFrame,
    *,
    metrics: Sequence[str],
    target: str,
    datasets: Sequence[str],
    description: str,
) -> dict[str, tuple[float, ...]]:
    """Validate bootstrap coverage and return each run's aligned draw IDs.

    Every selected run has to record the same draw IDs for every model, metric,
    and evaluation cohort, and to cover exactly the cells its point data declares.
    Equal IDs alone never authorize pairing across unrelated runs; this only makes
    repeated runs within one experiment alignable.
    """
    required = {
        "pipeline_mlflow_run_id",
        "dataset",
        "target",
        "metric",
        "bootstrap_id",
        "score",
        "model_name",
        "model_instance",
    }
    require_columns(artifacts.bootstrap_scores, required, f"{description} bootstrap scores")
    rows = artifacts.bootstrap_scores.loc[artifacts.bootstrap_scores["metric"].isin(metrics)].copy()
    if rows.empty:
        raise ValueError(f"No bootstrap scores are available for metrics {list(metrics)}")
    for column in ("pipeline_mlflow_run_id", "model_instance", "dataset", "metric"):
        rows[column] = rows[column].astype(str)
    if single_value(rows, "target", f"{description} bootstrap data") != target:
        raise ValueError(f"{description} point and bootstrap data record different targets")
    found = set(rows["dataset"])
    if found != set(datasets):
        raise ValueError(
            f"{description} bootstrap data must contain only the declared evaluation cohorts "
            f"{sorted(datasets)}; found {sorted(found)}"
        )

    rows["bootstrap_id"] = pd.to_numeric(rows["bootstrap_id"], errors="coerce")
    rows["score"] = pd.to_numeric(rows["score"], errors="coerce")
    if (
        rows[["bootstrap_id", "score"]].isna().any().any()
        or not np.isfinite(rows[["bootstrap_id", "score"]]).all().all()
    ):
        raise ValueError(f"{description} bootstrap IDs and scores must be finite numeric values")

    duplicate_keys = ["pipeline_mlflow_run_id", "dataset", "metric", "bootstrap_id", "model_instance"]
    if rows.duplicated(duplicate_keys).any():
        raise ValueError(f"{description} bootstrap data contains duplicate run/dataset/metric/draw/model rows")

    point_models = {
        str(run_id): set(run_rows["model_instance"].astype(str))
        for run_id, run_rows in points.groupby("pipeline_mlflow_run_id", sort=False)
    }
    expected_cells = {
        (run_id, dataset, metric, model)
        for run_id, models in point_models.items()
        for dataset in datasets
        for metric in metrics
        for model in models
    }
    id_sets = rows.groupby(
        ["pipeline_mlflow_run_id", "dataset", "metric", "model_instance"],
        sort=False,
    )["bootstrap_id"].agg(lambda values: tuple(sorted(values.tolist())))
    observed_cells = set(id_sets.index.tolist())
    if observed_cells != expected_cells:
        raise ValueError(
            f"{description} bootstrap coverage is incomplete or contains unmatched cells; "
            f"missing={preview_cells(expected_cells - observed_cells)}, "
            f"extra={preview_cells(observed_cells - expected_cells)}"
        )

    ids_by_run = {}
    for run_id, run_id_sets in id_sets.groupby(level="pipeline_mlflow_run_id", sort=False):
        distinct = set(run_id_sets.tolist())
        if len(distinct) != 1:
            raise ValueError(f"Run {run_id} bootstrap IDs must match across every model/metric/evaluation cell")
        ids_by_run[str(run_id)] = next(iter(distinct))
    return ids_by_run


def require_aligned_bootstrap_ids(
    ids_by_run: Mapping[str, tuple[float, ...]],
    *,
    group_by_run: Mapping[str, str],
    description: str,
) -> None:
    """Require one shared bootstrap ID array inside each comparison group."""
    ids_by_group: dict[str, set[tuple[float, ...]]] = {}
    for run_id, group in group_by_run.items():
        ids_by_group.setdefault(str(group), set()).add(ids_by_run[str(run_id)])
    mismatched = sorted(group for group, values in ids_by_group.items() if len(values) != 1)
    if mismatched:
        raise ValueError(
            f"{description} must use aligned bootstrap IDs within each comparison group; "
            "mismatched groups: " + ", ".join(mismatched)
        )


def read_training_sample_seed(client: MlflowClient, run_id: str) -> int:
    """Return the training-side subset seed recorded for one run."""
    payload = read_json_artifact(client, run_id, ARTIFACT_CONFIG)
    random_states = payload.get("random_states")
    if not isinstance(random_states, Mapping) or "training_sample_seed" not in random_states:
        raise ValueError(
            f"Run {run_id} has an invalid {ARTIFACT_CONFIG}; random_states.training_sample_seed is required"
        )
    seed = random_states["training_sample_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"Run {run_id} random_states.training_sample_seed must be a JSON integer")
    return int(seed)


def select_artifact_runs(artifacts: PlotArtifacts, run_ids: Sequence[str]) -> PlotArtifacts:
    """Return the artifacts restricted to the given run IDs, in that order."""
    selected = {str(run_id) for run_id in run_ids}
    metric_ids = artifacts.metrics["pipeline_mlflow_run_id"].astype(str)
    bootstrap_ids = artifacts.bootstrap_scores["pipeline_mlflow_run_id"].astype(str)
    return PlotArtifacts(
        metrics=artifacts.metrics.loc[metric_ids.isin(selected)].copy(),
        bootstrap_scores=artifacts.bootstrap_scores.loc[bootstrap_ids.isin(selected)].copy(),
        experiment_name=artifacts.experiment_name,
        run_ids=tuple(str(run_id) for run_id in run_ids),
    )


def single_value(frame: pd.DataFrame, column: str, description: str) -> str:
    """Return the sole non-missing value of one column."""
    values = frame[column].dropna().astype(str).unique().tolist()
    if len(values) != 1:
        raise ValueError(f"Expected exactly one {column} in {description}; found {values}")
    return values[0]


def require_columns(frame: pd.DataFrame, required: set[str], description: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing {description} columns: {', '.join(missing)}")


def validate_model_identity(frame: pd.DataFrame) -> None:
    """Require every model instance to identify the same model in all rows."""
    names = frame[["model_instance", "model_name"]].drop_duplicates()
    conflicts = names.groupby("model_instance", sort=False)["model_name"].nunique()
    if conflicts.ne(1).any():
        bad = conflicts[conflicts.ne(1)].index.astype(str).tolist()
        raise ValueError("Model instances map to multiple model names: " + ", ".join(bad))


def preview_cells(cells: set[tuple[object, ...]], maximum: int = 3) -> str:
    if not cells:
        return "none"
    ordered = sorted(map(str, cells))
    suffix = f" (+{len(ordered) - maximum} more)" if len(ordered) > maximum else ""
    return ", ".join(ordered[:maximum]) + suffix
