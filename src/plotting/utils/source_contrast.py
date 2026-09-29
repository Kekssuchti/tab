"""Prepare fixed-cohort local-minus-external full-training contrasts."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException

from mlflow import MlflowClient
from src.mlflow.evaluation_data import DEFAULT_TRACKING_URI
from src.mlflow.tracking_contract import ARTIFACT_CONFIG, ARTIFACT_TEST_PREDICTIONS
from src.plotting.utils.aggregation import AggregatedEvaluation, aggregate_evaluation_runs
from src.plotting.utils.artifacts import PlotArtifacts
from src.plotting.utils.sample_size import full_training_count
from src.utils.prediction_tables import (
    PREDICTION_DATASETS,
    PREDICTION_MANIFEST_FILENAME,
    TEST_SET_ID_COLUMN,
    Y_TRUE_COLUMN,
)


@dataclass(frozen=True)
class PairingEvidence:
    """Recorded evidence that makes one center's bootstrap draws pairable."""

    evaluation_center: str
    cohort_fingerprint: str
    cohort_size: int
    prediction_schema_version: str
    prediction_dataset_order: tuple[str, ...]
    dataset_target: str
    dataset_random_state: int
    dataset_train_size: float
    evaluation_bootstrap_seed: int
    bootstrap_count: int
    bootstrap_id_first: int
    bootstrap_id_last: int


@dataclass(frozen=True)
class SourceContrastEvaluation:
    """Reciprocal full-data source contrasts on two fixed test cohorts."""

    performance: pd.DataFrame
    bootstrap_differences: pd.DataFrame
    model_metadata: pd.DataFrame
    model_instances: tuple[str, ...]
    evaluation_centers: tuple[str, ...]
    metrics: tuple[str, ...]
    target: str
    training_counts: dict[str, int]
    run_ids: dict[str, tuple[str, ...]]
    pairing_evidence: dict[str, PairingEvidence]
    ci_level: float

    @property
    def bootstrap_count(self) -> int:
        counts = {evidence.bootstrap_count for evidence in self.pairing_evidence.values()}
        if len(counts) != 1:
            raise ValueError(f"Evaluation centers disagree on bootstrap count: {sorted(counts)}")
        return next(iter(counts))

    def run_count(self, source: str) -> int:
        """Return the number of repeated complete-data runs for one source."""
        return len(self.run_ids[source])


@dataclass(frozen=True)
class _RunEvidence:
    run_id: str
    alignment_config: dict[str, object]
    configured_models: dict[str, tuple[str, str]]
    manifest_version: str
    dataset_order: tuple[str, ...]
    cohort_fingerprints: dict[str, str]
    cohort_tables: dict[str, pd.DataFrame]


def prepare_source_contrast(
    artifacts_by_source: Mapping[str, PlotArtifacts],
    *,
    metrics: Sequence[str],
    ci_level: float = 0.95,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> SourceContrastEvaluation:
    """Compute local-full minus external-full scores on each fixed cohort.

    Repeated complete-data runs are averaged within each source before the two
    source scores are subtracted. Interval endpoints are quantiles of paired
    bootstrap score differences, never differences between absolute-score
    interval endpoints.

    Pairing is accepted only after validating the recorded cohort fingerprint,
    exact ordered test IDs and labels, split and bootstrap settings, prediction
    dataset order, bootstrap count and ID arrays, and model roster across every
    selected run from both sources.
    """
    sources = tuple(PREDICTION_DATASETS)
    supplied_sources = set(artifacts_by_source)
    if supplied_sources != set(sources):
        raise ValueError(
            "Source contrasts require exactly the reciprocal MIMIC and TUDD inputs; "
            f"received {sorted(supplied_sources)}"
        )
    if not 0 < ci_level < 1:
        raise ValueError("ci_level must lie strictly between zero and one")

    selected_metrics = tuple(dict.fromkeys(metrics))
    if not selected_metrics:
        raise ValueError("metrics must contain at least one metric")

    aggregated = {
        source: aggregate_evaluation_runs(
            artifacts_by_source[source],
            metrics=selected_metrics,
            ci_level=ci_level,
        )
        for source in sources
    }
    target = _validate_aggregated_inputs(aggregated, sources, selected_metrics)

    client = MlflowClient(tracking_uri=tracking_uri)
    run_evidence = {
        source: tuple(_load_run_evidence(client, run_id) for run_id in artifacts_by_source[source].run_ids)
        for source in sources
    }
    _validate_model_rosters(aggregated, run_evidence, sources)

    pairing_evidence = {
        center: _validate_pairing(
            center,
            target,
            artifacts_by_source,
            aggregated,
            run_evidence,
            sources,
            selected_metrics,
        )
        for center in sources
    }
    performance, bootstrap_differences = _source_differences(
        aggregated,
        sources,
        selected_metrics,
        ci_level,
    )

    reference = aggregated[sources[0]]
    return SourceContrastEvaluation(
        performance=performance,
        bootstrap_differences=bootstrap_differences,
        model_metadata=reference.model_metadata.copy(),
        model_instances=reference.model_instances,
        evaluation_centers=sources,
        metrics=selected_metrics,
        target=target,
        training_counts={source: full_training_count(artifacts_by_source[source]) for source in sources},
        run_ids={source: aggregated[source].run_ids for source in sources},
        pairing_evidence=pairing_evidence,
        ci_level=ci_level,
    )


def _validate_aggregated_inputs(
    aggregated: Mapping[str, AggregatedEvaluation],
    sources: tuple[str, ...],
    metrics: tuple[str, ...],
) -> str:
    targets = {data.target for data in aggregated.values()}
    if len(targets) != 1:
        raise ValueError(f"Reciprocal inputs contain different targets: {sorted(targets)}")

    for source in sources:
        data = aggregated[source]
        if data.trained_on != source:
            raise ValueError(f"Input registered for {source!r} records trained_on={data.trained_on!r}")
        if set(data.datasets) != set(sources):
            raise ValueError(
                f"Input trained on {source!r} must evaluate the ordinary MIMIC and TUDD cohorts; "
                f"found {list(data.datasets)}. Retrieval cohorts are not mapped into F3."
            )
        if data.metrics != metrics:
            raise ValueError(f"Input trained on {source!r} returned metrics {data.metrics}, expected {metrics}")
    return next(iter(targets))


def _load_run_evidence(client: MlflowClient, run_id: str) -> _RunEvidence:
    config = _read_json_artifact(client, run_id, ARTIFACT_CONFIG)
    manifest_path = f"{ARTIFACT_TEST_PREDICTIONS}/{PREDICTION_MANIFEST_FILENAME}"
    manifest = _read_json_artifact(client, run_id, manifest_path)

    version = _required_value(manifest, ("version",), run_id, manifest_path)
    if str(version) == "1":
        dataset_order = tuple(PREDICTION_DATASETS)
    else:
        recorded_datasets = _required_value(manifest, ("datasets",), run_id, manifest_path)
        if not isinstance(recorded_datasets, list) or not all(
            isinstance(dataset, str) for dataset in recorded_datasets
        ):
            raise ValueError(f"Run {run_id} has an invalid {manifest_path} datasets list")
        dataset_order = tuple(recorded_datasets)
    if dataset_order != tuple(PREDICTION_DATASETS):
        raise ValueError(
            f"Run {run_id} records prediction dataset order {dataset_order}; F3 requires the ordinary "
            f"{tuple(PREDICTION_DATASETS)} cohorts and does not map retrieval fingerprints"
        )

    fingerprints = _required_value(manifest, ("cohort_fingerprints",), run_id, manifest_path)
    if not isinstance(fingerprints, dict):
        raise TypeError(f"Run {run_id} has invalid cohort_fingerprints in {manifest_path}")
    cohort_fingerprints = {}
    cohort_tables = {}
    for center in PREDICTION_DATASETS:
        fingerprint = fingerprints.get(center)
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError(f"Run {run_id} has no non-empty {center!r} cohort fingerprint in {manifest_path}")
        cohort_fingerprints[center] = fingerprint
        cohort_tables[center] = _read_cohort_table(client, run_id, center)

    alignment_config = {
        "dataset.target": _required_value(config, ("dataset", "target"), run_id, ARTIFACT_CONFIG),
        "dataset.random_state": _required_value(
            config,
            ("dataset", "random_state"),
            run_id,
            ARTIFACT_CONFIG,
        ),
        "dataset.train_size": _required_value(
            config,
            ("dataset", "train_size"),
            run_id,
            ARTIFACT_CONFIG,
        ),
        "random_states.evaluation_bootstrap_seed": _required_value(
            config,
            ("random_states", "evaluation_bootstrap_seed"),
            run_id,
            ARTIFACT_CONFIG,
        ),
    }
    return _RunEvidence(
        run_id=run_id,
        alignment_config=alignment_config,
        configured_models=_configured_models(config, run_id),
        manifest_version=str(version),
        dataset_order=dataset_order,
        cohort_fingerprints=cohort_fingerprints,
        cohort_tables=cohort_tables,
    )


def _read_json_artifact(client: MlflowClient, run_id: str, artifact_path: str) -> dict[str, object]:
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


def _read_cohort_table(client: MlflowClient, run_id: str, center: str) -> pd.DataFrame:
    artifact_path = f"{ARTIFACT_TEST_PREDICTIONS}/{center}.csv"
    try:
        path = Path(client.download_artifacts(run_id, artifact_path))
    except MlflowException as error:
        raise ValueError(f"Run {run_id} has no readable {artifact_path} artifact") from error
    try:
        table = pd.read_csv(path, usecols=[TEST_SET_ID_COLUMN, Y_TRUE_COLUMN])
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Run {run_id} {artifact_path} must contain {TEST_SET_ID_COLUMN} and {Y_TRUE_COLUMN}"
        ) from error
    if table.empty:
        raise ValueError(f"Run {run_id} {artifact_path} contains no held-out observations")
    if table.isna().any().any() or table[TEST_SET_ID_COLUMN].duplicated().any():
        raise ValueError(f"Run {run_id} {artifact_path} has missing or duplicate cohort identity values")
    return table


def _required_value(
    payload: Mapping[str, object],
    path: tuple[str, ...],
    run_id: str,
    artifact_path: str,
) -> object:
    value: object = payload
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            dotted = ".".join(path)
            raise ValueError(f"Run {run_id} {artifact_path} does not record required setting {dotted}")
        value = value[part]
    return value


def _configured_models(config: Mapping[str, object], run_id: str) -> dict[str, tuple[str, str]]:
    training = _required_value(config, ("training",), run_id, ARTIFACT_CONFIG)
    if not isinstance(training, list) or not training:
        raise ValueError(f"Run {run_id} {ARTIFACT_CONFIG} must contain a non-empty training list")
    names = []
    for index, model in enumerate(training):
        if not isinstance(model, dict) or not isinstance(model.get("name"), str):
            raise TypeError(f"Run {run_id} has an invalid training model at index {index}")
        names.append(str(model["name"]))

    counts = Counter(names)
    seen: Counter[str] = Counter()
    configured = {}
    for name, model in zip(names, training, strict=True):
        index = seen[name]
        seen[name] += 1
        instance = name if counts[name] == 1 else f"{name}__{index}"
        configured[instance] = (name, json.dumps(model, sort_keys=True, separators=(",", ":")))
    return configured


def _validate_model_rosters(
    aggregated: Mapping[str, AggregatedEvaluation],
    evidence: Mapping[str, tuple[_RunEvidence, ...]],
    sources: tuple[str, ...],
) -> None:
    reference_source = sources[0]
    reference_run = evidence[reference_source][0]
    reference_configured = reference_run.configured_models
    for source in sources:
        for run in evidence[source]:
            if run.configured_models != reference_configured:
                missing, unexpected, changed = _configured_roster_difference(
                    reference_configured,
                    run.configured_models,
                )
                raise ValueError(
                    "Reciprocal F3 inputs must configure the same unambiguous model instances; "
                    f"run {run.run_id} trained on {source!r} differs from run {reference_run.run_id}: "
                    f"missing={missing}, unexpected={unexpected}, changed_instance_configs={changed}"
                )

        actual = {
            str(row.model_instance): str(row.model_name) for row in aggregated[source].model_metadata.itertuples()
        }
        expected = {instance: values[0] for instance, values in reference_configured.items()}
        if actual != expected:
            missing = sorted(set(expected) - set(actual))
            unexpected = sorted(set(actual) - set(expected))
            renamed = sorted(
                instance for instance in set(actual) & set(expected) if actual[instance] != expected[instance]
            )
            raise ValueError(
                f"Complete-data runs trained on {source!r} exclude or lack configured models; "
                f"missing={missing}, unexpected={unexpected}, renamed_instances={renamed}. "
                "F3 does not compare unequal or filtered rosters."
            )

    left = {str(row.model_instance): str(row.model_name) for row in aggregated[sources[0]].model_metadata.itertuples()}
    right = {str(row.model_instance): str(row.model_name) for row in aggregated[sources[1]].model_metadata.itertuples()}
    if left != right:
        raise ValueError("Reciprocal F3 inputs produced different model instance rosters")


def _configured_roster_difference(
    reference: Mapping[str, tuple[str, str]],
    candidate: Mapping[str, tuple[str, str]],
) -> tuple[list[str], list[str], list[str]]:
    shared = set(reference) & set(candidate)
    return (
        sorted(set(reference) - set(candidate)),
        sorted(set(candidate) - set(reference)),
        sorted(instance for instance in shared if reference[instance] != candidate[instance]),
    )


def _validate_pairing(
    center: str,
    target: str,
    artifacts: Mapping[str, PlotArtifacts],
    aggregated: Mapping[str, AggregatedEvaluation],
    evidence: Mapping[str, tuple[_RunEvidence, ...]],
    sources: tuple[str, ...],
    metrics: tuple[str, ...],
) -> PairingEvidence:
    runs = [(source, run) for source in sources for run in evidence[source]]
    reference_source, reference = runs[0]

    for key in reference.alignment_config:
        values = {json.dumps(run.alignment_config[key], sort_keys=True) for _, run in runs}
        if len(values) != 1:
            details = ", ".join(f"{source}/{run.run_id}={run.alignment_config[key]!r}" for source, run in runs)
            raise ValueError(
                f"Cannot pair {center!r} bootstrap draws: recorded {key} differs across sources/runs ({details})"
            )
    if reference.alignment_config["dataset.target"] != target:
        raise ValueError(
            f"Cannot pair {center!r}: config target {reference.alignment_config['dataset.target']!r} "
            f"does not match artifact target {target!r}"
        )

    fingerprints = {run.cohort_fingerprints[center] for _, run in runs}
    if len(fingerprints) != 1:
        details = ", ".join(f"{source}/{run.run_id}={run.cohort_fingerprints[center]}" for source, run in runs)
        raise ValueError(
            f"Cannot pair {center!r}: test_predictions/manifest.json cohort fingerprints differ ({details})"
        )

    manifest_versions = {run.manifest_version for _, run in runs}
    dataset_orders = {run.dataset_order for _, run in runs}
    if len(manifest_versions) != 1 or len(dataset_orders) != 1:
        raise ValueError(f"Cannot pair {center!r}: prediction manifest schema/order differs across selected runs")

    reference_table = reference.cohort_tables[center]
    for source, run in runs[1:]:
        if not run.cohort_tables[center].equals(reference_table):
            raise ValueError(
                f"Cannot pair {center!r}: ordered held-out IDs or labels differ between "
                f"{reference_source}/{reference.run_id} and {source}/{run.run_id}"
            )

    recorded_bootstrap_count = _recorded_bootstrap_count(center, artifacts, sources)
    reference_ids: np.ndarray | None = None
    for source in sources:
        for metric in metrics:
            ids = aggregated[source].scores(center, metric).index.to_numpy(copy=True)
            if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
                raise ValueError(f"Cannot pair {center!r}: {source!r}/{metric!r} bootstrap IDs are not a unique array")
            if reference_ids is None:
                reference_ids = ids
            elif not np.array_equal(ids, reference_ids):
                raise ValueError(f"Cannot pair {center!r}: bootstrap ID arrays differ across sources or metrics")
    assert reference_ids is not None
    if len(reference_ids) != recorded_bootstrap_count:
        raise ValueError(
            f"Cannot pair {center!r}: config metrics record {recorded_bootstrap_count} bootstrap draws, "
            f"but the score artifact contains {len(reference_ids)} IDs"
        )

    return PairingEvidence(
        evaluation_center=center,
        cohort_fingerprint=next(iter(fingerprints)),
        cohort_size=len(reference_table),
        prediction_schema_version=next(iter(manifest_versions)),
        prediction_dataset_order=next(iter(dataset_orders)),
        dataset_target=str(reference.alignment_config["dataset.target"]),
        dataset_random_state=int(reference.alignment_config["dataset.random_state"]),
        dataset_train_size=float(reference.alignment_config["dataset.train_size"]),
        evaluation_bootstrap_seed=int(reference.alignment_config["random_states.evaluation_bootstrap_seed"]),
        bootstrap_count=recorded_bootstrap_count,
        bootstrap_id_first=int(reference_ids[0]),
        bootstrap_id_last=int(reference_ids[-1]),
    )


def _recorded_bootstrap_count(
    center: str,
    artifacts: Mapping[str, PlotArtifacts],
    sources: tuple[str, ...],
) -> int:
    values = []
    for source in sources:
        frame = artifacts[source].metrics
        if "n_bootstrap" not in frame.columns:
            raise ValueError(f"Cannot pair {center!r}: {source!r} prediction metrics do not record n_bootstrap")
        points = frame.loc[frame["scope"].eq("test") & frame["statistic"].eq("point") & frame["dataset"].eq(center)]
        for run_id, rows in points.groupby("pipeline_mlflow_run_id", sort=False):
            counts = pd.to_numeric(rows["n_bootstrap"], errors="coerce").dropna().unique()
            if len(counts) != 1 or counts[0] <= 0 or counts[0] % 1:
                raise ValueError(f"Cannot pair {center!r}: run {run_id} has invalid or ambiguous n_bootstrap metadata")
            values.append(int(counts[0]))
    if not values or len(set(values)) != 1:
        raise ValueError(f"Cannot pair {center!r}: recorded bootstrap counts differ or are absent ({values})")
    return values[0]


def _source_differences(
    aggregated: Mapping[str, AggregatedEvaluation],
    sources: tuple[str, ...],
    metrics: tuple[str, ...],
    ci_level: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = aggregated[sources[0]].model_metadata.set_index("model_instance")["model_name"]
    instances = aggregated[sources[0]].model_instances
    alpha = (1.0 - ci_level) / 2.0
    performance_rows = []
    bootstrap_frames = []

    for center in sources:
        external_source = next(source for source in sources if source != center)
        local = aggregated[center]
        external = aggregated[external_source]
        for metric in metrics:
            local_points = _point_scores(local, center, metric, instances)
            external_points = _point_scores(external, center, metric, instances)
            differences = local.scores(center, metric) - external.scores(center, metric)
            if differences.isna().any().any():
                raise ValueError(f"Paired source difference produced missing scores for {center!r}/{metric!r}")
            bootstrap_frames.append(
                differences.rename_axis(columns="model_instance")
                .reset_index()
                .melt(
                    id_vars="bootstrap_id",
                    var_name="model_instance",
                    value_name="difference",
                )
                .assign(evaluation_center=center, metric=metric)
            )
            for instance in instances:
                lower, upper = differences[instance].quantile([alpha, 1.0 - alpha])
                performance_rows.append(
                    {
                        "model_name": str(names.loc[instance]),
                        "model_instance": instance,
                        "evaluation_center": center,
                        "metric": metric,
                        "estimate": float(local_points.loc[instance] - external_points.loc[instance]),
                        "lower": float(lower),
                        "upper": float(upper),
                    }
                )

    return pd.DataFrame(performance_rows), pd.concat(bootstrap_frames, ignore_index=True)


def _point_scores(
    data: AggregatedEvaluation,
    center: str,
    metric: str,
    instances: tuple[str, ...],
) -> pd.Series:
    rows = data.performance.loc[data.performance["dataset"].eq(center) & data.performance["metric"].eq(metric)]
    if rows["model_instance"].duplicated().any():
        raise ValueError(f"Point estimates are ambiguous for {center!r}/{metric!r}")
    indexed = rows.set_index("model_instance")["estimate"]
    missing = sorted(set(instances) - set(indexed.index.astype(str)))
    if missing:
        raise ValueError(f"Point estimates are missing model instances for {center!r}/{metric!r}: {missing}")
    return indexed.reindex(instances)
