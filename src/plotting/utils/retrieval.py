"""Read retrieval designs and prepare retrieval-budget curves (F10).

Retrieval is a target-conditioned, transductive batch setting: training rows are
selected with reference to the covariates of one specific target batch. Two
things therefore have to be verified before any budget curve is drawn.

* Every compared run has to select from the same eligible candidate pool with the
  same split and bootstrap settings.
* Every compared run has to be evaluated on the exact same target batch. The
  recorded prediction manifest identifies the cohort; the ordered held-out IDs
  and labels in the retriever table identify the batch itself, because the
  retriever cohort fingerprint does not depend on the batch sample.

The unrestricted reference trains on the entire eligible candidate pool. It is
registered as its own experiment (`retrieval_unrestricted` in
`src.plotting.experiments`) because ordinary full-held-out-set results are not
equivalent to a matched-batch evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from mlflow.exceptions import MlflowException

from mlflow import MlflowClient
from src.mlflow.evaluation_data import DEFAULT_TRACKING_URI
from src.mlflow.tracking_contract import ARTIFACT_CONFIG, ARTIFACT_TEST_PREDICTIONS
from src.plotting.utils.aggregation import aggregate_evaluation_runs
from src.plotting.utils.artifacts import PlotArtifacts
from src.plotting.utils.grouped import GroupedEvaluation, aggregate_runs_by_setting, require_grouped_coverage
from src.plotting.utils.runs import (
    IncompleteExperimentError,
    read_json_artifact,
    read_train_on_design,
    require_aligned_bootstrap_ids,
    select_artifact_runs,
    validated_bootstrap_ids,
    validated_point_rows,
)
from src.utils.prediction_tables import (
    PREDICTION_DATASETS,
    PREDICTION_MANIFEST_FILENAME,
    RETRIEVER_DATASET,
    TEST_SET_ID_COLUMN,
    Y_TRUE_COLUMN,
)

RANDOM_STRATEGY = "random"
RETRIEVER_COHORTS = (RETRIEVER_DATASET,)
SELECTION_STRATEGIES = (RANDOM_STRATEGY, "knn", "knn-diverse")


@dataclass(frozen=True)
class RetrieverDesign:
    """The retrieval design recorded in one run's `config.json`."""

    run_id: str
    target: str
    candidate_source: str
    batch_center: str
    batch_size: int
    selected_count: int
    strategy: str
    distance_metric: str
    diversity_clusters: int
    diversity_pool_multiplier: float
    test_sample_seed: int
    training_sample_seed: int
    model_training_seed: int
    setting: str
    setting_label: str
    setting_order: tuple[object, ...]

    @property
    def batch(self) -> tuple[int, int]:
        return (self.batch_size, self.test_sample_seed)

    @property
    def is_reference_strategy(self) -> bool:
        return self.strategy == RANDOM_STRATEGY


@dataclass(frozen=True)
class BatchIdentity:
    """The exact target batch one run was evaluated on."""

    run_id: str
    dataset: str
    cohort_fingerprint: str
    cohort_size: int
    ordered_ids: tuple[str, ...]
    ordered_labels: tuple[int, ...]

    @property
    def identity(self) -> tuple[object, ...]:
        return (self.cohort_fingerprint, self.cohort_size, self.ordered_ids, self.ordered_labels)

    def describe(self) -> str:
        return (
            f"{self.cohort_size:,} observations, fingerprint {self.cohort_fingerprint[:12]}, "
            f"first ID {self.ordered_ids[0][:16] if self.ordered_ids else 'none'}"
        )


@dataclass(frozen=True)
class RetrievalBatchView:
    """Budget curves for one target batch and the matched unrestricted reference."""

    grouped: GroupedEvaluation
    conditions: pd.DataFrame
    unrestricted: pd.DataFrame | None
    target: str
    candidate_source: str
    batch_center: str
    batch_size: int
    test_sample_seed: int
    batch_identity: BatchIdentity
    unrestricted_count: int | None

    @property
    def performance(self) -> pd.DataFrame:
        return self.grouped.performance.merge(self.conditions, on="setting", how="left", validate="many_to_one")

    @property
    def selected_counts(self) -> tuple[int, ...]:
        return tuple(sorted(int(value) for value in self.conditions["selected_count"].unique()))

    @property
    def settings(self) -> tuple[str, ...]:
        return self.grouped.settings

    @property
    def strategies(self) -> tuple[str, ...]:
        """Return each declared retrieval strategy once, in canonical order."""
        ordered = self.conditions.sort_values("strategy_order", kind="stable")
        return tuple(dict.fromkeys(ordered["strategy"]))

    def strategy_labels(self) -> dict[str, str]:
        """Return the display label of every declared retrieval strategy."""
        return dict(zip(self.conditions["strategy"], self.conditions["strategy_label"], strict=True))

    @property
    def model_metadata(self) -> pd.DataFrame:
        return self.grouped.model_metadata

    @property
    def model_instances(self) -> tuple[str, ...]:
        return self.grouped.model_instances

    @property
    def metrics(self) -> tuple[str, ...]:
        return self.grouped.metrics

    @property
    def run_counts(self) -> dict[str, int]:
        return self.grouped.run_counts

    @property
    def bootstrap_count(self) -> int:
        return self.grouped.bootstrap_count

    @property
    def ci_level(self) -> float:
        return self.grouped.ci_level


@dataclass(frozen=True)
class RetrievalBudgetEvaluation:
    """One view per target batch; batches are never pooled into one curve."""

    batch_views: tuple[RetrievalBatchView, ...]
    target: str
    metrics: tuple[str, ...]
    candidate_source: str
    batch_center: str
    unrestricted_run_ids: tuple[str, ...]

    def for_batch(self, batch_center: str, batch_size: int, test_sample_seed: int) -> RetrievalBatchView:
        matches = tuple(
            view
            for view in self.batch_views
            if view.batch_center == batch_center
            and view.batch_size == batch_size
            and view.test_sample_seed == test_sample_seed
        )
        if len(matches) != 1:
            raise ValueError(
                f"Expected one retrieval view for {batch_center}/{batch_size}/{test_sample_seed}; "
                f"found {len(matches)}"
            )
        return matches[0]


def read_retriever_design(client: MlflowClient, run_id: str) -> RetrieverDesign:
    """Read the retrieval design from one run's authoritative config artifact."""
    payload = read_json_artifact(client, run_id, ARTIFACT_CONFIG)
    dataset = payload.get("dataset")
    if not isinstance(dataset, Mapping):
        raise TypeError(f"Run {run_id} has an invalid {ARTIFACT_CONFIG}; dataset must be an object")
    target = dataset.get("target")
    if not isinstance(target, str) or not target:
        raise ValueError(f"Run {run_id} has an invalid {ARTIFACT_CONFIG}; dataset.target must be a string")
    retriever = dataset.get("custom_retriever")
    if not isinstance(retriever, Mapping):
        raise TypeError(f"Run {run_id} does not configure custom_retriever, so it is not a retrieval run")

    design = read_train_on_design(client, run_id)
    if len(design.entries) != 1 or not design.entries[0].is_full_pool:
        raise ValueError(
            f"Run {run_id} trains on {design.describe()}; retrieval requires exactly one candidate-pool source "
            "recorded with the full-pool fraction 1.0, so that the unrestricted reference has a defined pool"
        )
    candidate_source = design.entries[0].source

    selected_count = retriever.get("train_size")
    if isinstance(selected_count, bool) or not isinstance(selected_count, int) or selected_count <= 0:
        raise ValueError(f"Run {run_id} custom_retriever.train_size must be a positive JSON integer")

    test_on = retriever.get("test_on")
    if not isinstance(test_on, list) or len(test_on) != 1:
        raise ValueError(f"Run {run_id} custom_retriever.test_on must contain exactly one target batch")
    batch = test_on[0]
    if not isinstance(batch, Mapping) or "dataset" not in batch or "fraction" not in batch:
        raise ValueError(f"Run {run_id} custom_retriever.test_on[0] must contain dataset and fraction")
    batch_center = batch["dataset"]
    batch_size = batch["fraction"]
    if batch_center not in PREDICTION_DATASETS:
        raise ValueError(f"Run {run_id} custom_retriever.test_on[0] has unknown center {batch_center!r}")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise TypeError(
            f"Run {run_id} custom_retriever.test_on[0].fraction must be a positive JSON integer batch size"
        )

    strategy = str(retriever.get("selection_strategy"))
    if strategy not in SELECTION_STRATEGIES:
        raise ValueError(
            f"Run {run_id} uses retrieval strategy {strategy!r}; declared strategies are {SELECTION_STRATEGIES}"
        )
    metric = str(retriever.get("distance_metric", "euclidean"))
    clusters = int(retriever.get("diversity_clusters", 0))
    pool = float(retriever.get("diversity_pool_multiplier", 0.0))
    setting, setting_label, setting_order = retriever_setting_identity(strategy, metric, clusters, pool)

    random_states = payload.get("random_states")
    if not isinstance(random_states, Mapping):
        raise TypeError(f"Run {run_id} has an invalid {ARTIFACT_CONFIG}; random_states must be an object")
    return RetrieverDesign(
        run_id=str(run_id),
        target=target,
        candidate_source=candidate_source,
        batch_center=str(batch_center),
        batch_size=int(batch_size),
        selected_count=int(selected_count),
        strategy=strategy,
        distance_metric=metric,
        diversity_clusters=clusters,
        diversity_pool_multiplier=pool,
        test_sample_seed=_required_int(retriever, "test_sample_seed", run_id, "custom_retriever"),
        training_sample_seed=_required_int(random_states, "training_sample_seed", run_id, "random_states"),
        model_training_seed=_required_int(random_states, "model_training_seed", run_id, "random_states"),
        setting=setting,
        setting_label=setting_label,
        setting_order=setting_order,
    )


def retriever_setting_identity(
    strategy: str,
    distance_metric: str,
    clusters: int,
    pool_multiplier: float,
) -> tuple[str, str, tuple[object, ...]]:
    """Return the stable identity, label, and sort key of one retrieval configuration."""
    if strategy == RANDOM_STRATEGY:
        return RANDOM_STRATEGY, "Random", (-1,)
    if strategy == "knn":
        setting = f"knn:{distance_metric}"
        label = f"KNN: {distance_metric.title()}"
        metric_order = {"euclidean": 0, "manhattan": 1}.get(distance_metric, 99)
        return setting, label, (0, metric_order, distance_metric)
    if strategy == "knn-diverse":
        pool_label = f"{pool_multiplier:g}"
        setting = f"knn-diverse:{clusters}:{pool_label}"
        label = f"Diverse: k={clusters}, pool={pool_label}x"
        return setting, label, (1, clusters, pool_multiplier)
    raise ValueError(f"Unsupported retriever selection_strategy {strategy!r}")


def read_alignment_config(client: MlflowClient, run_id: str) -> dict[str, object]:
    """Return the recorded settings that must match before retrieval runs are comparable."""
    payload = read_json_artifact(client, run_id, ARTIFACT_CONFIG)
    dataset = payload.get("dataset")
    random_states = payload.get("random_states")
    if not isinstance(dataset, Mapping) or not isinstance(random_states, Mapping):
        raise TypeError(f"Run {run_id} has an invalid {ARTIFACT_CONFIG}; dataset and random_states are required")
    retriever = dataset.get("custom_retriever")
    if not isinstance(retriever, Mapping):
        raise TypeError(f"Run {run_id} does not configure custom_retriever, so it is not a retrieval run")
    # The target batch itself is deliberately absent: budget runs of different
    # batches are never pooled, and each batch is matched separately through its
    # recorded prediction cohort.
    return {
        "dataset.target": dataset.get("target"),
        "dataset.random_state": dataset.get("random_state"),
        "dataset.train_size": dataset.get("train_size"),
        "dataset.train_on": dataset.get("train_on"),
        "random_states.evaluation_bootstrap_seed": random_states.get("evaluation_bootstrap_seed"),
    }


def read_batch_identity(client: MlflowClient, run_id: str, dataset: str = RETRIEVER_DATASET) -> BatchIdentity:
    """Read the exact target batch one run was evaluated on."""
    manifest_path = f"{ARTIFACT_TEST_PREDICTIONS}/{PREDICTION_MANIFEST_FILENAME}"
    manifest = read_json_artifact(client, run_id, manifest_path)
    fingerprints = manifest.get("cohort_fingerprints")
    if not isinstance(fingerprints, Mapping):
        raise TypeError(f"Run {run_id} has an invalid {manifest_path}; cohort_fingerprints must be an object")
    fingerprint = fingerprints.get(dataset)
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(f"Run {run_id} has no {dataset!r} cohort fingerprint in {manifest_path}")
    table = _read_batch_table(client, run_id, dataset)
    return BatchIdentity(
        run_id=str(run_id),
        dataset=dataset,
        cohort_fingerprint=fingerprint,
        cohort_size=len(table),
        ordered_ids=tuple(table[TEST_SET_ID_COLUMN].astype(str)),
        ordered_labels=tuple(int(value) for value in table[Y_TRUE_COLUMN]),
    )


def _read_batch_table(client: MlflowClient, run_id: str, dataset: str) -> pd.DataFrame:
    artifact_path = f"{ARTIFACT_TEST_PREDICTIONS}/{dataset}.csv"
    try:
        path = Path(client.download_artifacts(run_id, artifact_path))
    except MlflowException as error:
        raise ValueError(f"Run {run_id} has no readable {artifact_path} artifact") from error
    table = pd.read_csv(path, usecols=[TEST_SET_ID_COLUMN, Y_TRUE_COLUMN])
    if table.empty:
        raise ValueError(f"Run {run_id} {artifact_path} contains no target-batch observations")
    if table.isna().any().any() or table[TEST_SET_ID_COLUMN].duplicated().any():
        raise ValueError(f"Run {run_id} {artifact_path} has missing or duplicate batch identity values")
    if not set(table[Y_TRUE_COLUMN].unique()) <= {0, 1}:
        raise ValueError(f"Run {run_id} {artifact_path} has non-binary labels")
    return table


def require_matched_batches(identities: Sequence[BatchIdentity], description: str) -> BatchIdentity:
    """Require one identical ordered target batch across the given runs."""
    reference = identities[0]
    for identity in identities[1:]:
        if identity.identity != reference.identity:
            raise ValueError(
                f"{description} must be evaluated on the same target batch; run {identity.run_id} "
                f"({identity.describe()}) differs from run {reference.run_id} ({reference.describe()})"
            )
    return reference


def require_matched_alignments(configs: Mapping[str, Mapping[str, object]], description: str) -> None:
    """Require identical recorded split, pool, batch, and bootstrap settings."""
    reference_run = next(iter(configs))
    reference = configs[reference_run]
    for run_id, config in configs.items():
        for key, value in reference.items():
            if config.get(key) != value:
                raise ValueError(
                    f"{description} must share the recorded setting {key}; run {run_id} records "
                    f"{config.get(key)!r} while run {reference_run} records {value!r}"
                )


def prepare_retrieval_budget_evaluation(
    budget_artifacts: PlotArtifacts,
    *,
    unrestricted_artifacts: PlotArtifacts | None,
    metrics: Sequence[str],
    ci_level: float = 0.95,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> RetrievalBudgetEvaluation:
    """Prepare one budget-curve view per target batch, with its unrestricted reference.

    The reference is optional: when the unrestricted input is not registered the
    views report it as unavailable instead of borrowing an ordinary full-data
    result whose evaluation population differs.
    """
    selected_metrics = tuple(dict.fromkeys(metrics))
    if not selected_metrics:
        raise ValueError("metrics must contain at least one metric")
    if not 0 < ci_level < 1:
        raise ValueError("ci_level must lie strictly between zero and one")
    if not budget_artifacts.run_ids:
        raise ValueError("Retrieval budget preparation requires at least one pipeline run")

    point_data = validated_point_rows(
        budget_artifacts,
        metrics=selected_metrics,
        datasets=RETRIEVER_COHORTS,
        description="Retrieval budget",
    )
    bootstrap_ids = validated_bootstrap_ids(
        budget_artifacts,
        point_data.points,
        metrics=selected_metrics,
        target=point_data.target,
        datasets=RETRIEVER_COHORTS,
        description="Retrieval budget",
    )

    client = MlflowClient(tracking_uri=tracking_uri)
    designs = tuple(read_retriever_design(client, str(run_id)) for run_id in budget_artifacts.run_ids)
    _require_uniform({design.target for design in designs}, point_data.target, "target")
    candidate_source = _require_uniform(
        {design.candidate_source for design in designs}, None, "candidate-pool source"
    )
    batch_center = _require_uniform({design.batch_center for design in designs}, None, "target-batch center")
    for design in designs:
        _require_realized_count(point_data.training_sizes, design)
    require_matched_alignments(
        {design.run_id: read_alignment_config(client, design.run_id) for design in designs},
        description="Retrieval budget runs",
    )
    require_aligned_bootstrap_ids(
        bootstrap_ids,
        group_by_run={design.run_id: "retrieval-budget" for design in designs},
        description="Retrieval budget runs",
    )

    unrestricted_designs = ()
    unrestricted_count: int | None = None
    unrestricted_run_ids: tuple[str, ...] = ()
    if unrestricted_artifacts is not None:
        unrestricted_designs, unrestricted_count = _validate_unrestricted(
            client,
            unrestricted_artifacts,
            target=point_data.target,
            candidate_source=candidate_source,
            batch_center=batch_center,
            metrics=selected_metrics,
        )
        unrestricted_run_ids = tuple(design.run_id for design in unrestricted_designs)

    batch_keys = sorted({(design.batch_size, design.test_sample_seed) for design in designs})
    views = []
    for batch_key in batch_keys:
        group = tuple(design for design in designs if design.batch == batch_key)
        views.append(
            _prepare_batch_view(
                budget_artifacts,
                group,
                target=point_data.target,
                candidate_source=candidate_source,
                batch_center=batch_center,
                metrics=selected_metrics,
                ci_level=ci_level,
                unrestricted_designs=tuple(d for d in unrestricted_designs if d.batch == batch_key),
                unrestricted_count=unrestricted_count,
                unrestricted_artifacts=unrestricted_artifacts,
                client=client,
            )
        )
    return RetrievalBudgetEvaluation(
        batch_views=tuple(views),
        target=point_data.target,
        metrics=selected_metrics,
        candidate_source=candidate_source,
        batch_center=batch_center,
        unrestricted_run_ids=unrestricted_run_ids,
    )


def _prepare_batch_view(
    budget_artifacts: PlotArtifacts,
    group: Sequence[RetrieverDesign],
    *,
    target: str,
    candidate_source: str,
    batch_center: str,
    metrics: tuple[str, ...],
    ci_level: float,
    unrestricted_designs: Sequence[RetrieverDesign],
    unrestricted_count: int | None,
    unrestricted_artifacts: PlotArtifacts | None,
    client: MlflowClient,
) -> RetrievalBatchView:
    batch_size, test_sample_seed = group[0].batch
    counts = sorted({design.selected_count for design in group})
    if len(counts) < 2:
        raise IncompleteExperimentError(
            f"Target batch {batch_center}/{batch_size}/seed {test_sample_seed} measures only "
            f"{counts[0]:,} selected training observations. A budget curve needs several measured budgets; "
            "with a single budget the paired comparison in retriever_comparison (F9) is the right figure."
        )
    strategies = {design.setting for design in group}
    for count in counts:
        measured = {design.setting for design in group if design.selected_count == count}
        if measured != strategies:
            raise ValueError(
                f"Target batch {batch_center}/{batch_size}/seed {test_sample_seed} does not measure every "
                f"retrieval strategy at {count:,} selected observations; missing="
                f"{sorted(strategies - measured)}, extra={sorted(measured - strategies)}"
            )

    setting_by_run = {design.run_id: f"{design.setting}|n{design.selected_count}" for design in group}
    order_key = {
        design.run_id: (*design.setting_order, design.selected_count) for design in group
    }
    ordered = sorted(group, key=lambda design: order_key[design.run_id])
    setting_order = tuple(dict.fromkeys(setting_by_run[design.run_id] for design in ordered))
    selected = select_artifact_runs(budget_artifacts, [design.run_id for design in ordered])
    grouped = aggregate_runs_by_setting(
        selected,
        setting_by_run,
        setting_order=setting_order,
        metrics=metrics,
        ci_level=ci_level,
    )
    require_grouped_coverage(grouped, datasets=RETRIEVER_COHORTS, description="Retrieval budget")

    conditions = (
        pd.DataFrame(
            [
                {
                    "setting": setting_by_run[design.run_id],
                    "strategy": design.setting,
                    "strategy_label": design.setting_label,
                    "strategy_order": design.setting_order,
                    "selected_count": design.selected_count,
                    "run_count": grouped.run_counts[setting_by_run[design.run_id]],
                }
                for design in ordered
            ]
        )
        .drop_duplicates("setting")
        .reset_index(drop=True)
    )

    batch_identity = require_matched_batches(
        [read_batch_identity(client, design.run_id) for design in group],
        description=f"Retrieval budget runs of target batch {batch_center}/{batch_size}",
    )

    unrestricted = None
    if unrestricted_designs:
        if unrestricted_count is None:
            raise RuntimeError("Unrestricted designs require a recorded candidate-pool count")
        if unrestricted_count <= max(counts):
            raise ValueError(
                f"The unrestricted reference records {unrestricted_count:,} training observations, which is not "
                f"larger than the largest selected budget {max(counts):,}"
            )
        reference_identity = require_matched_batches(
            [read_batch_identity(client, design.run_id) for design in unrestricted_designs],
            description=f"Unrestricted reference runs of target batch {batch_center}/{batch_size}",
        )
        if reference_identity.identity != batch_identity.identity:
            raise ValueError(
                "The unrestricted reference must be evaluated on the same target batch as the budget runs; "
                f"it records {reference_identity.describe()} while the budget runs record "
                f"{batch_identity.describe()}"
            )
        assert unrestricted_artifacts is not None
        unrestricted = _aggregate_unrestricted(
            unrestricted_artifacts,
            unrestricted_designs,
            metrics=metrics,
            ci_level=ci_level,
            model_instances=grouped.model_instances,
            training_count=unrestricted_count,
        )

    return RetrievalBatchView(
        grouped=grouped,
        conditions=conditions,
        unrestricted=unrestricted,
        target=target,
        candidate_source=candidate_source,
        batch_center=batch_center,
        batch_size=batch_size,
        test_sample_seed=test_sample_seed,
        batch_identity=batch_identity,
        unrestricted_count=unrestricted_count,
    )


def _aggregate_unrestricted(
    artifacts: PlotArtifacts,
    designs: Sequence[RetrieverDesign],
    *,
    metrics: tuple[str, ...],
    ci_level: float,
    model_instances: Sequence[str],
    training_count: int,
) -> pd.DataFrame:
    selected = select_artifact_runs(artifacts, [design.run_id for design in designs])
    aggregated = aggregate_evaluation_runs(selected, metrics=metrics, ci_level=ci_level)
    rows = aggregated.performance.loc[aggregated.performance["dataset"].eq(RETRIEVER_DATASET)].copy()
    if rows.empty:
        raise ValueError("The unrestricted reference has no retriever-cohort performance rows")
    found = {str(instance) for instance in rows["model_instance"]}
    missing = sorted(set(map(str, model_instances)) - found)
    unexpected = sorted(found - set(map(str, model_instances)))
    if missing or unexpected:
        raise ValueError(
            "The unrestricted reference must cover the same models as the budget runs; "
            f"missing={missing}, unexpected={unexpected}"
        )
    rows["training_count"] = training_count
    return rows.reset_index(drop=True)


def _validate_unrestricted(
    client: MlflowClient,
    artifacts: PlotArtifacts,
    *,
    target: str,
    candidate_source: str,
    batch_center: str,
    metrics: tuple[str, ...],
) -> tuple[tuple[RetrieverDesign, ...], int]:
    point_data = validated_point_rows(
        artifacts,
        metrics=metrics,
        datasets=RETRIEVER_COHORTS,
        description="Unrestricted retrieval reference",
    )
    if point_data.target != target:
        raise ValueError(
            f"The unrestricted reference records target {point_data.target!r}, but the budget runs record "
            f"{target!r}"
        )
    designs = tuple(read_retriever_design(client, str(run_id)) for run_id in artifacts.run_ids)
    for design in designs:
        if design.candidate_source != candidate_source:
            raise ValueError(
                f"Unrestricted run {design.run_id} selects from {design.candidate_source!r}, but the budget runs "
                f"select from {candidate_source!r}"
            )
        if design.batch_center != batch_center:
            raise ValueError(
                f"Unrestricted run {design.run_id} targets {design.batch_center!r}, but the budget runs target "
                f"{batch_center!r}"
            )
        _require_realized_count(point_data.training_sizes, design)
    counts = {design.selected_count for design in designs}
    if len(counts) != 1:
        raise ValueError(
            "Unrestricted reference runs must all record the same candidate-pool size; found "
            f"{sorted(counts)}"
        )
    require_matched_alignments(
        {design.run_id: read_alignment_config(client, design.run_id) for design in designs},
        description="Unrestricted reference runs",
    )
    return designs, next(iter(counts))


def _require_realized_count(training_sizes: Mapping[str, int], design: RetrieverDesign) -> None:
    realized = training_sizes[design.run_id]
    if realized != design.selected_count:
        raise ValueError(
            f"Run {design.run_id} configures train_size={design.selected_count:,} but evaluation data records "
            f"training_size={realized:,}"
        )


def _require_uniform(values: set[object], expected: object | None, description: str) -> object:
    if len(values) != 1:
        raise ValueError(f"Retrieval budget runs must share one {description}; found {sorted(map(str, values))}")
    value = next(iter(values))
    if expected is not None and value != expected:
        raise ValueError(f"Retrieval budget runs record {description} {value!r}, expected {expected!r}")
    return value


def _required_int(payload: Mapping[str, object], key: str, run_id: str, section: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Run {run_id} {section}.{key} must be a JSON integer")
    return int(value)
