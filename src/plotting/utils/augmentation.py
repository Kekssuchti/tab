"""Prepare augmentation curves: local-only versus combined training.

Two RQ2 designs share this preparation. In both, one source is the designated
target center (local) and the other is the external source.

* Full-external augmentation (F7) fixes the complete external training pool and
  varies the number of local observations. Its combined runs therefore record one
  full-pool entry and one integer local count, and the measured external count is
  the remainder of the realized training size.
* Fixed-local augmentation (F8) fixes a local budget and varies the number of
  added external observations. The local budget may be an integer or the complete
  pool; the external additions are integers, and the realized training size has
  to equal their sum.

Because the dataset builder draws each source's subset independently from that
source's split with the training-side sample seed, two runs that share a seed and
an integer local count contain the same local observations. That is what makes
the local-only reference and the combined curve comparable, and it is checked
here by requiring matched repeat seeds inside every comparison cell.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from mlflow import MlflowClient
from src.mlflow.evaluation_data import DEFAULT_TRACKING_URI
from src.plotting.utils.artifacts import PlotArtifacts
from src.plotting.utils.grouped import GroupedEvaluation, aggregate_runs_by_setting, require_grouped_coverage
from src.plotting.utils.runs import (
    IncompleteExperimentError,
    RunPointData,
    read_train_on_design,
    read_training_sample_seed,
    require_aligned_bootstrap_ids,
    select_artifact_runs,
    validated_bootstrap_ids,
    validated_point_rows,
)
from src.utils.prediction_tables import PREDICTION_DATASETS

LOCAL_ONLY = "local_only"
COMBINED = "combined"
EXTERNAL_ONLY = "external_only"

CONDITION_LABELS = {LOCAL_ONLY: "Local only", COMBINED: "Full external + local"}
EVALUATION_CENTERS = PREDICTION_DATASETS


@dataclass(frozen=True)
class _RunDesign:
    """The augmentation design recorded by one run."""

    run_id: str
    condition: str
    local_count: int
    external_count: int
    external_is_full: bool
    training_sample_seed: int
    training_size: int

    def describe(self) -> str:
        return (
            f"{self.condition} (local={self.local_count:,}, external={self.external_count:,}"
            f"{', complete pool' if self.external_is_full else ''}, seed={self.training_sample_seed})"
        )


@dataclass(frozen=True)
class FullExternalAugmentation:
    """Local-only and full-external-plus-local curves over the local sample count."""

    grouped: GroupedEvaluation
    conditions: pd.DataFrame
    target: str
    local_center: str
    external_source: str
    external_count: int
    local_counts: tuple[int, ...]
    external_only_measured: bool
    ignored_runs: tuple[str, ...]
    dropped_runs: tuple[str, ...] = ()
    local_only_experiment: str | None = None

    @property
    def performance(self) -> pd.DataFrame:
        return self.grouped.performance.merge(self.conditions, on="setting", how="left", validate="many_to_one")

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
    def evaluation_centers(self) -> tuple[str, ...]:
        return EVALUATION_CENTERS

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
class FixedLocalBudgetView:
    """Augmentation curves for one fixed local training budget."""

    grouped: GroupedEvaluation
    conditions: pd.DataFrame
    target: str
    local_center: str
    external_source: str
    local_count: int
    local_only_experiment: str | None = None
    dropped_runs: tuple[str, ...] = ()

    @property
    def performance(self) -> pd.DataFrame:
        return self.grouped.performance.merge(self.conditions, on="setting", how="left", validate="many_to_one")

    @property
    def external_counts(self) -> tuple[int, ...]:
        counts = set(self.conditions["external_count"].astype(int))
        return tuple(sorted(count for count in counts if count > 0))

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
    def evaluation_centers(self) -> tuple[str, ...]:
        return EVALUATION_CENTERS

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
class FixedLocalAugmentation:
    """One view per measured local budget; budgets are never averaged together."""

    budget_views: tuple[FixedLocalBudgetView, ...]
    target: str
    metrics: tuple[str, ...]
    local_center: str
    external_source: str
    ignored_runs: tuple[str, ...]
    dropped_runs: tuple[str, ...] = ()
    local_only_experiment: str | None = None

    @property
    def local_counts(self) -> tuple[int, ...]:
        return tuple(view.local_count for view in self.budget_views)

    def for_local_count(self, local_count: int) -> FixedLocalBudgetView:
        matches = tuple(view for view in self.budget_views if view.local_count == local_count)
        if len(matches) != 1:
            raise ValueError(f"Expected one augmentation view for local budget {local_count}; found {len(matches)}")
        return matches[0]


def prepare_full_external_augmentation(
    artifacts: PlotArtifacts,
    *,
    local_center: str,
    external_source: str,
    metrics: Sequence[str],
    ci_level: float = 0.95,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    local_only_artifacts: PlotArtifacts | None = None,
    external_only_artifacts: PlotArtifacts | None = None,
) -> FullExternalAugmentation:
    """Prepare F7 with a fixed complete external pool and a varying local count.

    The local-only curve may come from the local center's own single-source
    experiment, which is the same sweep the local learning curves use: its
    training-side sample seed and its observed counts are matched cell by cell,
    and only runs whose realized training size confirms the configured count are
    accepted. The external-only boundary may likewise come from the external
    source's single-source experiment.
    """
    selected_metrics, point_data, designs, bootstrap_ids = _read_run_designs(
        artifacts,
        local_center=local_center,
        external_source=external_source,
        metrics=metrics,
        tracking_uri=tracking_uri,
    )

    combined = tuple(design for design in designs if design.condition == COMBINED and design.external_is_full)
    external_only = tuple(design for design in designs if design.condition == EXTERNAL_ONLY)
    local_only = tuple(design for design in designs if design.condition == LOCAL_ONLY)
    if local_only_artifacts is not None:
        reference = _read_reference_designs(
            local_only_artifacts,
            role=LOCAL_ONLY,
            local_center=local_center,
            external_source=external_source,
            metrics=metrics,
            tracking_uri=tracking_uri,
            expected_target=point_data.target,
        )
        local_only = _merge_cells(local_only, reference.designs, description="local-only")
        bootstrap_ids = {**bootstrap_ids, **reference.bootstrap_ids}
    if external_only_artifacts is not None:
        reference = _read_reference_designs(
            external_only_artifacts,
            role=EXTERNAL_ONLY,
            local_center=local_center,
            external_source=external_source,
            metrics=metrics,
            tracking_uri=tracking_uri,
            expected_target=point_data.target,
        )
        external_only = _merge_cells(external_only, reference.designs, description="external-only")
        bootstrap_ids = {**bootstrap_ids, **reference.bootstrap_ids}
    if not combined:
        raise IncompleteExperimentError(
            f"The {artifacts.experiment_name!r} experiment does not measure the full-external design: no run "
            "records the complete external pool (fraction 1.0) plus a positive local observation count. Runs that "
            "record an integer external count belong to the fixed-local augmentation figure instead."
        )

    external_counts = {design.external_count for design in combined}
    if len(external_counts) != 1:
        raise ValueError(
            "Full-external augmentation fixes the complete external pool, but the selected runs realize "
            f"{sorted(external_counts)} external observations. A varying external contribution is the "
            "fixed-local augmentation design, not this one."
        )
    external_count = next(iter(external_counts))
    for design in external_only:
        if design.external_count != external_count:
            raise ValueError(
                f"Run {design.run_id} realizes {design.external_count:,} external observations, but the combined "
                f"runs fix the complete external pool at {external_count:,}"
            )

    local_counts = tuple(sorted({design.local_count for design in combined}))
    measured_local_only = tuple(design for design in local_only if design.local_count in set(local_counts))
    measured_local_only, dropped_local_only = _matched_local_cells(measured_local_only, combined, local_counts)
    external_only, dropped_external_only = _selected_external_only(external_only, combined)

    ignored = tuple(
        design.run_id
        for design in designs
        if design.condition == COMBINED and not design.external_is_full
        or design.condition == LOCAL_ONLY and design.local_count not in set(local_counts)
    )
    runs = (*external_only, *measured_local_only, *combined)
    setting_by_run = {
        design.run_id: _full_external_setting(design) for design in runs
    }
    setting_order = _full_external_setting_order(local_counts, measured_zero=bool(external_only))
    grouped = _aggregate(
        _combine_artifacts(artifacts, local_only_artifacts, external_only_artifacts),
        runs,
        setting_by_run,
        setting_order,
        selected_metrics,
        ci_level,
    )
    require_aligned_bootstrap_ids(
        bootstrap_ids,
        group_by_run={design.run_id: "full-external" for design in runs},
        description="Full-external augmentation runs",
    )

    conditions = _conditions_frame(runs, setting_by_run, grouped)
    return FullExternalAugmentation(
        grouped=grouped,
        conditions=conditions,
        target=point_data.target,
        local_center=local_center,
        external_source=external_source,
        external_count=external_count,
        local_counts=local_counts,
        external_only_measured=bool(external_only),
        ignored_runs=ignored,
        dropped_runs=(*dropped_local_only, *dropped_external_only),
        local_only_experiment=None if local_only_artifacts is None else local_only_artifacts.experiment_name,
    )


def prepare_fixed_local_augmentation(
    artifacts: PlotArtifacts,
    *,
    local_center: str,
    external_source: str,
    metrics: Sequence[str],
    ci_level: float = 0.95,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    local_only_artifacts: PlotArtifacts | None = None,
) -> FixedLocalAugmentation:
    """Prepare F8 with one view per fixed local budget and a varying external count.

    As in F7, the local-only reference may be supplied by the local center's
    single-source experiment; a budget is plotted only when that reference is
    measured with the same repeat seeds.
    """
    selected_metrics, point_data, designs, bootstrap_ids = _read_run_designs(
        artifacts,
        local_center=local_center,
        external_source=external_source,
        metrics=metrics,
        tracking_uri=tracking_uri,
    )

    combined = tuple(design for design in designs if design.condition == COMBINED and not design.external_is_full)
    local_only = tuple(design for design in designs if design.condition == LOCAL_ONLY)
    if local_only_artifacts is not None:
        reference = _read_reference_designs(
            local_only_artifacts,
            role=LOCAL_ONLY,
            local_center=local_center,
            external_source=external_source,
            metrics=metrics,
            tracking_uri=tracking_uri,
            expected_target=point_data.target,
        )
        local_only = _merge_cells(local_only, reference.designs, description="local-only")
        bootstrap_ids = {**bootstrap_ids, **reference.bootstrap_ids}
    budgets = sorted({design.local_count for design in combined})
    if not budgets:
        raise IncompleteExperimentError(
            f"The {artifacts.experiment_name!r} experiment does not measure the fixed-local design: no run "
            "records a fixed local contribution plus added external observations"
        )

    references = {
        count: tuple(design for design in local_only if design.local_count == count) for count in budgets
    }
    missing = [count for count, runs in references.items() if not runs]
    if missing:
        raise IncompleteExperimentError(
            "Fixed-local augmentation needs the local-only reference at the same local budget; missing local "
            "budget(s): " + ", ".join(f"{count:,}" for count in missing)
        )
    dropped: list[str] = []
    for count in budgets:
        matched, unmatched = _selected_reference_cells(
            references[count],
            tuple(design for design in combined if design.local_count == count),
            count,
        )
        references[count] = matched
        dropped.extend(unmatched)

    ignored = tuple(
        design.run_id
        for design in designs
        if design.condition == LOCAL_ONLY and design.local_count not in set(budgets)
    )
    reference_runs = (*local_only, *combined)
    require_aligned_bootstrap_ids(
        bootstrap_ids,
        group_by_run={design.run_id: "fixed-local" for design in reference_runs},
        description="Fixed-local augmentation runs",
    )

    views = []
    for count in budgets:
        runs = (*references[count], *(d for d in combined if d.local_count == count))
        setting_by_run = {design.run_id: _fixed_local_setting(design) for design in runs}
        external_counts = sorted({design.external_count for design in runs if design.condition == COMBINED})
        setting_order = ("local_only", *(f"external:e{value}" for value in external_counts))
        grouped = _aggregate(
            _combine_artifacts(artifacts, local_only_artifacts),
            runs,
            setting_by_run,
            setting_order,
            selected_metrics,
            ci_level,
        )
        views.append(
            FixedLocalBudgetView(
                grouped=grouped,
                conditions=_conditions_frame(runs, setting_by_run, grouped),
                target=point_data.target,
                local_center=local_center,
                external_source=external_source,
                local_count=count,
                local_only_experiment=(
                    None if local_only_artifacts is None else local_only_artifacts.experiment_name
                ),
                dropped_runs=tuple(dropped),
            )
        )
    return FixedLocalAugmentation(
        budget_views=tuple(views),
        target=point_data.target,
        metrics=selected_metrics,
        local_center=local_center,
        external_source=external_source,
        ignored_runs=ignored,
        dropped_runs=tuple(dropped),
        local_only_experiment=None if local_only_artifacts is None else local_only_artifacts.experiment_name,
    )


@dataclass(frozen=True)
class _RawRun:
    """One augmentation run before its two contributions are resolved."""

    run_id: str
    local_configured: int | None
    external_configured: int | None
    local_is_full: bool
    external_is_full: bool
    training_size: int
    training_sample_seed: int


@dataclass(frozen=True)
class _ReferenceDesigns:
    """Local-only or external-only runs contributed by a single-source experiment."""

    designs: tuple[_RunDesign, ...]
    bootstrap_ids: dict[str, tuple[float, ...]]
    skipped_runs: tuple[str, ...]


def _read_reference_designs(
    artifacts: PlotArtifacts,
    *,
    role: str,
    local_center: str,
    external_source: str,
    metrics: Sequence[str],
    tracking_uri: str,
    expected_target: str,
) -> _ReferenceDesigns:
    """Read the single-source runs that supply one reference curve.

    A single-source experiment holds many sizes; only the runs that fill the
    requested role are kept, and the others are reported as skipped instead of
    being mistaken for part of the augmentation design.
    """
    selected_metrics = tuple(dict.fromkeys(metrics))
    point_data = validated_point_rows(
        artifacts,
        metrics=selected_metrics,
        datasets=PREDICTION_DATASETS,
        description=f"Augmentation {role} reference",
    )
    if point_data.target != expected_target:
        raise ValueError(
            f"The {artifacts.experiment_name!r} reference experiment records target {point_data.target!r}, but the "
            f"augmentation sweep records {expected_target!r}"
        )
    bootstrap_ids = validated_bootstrap_ids(
        artifacts,
        point_data.points,
        metrics=selected_metrics,
        target=point_data.target,
        datasets=PREDICTION_DATASETS,
        description=f"Augmentation {role} reference",
    )
    _require_distinct_sources(local_center, external_source)

    client = MlflowClient(tracking_uri=tracking_uri)
    designs = []
    skipped = []
    for run_id in artifacts.run_ids:
        design = read_train_on_design(client, str(run_id))
        if design.uses_retriever:
            raise ValueError(f"Run {run_id} configures custom_retriever; retrieval runs are not augmentation inputs")
        training_size = point_data.training_sizes[str(run_id)]
        if role == LOCAL_ONLY:
            if design.sources != (local_center,):
                skipped.append(str(run_id))
                continue
            condition = LOCAL_ONLY
            local_count = training_size
            external_count = 0
        elif role == EXTERNAL_ONLY:
            if design.sources != (external_source,) or not design.is_full_pool(external_source):
                skipped.append(str(run_id))
                continue
            condition = EXTERNAL_ONLY
            local_count = 0
            external_count = training_size
        else:
            raise ValueError(f"Unknown augmentation reference role {role!r}")
        if local_count <= 0 and condition == LOCAL_ONLY:
            raise ValueError(f"Reference run {run_id} has no local observations")
        designs.append(
            _RunDesign(
                run_id=str(run_id),
                condition=condition,
                local_count=local_count,
                external_count=external_count,
                external_is_full=design.is_full_pool(external_source),
                training_sample_seed=read_training_sample_seed(client, str(run_id)),
                training_size=training_size,
            )
        )
    if not designs:
        raise IncompleteExperimentError(
            f"The {artifacts.experiment_name!r} reference experiment contains no {role} run for "
            f"{local_center!r}/{external_source!r}"
        )
    return _ReferenceDesigns(designs=tuple(designs), bootstrap_ids=bootstrap_ids, skipped_runs=tuple(skipped))


def _merge_cells(
    primary: Sequence[_RunDesign],
    reference: Sequence[_RunDesign],
    *,
    description: str,
) -> tuple[_RunDesign, ...]:
    """Add reference runs without ever counting one measurement twice."""
    merged: dict[tuple[str, int, int], _RunDesign] = {}
    for design in (*primary, *reference):
        key = (design.condition, design.local_count, design.training_sample_seed)
        if key in merged:
            raise ValueError(
                f"Duplicate {description} measurement for {key}: runs {merged[key].run_id} and {design.run_id}"
            )
        merged[key] = design
    return tuple(merged.values())


def _combine_artifacts(*sets: PlotArtifacts | None) -> PlotArtifacts:
    """Concatenate the artifacts of several experiments into one selection."""
    selected = [item for item in sets if item is not None]
    return PlotArtifacts(
        metrics=pd.concat([item.metrics for item in selected], ignore_index=True),
        bootstrap_scores=pd.concat([item.bootstrap_scores for item in selected], ignore_index=True),
        experiment_name=" + ".join(dict.fromkeys(item.experiment_name for item in selected)),
        run_ids=tuple(run_id for item in selected for run_id in item.run_ids),
    )


def _read_run_designs(
    artifacts: PlotArtifacts,
    *,
    local_center: str,
    external_source: str,
    metrics: Sequence[str],
    tracking_uri: str,
) -> tuple[tuple[str, ...], RunPointData, tuple[_RunDesign, ...], dict[str, tuple[float, ...]]]:
    selected_metrics = tuple(dict.fromkeys(metrics))
    if not selected_metrics:
        raise ValueError("metrics must contain at least one metric")
    if not artifacts.run_ids:
        raise ValueError("Augmentation preparation requires at least one pipeline run")
    _require_distinct_sources(local_center, external_source)

    point_data = validated_point_rows(
        artifacts,
        metrics=selected_metrics,
        datasets=PREDICTION_DATASETS,
        description="Augmentation",
    )
    bootstrap_ids = validated_bootstrap_ids(
        artifacts,
        point_data.points,
        metrics=selected_metrics,
        target=point_data.target,
        datasets=PREDICTION_DATASETS,
        description="Augmentation",
    )

    client = MlflowClient(tracking_uri=tracking_uri)
    raw: list[_RawRun] = []
    for run_id in artifacts.run_ids:
        design = read_train_on_design(client, str(run_id), allowed_sources=PREDICTION_DATASETS)
        if design.target != point_data.target:
            raise ValueError(
                f"Run {run_id} config.json records target {design.target!r}, but its evaluation data records "
                f"{point_data.target!r}"
            )
        if design.uses_retriever:
            raise ValueError(f"Run {run_id} configures custom_retriever; retrieval runs are not augmentation inputs")
        unknown = sorted(set(design.sources) - {local_center, external_source})
        if unknown:
            raise ValueError(
                f"Run {run_id} trains on {unknown}, which is neither the local center {local_center!r} nor the "
                f"external source {external_source!r}"
            )
        raw.append(
            _RawRun(
                run_id=str(run_id),
                local_configured=design.sample_count(local_center),
                external_configured=design.sample_count(external_source),
                local_is_full=design.is_full_pool(local_center),
                external_is_full=design.is_full_pool(external_source),
                training_size=point_data.training_sizes[str(run_id)],
                training_sample_seed=read_training_sample_seed(client, str(run_id)),
            )
        )

    external_pools = {
        item.training_size - (item.local_configured or 0)
        for item in raw
        if item.external_is_full and not item.local_is_full
    }
    if len(external_pools) > 1:
        raise ValueError(
            "Augmentation runs that fix the complete external pool disagree on its realized size: "
            + ", ".join(f"{value:,}" for value in sorted(external_pools))
        )
    external_pool = next(iter(external_pools)) if external_pools else None

    local_pools = {
        item.training_size - int(item.external_configured)
        for item in raw
        if item.local_is_full and not item.external_is_full and item.external_configured is not None
    }
    if len(local_pools) > 1:
        raise ValueError(
            "Augmentation runs that fix the complete local pool disagree on its realized size: "
            + ", ".join(f"{value:,}" for value in sorted(local_pools))
        )
    local_pool = next(iter(local_pools)) if local_pools else None

    designs = []
    for item in raw:
        local_count = item.local_configured or 0
        external_count = item.external_configured or 0
        if item.local_is_full and item.external_is_full:
            if external_pool is not None:
                external_count = external_pool
                local_count = item.training_size - external_count
            elif local_pool is not None:
                local_count = local_pool
                external_count = item.training_size - local_count
            else:
                raise IncompleteExperimentError(
                    f"Run {item.run_id} trains on both complete pools, but no run of "
                    f"{artifacts.experiment_name!r} realizes either pool size, so its contributions "
                    "cannot be determined"
                )
            condition = COMBINED
        elif item.local_is_full and item.external_configured is not None:
            external_count = int(item.external_configured)
            local_count = item.training_size - external_count
            condition = COMBINED
        elif item.local_is_full:
            local_count = item.training_size
            external_count = 0
            condition = LOCAL_ONLY
        elif item.external_is_full:
            local_count = item.local_configured or 0
            external_count = item.training_size - local_count
            condition = EXTERNAL_ONLY if local_count == 0 else COMBINED
        else:
            condition = COMBINED if external_count else LOCAL_ONLY
        if local_count <= 0 or external_count < 0 or item.training_size != local_count + external_count:
            raise ValueError(
                f"Run {item.run_id} resolves to {local_count:,} local and {external_count:,} external observations, "
                f"which does not match its recorded training_size={item.training_size:,}"
            )
        designs.append(
            _RunDesign(
                run_id=item.run_id,
                condition=condition,
                local_count=local_count,
                external_count=external_count,
                external_is_full=item.external_is_full,
                training_sample_seed=item.training_sample_seed,
                training_size=item.training_size,
            )
        )
    return selected_metrics, point_data, tuple(designs), bootstrap_ids


def _conditions_frame(
    runs: Sequence[_RunDesign],
    setting_by_run: Mapping[str, str],
    grouped: GroupedEvaluation,
) -> pd.DataFrame:
    rows = [
        {
            "setting": setting_by_run[design.run_id],
            "training_condition": design.condition,
            "local_count": design.local_count,
            "external_count": design.external_count,
            "run_count": grouped.run_counts[setting_by_run[design.run_id]],
        }
        for design in runs
    ]
    return (
        pd.DataFrame(rows)
        .drop_duplicates("setting")
        .sort_values(["local_count", "external_count"], kind="stable")
        .reset_index(drop=True)
    )


def _aggregate(
    artifacts: PlotArtifacts,
    runs: Sequence[_RunDesign],
    setting_by_run: Mapping[str, str],
    setting_order: Sequence[str],
    metrics: Sequence[str],
    ci_level: float,
) -> GroupedEvaluation:
    selected = select_artifact_runs(artifacts, [design.run_id for design in runs])
    grouped = aggregate_runs_by_setting(
        selected,
        setting_by_run,
        setting_order=setting_order,
        metrics=metrics,
        ci_level=ci_level,
    )
    require_grouped_coverage(grouped, datasets=PREDICTION_DATASETS, description="Augmentation")
    return grouped


def _matched_local_cells(
    local_only: Sequence[_RunDesign],
    combined: Sequence[_RunDesign],
    local_counts: Sequence[int],
) -> tuple[tuple[_RunDesign, ...], tuple[str, ...]]:
    """Keep only the local-only runs that repeat the combined curve's seeds.

    A single-source sweep may repeat a size more often than the augmentation
    sweep does. Averaging the reference over more repeats than the curve it is
    compared with would mix repeat counts, so unmatched reference runs are
    dropped and reported instead of being averaged in.
    """
    kept: list[_RunDesign] = []
    dropped: list[_RunDesign] = []
    for count in local_counts:
        combined_seeds = {design.training_sample_seed for design in combined if design.local_count == count}
        candidates = [design for design in local_only if design.local_count == count]
        if not candidates:
            raise IncompleteExperimentError(
                f"Full-external augmentation needs the matching local-only run at {count:,} local observations; "
                "none was selected"
            )
        matched = [design for design in candidates if design.training_sample_seed in combined_seeds]
        if not matched:
            raise IncompleteExperimentError(
                f"Local count {count:,} has no local-only run with the repeat seeds of the combined curve: "
                f"local-only {sorted(design.training_sample_seed for design in candidates)} versus combined "
                f"{sorted(combined_seeds)}"
            )
        kept.extend(matched)
        dropped.extend(design for design in candidates if design not in matched)
    return tuple(kept), tuple(design.run_id for design in dropped)


def _selected_external_only(
    external_only: Sequence[_RunDesign],
    combined: Sequence[_RunDesign],
) -> tuple[tuple[_RunDesign, ...], tuple[str, ...]]:
    """Keep only the external-only runs measured with the combined curve's seeds."""
    if not external_only:
        return (), ()
    combined_seeds = {design.training_sample_seed for design in combined}
    matched = tuple(design for design in external_only if design.training_sample_seed in combined_seeds)
    if not matched:
        raise IncompleteExperimentError(
            "No external-only run shares the repeat seeds of the combined curve: external-only "
            f"{sorted(design.training_sample_seed for design in external_only)} versus combined "
            f"{sorted(combined_seeds)}"
        )
    dropped = tuple(design.run_id for design in external_only if design.training_sample_seed not in combined_seeds)
    return matched, dropped


def _selected_reference_cells(
    reference: Sequence[_RunDesign],
    combined: Sequence[_RunDesign],
    local_count: int,
) -> tuple[tuple[_RunDesign, ...], tuple[str, ...]]:
    """Keep the local-only reference runs whose repeats match one local budget."""
    combined_seeds = {design.training_sample_seed for design in combined}
    matched = tuple(design for design in reference if design.training_sample_seed in combined_seeds)
    if not matched:
        raise IncompleteExperimentError(
            f"Local budget {local_count:,} has no local-only reference with the repeat seeds of its combined runs: "
            f"reference {sorted(design.training_sample_seed for design in reference)} versus combined "
            f"{sorted(combined_seeds)}"
        )
    dropped = tuple(design.run_id for design in reference if design.training_sample_seed not in combined_seeds)
    return matched, dropped


def _full_external_setting(design: _RunDesign) -> str:
    if design.condition == LOCAL_ONLY:
        return f"local_only:n{design.local_count}"
    if design.condition == EXTERNAL_ONLY:
        return "combined:n0"
    return f"combined:n{design.local_count}"


def _full_external_setting_order(local_counts: Sequence[int], *, measured_zero: bool) -> tuple[str, ...]:
    order = ["combined:n0"] if measured_zero else []
    for count in local_counts:
        order.extend((f"local_only:n{count}", f"combined:n{count}"))
    return tuple(order)


def _fixed_local_setting(design: _RunDesign) -> str:
    return "local_only" if design.condition == LOCAL_ONLY else f"external:e{design.external_count}"


def _require_distinct_sources(local_center: str, external_source: str) -> None:
    if local_center == external_source:
        raise ValueError("Augmentation requires two distinct sources: a local center and an external source")
    for source in (local_center, external_source):
        if source not in PREDICTION_DATASETS:
            raise ValueError(f"Unknown source {source!r}; expected one of {PREDICTION_DATASETS}")
