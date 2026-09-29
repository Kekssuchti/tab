"""Prepare augmentation curves: local-only versus combined training.

Two RQ2 designs share this preparation. In both, one source is the designated
target center (local) and the other is the external source.

* Full-external augmentation (F7) fixes the complete external training pool and
  varies the number of local observations. Its combined runs therefore record one
  full-pool entry and one integer local count, and the measured external count is
  the remainder of the realized training size.
* Fixed-local augmentation (F8) fixes a local budget and varies the number of
  added external observations. Both contributions are recorded as integers, and
  the realized training size has to equal their sum.

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
) -> FullExternalAugmentation:
    """Prepare F7 with a fixed complete external pool and a varying local count."""
    selected_metrics, point_data, designs, bootstrap_ids = _read_run_designs(
        artifacts,
        local_center=local_center,
        external_source=external_source,
        metrics=metrics,
        tracking_uri=tracking_uri,
    )

    combined = tuple(design for design in designs if design.condition == COMBINED and design.external_is_full)
    external_only = tuple(design for design in designs if design.condition == EXTERNAL_ONLY)
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
    local_only = tuple(design for design in designs if design.condition == LOCAL_ONLY)
    measured_local_only = tuple(design for design in local_only if design.local_count in set(local_counts))
    _require_matched_local_cells(measured_local_only, combined, local_counts)
    _require_matched_external_only(external_only, combined)

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
    grouped = _aggregate(artifacts, runs, setting_by_run, setting_order, selected_metrics, ci_level)
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
    )


def prepare_fixed_local_augmentation(
    artifacts: PlotArtifacts,
    *,
    local_center: str,
    external_source: str,
    metrics: Sequence[str],
    ci_level: float = 0.95,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> FixedLocalAugmentation:
    """Prepare F8 with one view per fixed local budget and a varying external count."""
    selected_metrics, point_data, designs, bootstrap_ids = _read_run_designs(
        artifacts,
        local_center=local_center,
        external_source=external_source,
        metrics=metrics,
        tracking_uri=tracking_uri,
    )

    combined = tuple(design for design in designs if design.condition == COMBINED and not design.external_is_full)
    local_only = tuple(design for design in designs if design.condition == LOCAL_ONLY)
    budgets = sorted({design.local_count for design in combined})
    if not budgets:
        raise IncompleteExperimentError(
            f"The {artifacts.experiment_name!r} experiment does not measure the fixed-local design: no run "
            "records an absolute local count plus added external observations"
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
    for count in budgets:
        _require_matched_external_cells(references[count], tuple(d for d in combined if d.local_count == count), count)

    ignored = tuple(
        design.run_id
        for design in designs
        if design.condition == LOCAL_ONLY and design.local_count not in set(budgets)
    )
    require_aligned_bootstrap_ids(
        bootstrap_ids,
        group_by_run={design.run_id: "fixed-local" for design in designs},
        description="Fixed-local augmentation runs",
    )

    views = []
    for count in budgets:
        runs = (*references[count], *(d for d in combined if d.local_count == count))
        setting_by_run = {design.run_id: _fixed_local_setting(design) for design in runs}
        external_counts = sorted({design.external_count for design in runs if design.condition == COMBINED})
        setting_order = ("local_only", *(f"external:e{value}" for value in external_counts))
        grouped = _aggregate(artifacts, runs, setting_by_run, setting_order, selected_metrics, ci_level)
        views.append(
            FixedLocalBudgetView(
                grouped=grouped,
                conditions=_conditions_frame(runs, setting_by_run, grouped),
                target=point_data.target,
                local_center=local_center,
                external_source=external_source,
                local_count=count,
            )
        )
    return FixedLocalAugmentation(
        budget_views=tuple(views),
        target=point_data.target,
        metrics=selected_metrics,
        local_center=local_center,
        external_source=external_source,
        ignored_runs=ignored,
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
    designs = []
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
        if design.is_full_pool(local_center):
            raise TypeError(
                f"Run {run_id} configures the complete local pool; augmentation requires an absolute local "
                "observation count so the fixed and varying contributions stay identifiable"
            )
        local_count = design.sample_count(local_center) or 0
        external_is_full = design.is_full_pool(external_source)
        training_size = point_data.training_sizes[str(run_id)]
        if external_is_full:
            external_count = training_size - local_count
            if external_count <= 0:
                raise ValueError(
                    f"Run {run_id} records training_size={training_size:,} with a complete external pool and "
                    f"{local_count:,} local observations, leaving no external observations"
                )
            condition = EXTERNAL_ONLY if local_count == 0 else COMBINED
        else:
            external_count = design.sample_count(external_source) or 0
            if external_count:
                condition = COMBINED
                if training_size != local_count + external_count:
                    raise ValueError(
                        f"Run {run_id} records {local_count:,} local + {external_count:,} external configured "
                        f"observations but training_size={training_size:,}"
                    )
            else:
                condition = LOCAL_ONLY
                if training_size != local_count:
                    raise ValueError(
                        f"Run {run_id} records {local_count:,} configured local observations but "
                        f"training_size={training_size:,}"
                    )
        designs.append(
            _RunDesign(
                run_id=str(run_id),
                condition=condition,
                local_count=local_count,
                external_count=external_count,
                external_is_full=external_is_full,
                training_sample_seed=read_training_sample_seed(client, str(run_id)),
                training_size=training_size,
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


def _require_matched_local_cells(
    local_only: Sequence[_RunDesign],
    combined: Sequence[_RunDesign],
    local_counts: Sequence[int],
) -> None:
    """Require the local-only reference to share the repeat seeds of the combined curve."""
    for count in local_counts:
        candidate_seeds = {design.training_sample_seed for design in local_only if design.local_count == count}
        reference_seeds = {design.training_sample_seed for design in combined if design.local_count == count}
        if not candidate_seeds:
            raise IncompleteExperimentError(
                f"Full-external augmentation needs the matching local-only run at {count:,} local observations; "
                "none was selected"
            )
        if candidate_seeds != reference_seeds:
            raise ValueError(
                f"Full-external augmentation local count {count:,} is not measured with matched repeat seeds: "
                f"local-only {sorted(candidate_seeds)} versus combined {sorted(reference_seeds)}"
            )


def _require_matched_external_only(
    external_only: Sequence[_RunDesign],
    combined: Sequence[_RunDesign],
) -> None:
    if not external_only:
        return
    reference_seeds = {design.training_sample_seed for design in combined}
    seeds = {design.training_sample_seed for design in external_only}
    if seeds != reference_seeds:
        raise ValueError(
            "External-only runs must be measured with the repeat seeds of the combined curve; found "
            f"{sorted(seeds)} versus {sorted(reference_seeds)}"
        )


def _require_matched_external_cells(
    reference: Sequence[_RunDesign],
    combined: Sequence[_RunDesign],
    local_count: int,
) -> None:
    expected = {design.training_sample_seed for design in reference}
    seeds = {design.training_sample_seed for design in combined}
    if expected != seeds:
        raise ValueError(
            f"Local budget {local_count:,} is not measured with matched repeat seeds: the local-only reference "
            f"uses {sorted(expected)} and the combined runs use {sorted(seeds)}"
        )


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
