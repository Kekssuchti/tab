"""Prepare fixed-budget MIMIC/TUDD training-composition results."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from mlflow import MlflowClient
from src.mlflow.evaluation_data import DEFAULT_TRACKING_URI
from src.plotting.utils.artifacts import PlotArtifacts
from src.plotting.utils.grouped import GroupedEvaluation, aggregate_runs_by_setting, require_grouped_coverage
from src.plotting.utils.runs import (
    TrainOnDesign,
    read_train_on_design,
    require_aligned_bootstrap_ids,
    select_artifact_runs,
    validated_bootstrap_ids,
    validated_point_rows,
)
from src.utils.prediction_tables import PREDICTION_DATASETS


@dataclass(frozen=True)
class CompositionBudgetView:
    """Complete model summaries for the measured compositions at one fixed budget."""

    grouped: GroupedEvaluation
    compositions: pd.DataFrame
    target: str
    total_count: int

    @property
    def performance(self) -> pd.DataFrame:
        """Return performance rows with realized source counts and MIMIC share."""
        return self.grouped.performance.merge(
            self.compositions,
            on="setting",
            how="left",
            validate="many_to_one",
        )

    @property
    def model_metadata(self) -> pd.DataFrame:
        return self.grouped.model_metadata

    @property
    def model_instances(self) -> tuple[str, ...]:
        return self.grouped.model_instances

    @property
    def evaluation_centers(self) -> tuple[str, ...]:
        return tuple(PREDICTION_DATASETS)

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
class CompositionEvaluation:
    """Fixed-budget views kept separate across all recorded total budgets."""

    budget_views: tuple[CompositionBudgetView, ...]
    target: str
    metrics: tuple[str, ...]

    @property
    def budgets(self) -> tuple[int, ...]:
        return tuple(view.total_count for view in self.budget_views)

    def for_budget(self, total_count: int) -> CompositionBudgetView:
        """Return the sole prepared view for ``total_count``."""
        matches = tuple(view for view in self.budget_views if view.total_count == total_count)
        if len(matches) != 1:
            raise ValueError(f"Expected one composition view for budget {total_count}; found {len(matches)}")
        return matches[0]


@dataclass(frozen=True)
class _RunComposition:
    run_id: str
    target: str
    total_count: int
    mimic_count: int
    tudd_count: int

    @property
    def setting(self) -> str:
        return f"n{self.total_count}_m{self.mimic_count}_t{self.tudd_count}"


def prepare_composition_evaluation(
    artifacts: PlotArtifacts,
    *,
    metrics: Sequence[str],
    ci_level: float = 0.95,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> CompositionEvaluation:
    """Validate recorded source counts and aggregate repeats within exact compositions.

    Each selected pipeline run is assigned only from its authoritative
    ``config.json``. Mixed-source runs must record one positive JSON integer count
    for each of MIMIC and TUDD. A pure-source endpoint may omit the zero-count
    source, but its present source must still be a positive JSON integer. Fractions,
    including ``1.0``, are never converted into counts.

    The recorded combined ``training_size`` must equal the two configured counts.
    Runs are then grouped by the exact ``(total, MIMIC, TUDD)`` tuple. Different
    total budgets are returned as separate views and are never averaged together.
    """
    selected_metrics = tuple(dict.fromkeys(metrics))
    if not selected_metrics:
        raise ValueError("metrics must contain at least one metric")
    if not artifacts.run_ids:
        raise ValueError("Composition preparation requires at least one pipeline run")

    point_data = validated_point_rows(
        artifacts,
        metrics=selected_metrics,
        datasets=PREDICTION_DATASETS,
        description="Composition",
    )
    target = point_data.target
    bootstrap_ids = validated_bootstrap_ids(
        artifacts,
        point_data.points,
        metrics=selected_metrics,
        target=target,
        datasets=PREDICTION_DATASETS,
        description="Composition",
    )

    client = MlflowClient(tracking_uri=tracking_uri)
    designs = []
    for run_id in artifacts.run_ids:
        design = _read_composition_design(client, str(run_id))
        if design.target != target:
            raise ValueError(
                f"Run {run_id} config.json records target {design.target!r}, but its evaluation data "
                f"records {target!r}"
            )
        mimic_count = design.sample_count("mimic") or 0
        tudd_count = design.sample_count("tudd") or 0
        total_count = mimic_count + tudd_count
        recorded_size = point_data.training_sizes[str(run_id)]
        if total_count != recorded_size:
            raise ValueError(
                f"Run {run_id} records {mimic_count} MIMIC + {tudd_count} TUDD training observations "
                f"in config.json ({total_count} total), but evaluation data records training_size={recorded_size}"
            )
        designs.append(
            _RunComposition(
                run_id=str(run_id),
                target=target,
                total_count=total_count,
                mimic_count=mimic_count,
                tudd_count=tudd_count,
            )
        )

    require_aligned_bootstrap_ids(
        bootstrap_ids,
        group_by_run={design.run_id: str(design.total_count) for design in designs},
        description="Composition runs within one fixed total budget",
    )
    _validate_run_model_coverage(point_data.points, tuple(designs))
    views = tuple(
        _prepare_budget_view(
            artifacts,
            tuple(design for design in designs if design.total_count == budget),
            target=target,
            metrics=selected_metrics,
            ci_level=ci_level,
        )
        for budget in sorted({design.total_count for design in designs})
    )
    return CompositionEvaluation(budget_views=views, target=target, metrics=selected_metrics)


def _validate_run_model_coverage(points: pd.DataFrame, designs: tuple[_RunComposition, ...]) -> None:
    settings = {design.run_id: design.setting for design in designs}
    budgets = {design.run_id: design.total_count for design in designs}
    for budget in sorted(set(budgets.values())):
        budget_runs = {run_id for run_id, total in budgets.items() if total == budget}
        budget_points = points.loc[points["pipeline_mlflow_run_id"].isin(budget_runs)]
        model_sets = budget_points.groupby("pipeline_mlflow_run_id", sort=False)["model_instance"].agg(
            lambda values: frozenset(values.astype(str))
        )
        if model_sets.empty:
            raise ValueError(f"Composition budget {budget} contains no evaluated models")
        reference = model_sets.iloc[0]
        mismatched = [str(run_id) for run_id, models in model_sets.items() if models != reference]
        if mismatched:
            details = ", ".join(f"{run_id} ({settings[run_id]})" for run_id in mismatched)
            raise ValueError(
                f"Composition budget {budget} does not have complete comparable model coverage; "
                f"mismatched run(s): {details}"
            )


def _prepare_budget_view(
    artifacts: PlotArtifacts,
    designs: tuple[_RunComposition, ...],
    *,
    target: str,
    metrics: tuple[str, ...],
    ci_level: float,
) -> CompositionBudgetView:
    budgets = {design.total_count for design in designs}
    if len(budgets) != 1:
        raise ValueError(f"A composition budget view must contain one fixed total; found {sorted(budgets)}")
    total_count = next(iter(budgets))
    ordered_designs = tuple(sorted(designs, key=lambda design: (design.mimic_count, design.tudd_count)))
    setting_order = tuple(dict.fromkeys(design.setting for design in ordered_designs))
    setting_by_run = {design.run_id: design.setting for design in ordered_designs}
    run_ids = tuple(design.run_id for design in ordered_designs)
    selected = select_artifact_runs(artifacts, run_ids)
    grouped = aggregate_runs_by_setting(
        selected,
        setting_by_run,
        setting_order=setting_order,
        metrics=metrics,
        ci_level=ci_level,
    )
    require_grouped_coverage(grouped, datasets=PREDICTION_DATASETS, description="Composition")

    unique_designs = {design.setting: design for design in ordered_designs}
    compositions = pd.DataFrame(
        [
            {
                "setting": setting,
                "total_count": total_count,
                "mimic_count": unique_designs[setting].mimic_count,
                "tudd_count": unique_designs[setting].tudd_count,
                "mimic_share": 100.0 * unique_designs[setting].mimic_count / total_count,
                "run_count": grouped.run_counts[setting],
            }
            for setting in setting_order
        ]
    )
    return CompositionBudgetView(
        grouped=grouped,
        compositions=compositions,
        target=target,
        total_count=total_count,
    )


def _read_composition_design(client: MlflowClient, run_id: str) -> TrainOnDesign:
    """Require exactly one recorded integer sample count per contributing source."""
    design = read_train_on_design(client, run_id, allowed_sources=PREDICTION_DATASETS)
    if design.uses_retriever:
        raise ValueError(f"Run {run_id} configures custom_retriever; retrieval runs are not valid F6 inputs")
    full_pool = [entry.source for entry in design.entries if entry.is_full_pool]
    if full_pool:
        raise TypeError(
            f"Run {run_id} dataset.train_on configures a full training pool for {sorted(full_pool)}; a "
            "fixed-budget composition run must record one positive JSON integer count per contributing source"
        )
    if len(design.entries) == 2 and set(design.sources) != set(PREDICTION_DATASETS):
        raise ValueError(
            f"Run {run_id} mixed dataset.train_on is ambiguous; expected exactly "
            f"{tuple(PREDICTION_DATASETS)} entries"
        )
    return design
