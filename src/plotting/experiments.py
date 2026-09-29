"""Central registry of intended plotting experiments.

The registry describes the complete main-experiment scope, including inputs that
have not been run or named yet.  Figure scripts therefore report a missing
registered input instead of guessing an experiment name or substituting a
different result.

``training_source`` is the sole source for single-source sweeps, the external
source for augmentation, and the candidate-pool source for retrieval.
``evaluation_center`` is set for directional augmentation/retrieval designs and
is ``None`` when a figure evaluates both centers (single-source and composition).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass

from src.plotting.defaults import metric_scale, task_label

MAIN_TARGETS = ("mortality", "LOS7", "hours_to_readmit_72")
DATA_SOURCES = ("mimic", "tudd")
CLASSIFICATION_METRICS = ("roc_auc", "prc_auc")

SINGLE_SOURCE = "single_source_sample_size"
COMPOSITION = "composition"
AUGMENTATION_FULL_EXTERNAL = "augmentation_full_external"
AUGMENTATION_FIXED_LOCAL = "augmentation_fixed_local"
RETRIEVAL = "retrieval"
# The unrestricted candidate-pool reference is a separately registered input:
# existing full-held-out-set results are not equivalent to a matched-batch run.
RETRIEVAL_UNRESTRICTED = "retrieval_unrestricted"


@dataclass(frozen=True)
class PlotExperiment:
    """One intended experiment input, whether available now or still missing."""

    family: str
    target: str
    training_source: str | None
    evaluation_center: str | None
    experiment_name: str | None
    metrics: tuple[str, ...] = CLASSIFICATION_METRICS

    def __post_init__(self) -> None:
        scales = {metric_scale(metric) for metric in self.metrics}
        if len(scales) != 1:
            raise ValueError(f"Task {self.target!r} mixes metrics on different display scales: {self.metrics}")
        for field, value in (
            ("training_source", self.training_source),
            ("evaluation_center", self.evaluation_center),
        ):
            if value is not None and value not in DATA_SOURCES:
                raise ValueError(f"Unknown {field} {value!r}; expected one of {DATA_SOURCES}")

    @property
    def label(self) -> str:
        return task_label(self.target)

    @property
    def score_scale(self) -> float:
        """Return the shared points-to-percent scale of this task's metrics."""
        return float(metric_scale(self.metrics[0]))

    @property
    def cross_metric(self) -> str:
        """Return the metric used where one representative is enough (matrices, ranks)."""
        return self.metrics[0]

    @property
    def available(self) -> bool:
        """Whether an MLflow experiment name has been registered for this input."""
        return self.experiment_name is not None

    @property
    def direction(self) -> str:
        """Return a reader-facing description of the source/evaluation direction."""
        if self.training_source is None:
            return "mixed sources -> both evaluation centers"
        if self.evaluation_center is None:
            return f"{self.training_source} -> both evaluation centers"
        return f"{self.training_source} -> {self.evaluation_center}"

    @property
    def direction_slug(self) -> str:
        """Return a filesystem-safe source/evaluation identity."""
        if self.training_source is None:
            return "mixed_sources"
        if self.evaluation_center is None:
            return self.training_source
        return f"{self.training_source}_to_{self.evaluation_center}"


# This sparse mapping is the only place MLflow experiment names are registered.
# Missing keys deliberately become ``experiment_name=None`` placeholders below.
_EXPERIMENT_NAMES: dict[tuple[str, str, str | None, str | None], str] = {
    (SINGLE_SOURCE, "mortality", "mimic", None): "sample_size_mimic_mortality",
    (SINGLE_SOURCE, "mortality", "tudd", None): "sample_size_tudd_mortality",
    (SINGLE_SOURCE, "hours_to_readmit_72", "tudd", None): "sample_size_tudd_hours_to_readmit_72",
    (RETRIEVAL, "mortality", "tudd", "tudd"): "retriever_tudd_mortality",
}


def _experiment(
    family: str,
    target: str,
    training_source: str | None,
    evaluation_center: str | None,
) -> PlotExperiment:
    key = (family, target, training_source, evaluation_center)
    return PlotExperiment(family, target, training_source, evaluation_center, _EXPERIMENT_NAMES.get(key))


def _other_source(source: str) -> str:
    return "tudd" if source == "mimic" else "mimic"


PLOT_EXPERIMENTS: tuple[PlotExperiment, ...] = (
    # F1/F2/F4/F5 inputs: every target in both single-source directions.
    *(_experiment(SINGLE_SOURCE, target, source, None) for target in MAIN_TARGETS for source in DATA_SOURCES),
    # F6 inputs: source composition is symmetric and both centers are evaluated.
    *(_experiment(COMPOSITION, target, None, None) for target in MAIN_TARGETS),
    # F7/F8 inputs: training_source is external; evaluation_center is local/target.
    *(
        _experiment(family, target, _other_source(target_center), target_center)
        for family in (AUGMENTATION_FULL_EXTERNAL, AUGMENTATION_FIXED_LOCAL)
        for target in MAIN_TARGETS
        for target_center in DATA_SOURCES
    ),
    # F9/F10 inputs: every candidate-source/target-batch direction, including
    # same-center retrieval.
    *(
        _experiment(RETRIEVAL, target, source, evaluation_center)
        for target in MAIN_TARGETS
        for source in DATA_SOURCES
        for evaluation_center in DATA_SOURCES
    ),
    # F10's unrestricted candidate-pool reference, registered per direction.
    *(
        _experiment(RETRIEVAL_UNRESTRICTED, target, source, evaluation_center)
        for target in MAIN_TARGETS
        for source in DATA_SOURCES
        for evaluation_center in DATA_SOURCES
    ),
)


_FIGURE_EXPERIMENT_FAMILY = {
    "baseline": SINGLE_SOURCE,
    "training_source_contrast": SINGLE_SOURCE,
    "sample_size": SINGLE_SOURCE,
    "pairwise_wins": SINGLE_SOURCE,
    "pairwise_tables": SINGLE_SOURCE,
    "composition": COMPOSITION,
    "augmentation_full_external": AUGMENTATION_FULL_EXTERNAL,
    "augmentation_fixed_local": AUGMENTATION_FIXED_LOCAL,
    "retriever_comparison": RETRIEVAL,
    "retrieval_budget": RETRIEVAL,
}


def tasks_for(
    figure_family: str,
    targets: Sequence[str] | None = None,
    *,
    training_sources: Sequence[str] | None = None,
    evaluation_centers: Sequence[str] | None = None,
) -> tuple[PlotExperiment, ...]:
    """Return declared inputs for a figure, optionally filtered by design dimensions."""
    if figure_family not in _FIGURE_EXPERIMENT_FAMILY:
        raise KeyError(
            f"Unknown figure family {figure_family!r}; declared families: {sorted(_FIGURE_EXPERIMENT_FAMILY)}"
        )
    experiment_family = _FIGURE_EXPERIMENT_FAMILY[figure_family]
    declared = tuple(task for task in PLOT_EXPERIMENTS if task.family == experiment_family)

    _require_known_filter("target", targets, {task.target for task in declared}, figure_family)
    _require_known_filter(
        "training source",
        training_sources,
        {task.training_source for task in declared if task.training_source is not None},
        figure_family,
    )
    _require_known_filter(
        "evaluation center",
        evaluation_centers,
        {task.evaluation_center for task in declared if task.evaluation_center is not None},
        figure_family,
    )

    selected = declared
    if targets is not None:
        selected = tuple(task for task in selected if task.target in set(targets))
    if training_sources is not None:
        selected = tuple(task for task in selected if task.training_source in set(training_sources))
    if evaluation_centers is not None:
        selected = tuple(task for task in selected if task.evaluation_center in set(evaluation_centers))
    return selected


def reciprocal_single_source(task: PlotExperiment) -> PlotExperiment:
    """Return the opposite-center single-source input for the same target."""
    if task.family != SINGLE_SOURCE or task.training_source is None or task.evaluation_center is not None:
        raise ValueError(f"Reciprocal lookup requires a single-source task; received {task}")
    reciprocal_source = _other_source(task.training_source)
    matches = tuple(
        candidate
        for candidate in PLOT_EXPERIMENTS
        if candidate.family == task.family
        and candidate.target == task.target
        and candidate.training_source == reciprocal_source
        and candidate.evaluation_center is None
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one reciprocal input for {task.target!r} trained on {task.training_source!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def retrieval_unrestricted_input(task: PlotExperiment) -> PlotExperiment:
    """Return the declared unrestricted candidate-pool input for a retrieval task."""
    if task.family != RETRIEVAL:
        raise ValueError(f"Unrestricted lookup requires a retrieval input; received {task}")
    matches = tuple(
        candidate
        for candidate in PLOT_EXPERIMENTS
        if candidate.family == RETRIEVAL_UNRESTRICTED
        and candidate.target == task.target
        and candidate.training_source == task.training_source
        and candidate.evaluation_center == task.evaluation_center
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one unrestricted retrieval declaration for {task.target!r}/{task.direction}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _require_known_filter(
    label: str,
    requested: Sequence[str] | None,
    available: set[str],
    figure_family: str,
) -> None:
    if requested is None:
        return
    unknown = set(requested) - available
    if unknown:
        raise ValueError(
            f"Family {figure_family!r} declares no {label}(s) {sorted(unknown)}; declared: {sorted(available)}"
        )


def warn_skipped(task: PlotExperiment, reason: object) -> None:
    """Report an intended input that is not available without replacing it."""
    experiment = task.experiment_name or "no MLflow experiment registered"
    print(f"warning: skipping {task.label} [{task.family}; {task.direction}] ({experiment}): {reason}", file=sys.stderr)
