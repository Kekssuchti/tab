"""MLflow experiment selection for the figure scripts.

Every figure script that reads MLflow artifacts for a prediction task asks this
module which experiments to read, so adding a task next to ``mortality`` is one
entry in one file instead of an experiment name in every script.

These families write into ``plots/<family>/<target>/``, so one run of a figure
script regenerates every declared task into its own directory. The investigation
figures (``ablation_estimators``, ``retriever_comparison``) ask a single one-off
question about a single experiment and keep naming it in their own file, because
their output path does not contain the prediction task.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass

from src.plotting.defaults import metric_scale, task_label


@dataclass(frozen=True)
class FigureTask:
    """One prediction task and the experiment that holds its evaluation runs."""

    target: str
    experiment: str
    metrics: tuple[str, ...]

    def __post_init__(self) -> None:
        scales = {metric_scale(metric) for metric in self.metrics}
        if len(scales) != 1:
            raise ValueError(
                f"Task {self.target!r} mixes metrics on different display scales: {self.metrics}"
            )

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


# One entry per prediction task a family rebuilds. Classification metrics are
# point-scale, so the percent scale of a task is derived from its metrics
# instead of being typed into every script.
FIGURE_TASKS: dict[str, tuple[FigureTask, ...]] = {
    "baseline": (FigureTask("mortality", "sample_size_mimic_mortality", ("roc_auc", "prc_auc")),),
    "sample_size": (FigureTask("mortality", "sample_size_mimic_mortality", ("roc_auc", "prc_auc")),),
    "pairwise_wins": (FigureTask("mortality", "sample_size_mimic_mortality", ("roc_auc", "prc_auc")),),
    "pairwise_tables": (FigureTask("mortality", "sample_size_mimic_mortality", ("roc_auc", "prc_auc")),),
}


def tasks_for(family: str, targets: Sequence[str] | None = None) -> tuple[FigureTask, ...]:
    """Return the declared tasks of one figure family, optionally filtered by target."""
    if family not in FIGURE_TASKS:
        raise KeyError(f"Unknown figure family {family!r}; declared families: {sorted(FIGURE_TASKS)}")
    declared = FIGURE_TASKS[family]
    if targets is None:
        return declared

    selected = tuple(task for task in declared if task.target in set(targets))
    unknown = set(targets) - {task.target for task in selected}
    if unknown:
        available = [task.target for task in declared]
        raise ValueError(f"Family {family!r} declares no target(s) {sorted(unknown)}; declared: {available}")
    return selected


def warn_skipped(task: FigureTask, reason: object) -> None:
    """Report a declared task whose artifacts do not exist yet without failing the run."""
    print(f"warning: skipping {task.label} ({task.experiment}): {reason}", file=sys.stderr)
