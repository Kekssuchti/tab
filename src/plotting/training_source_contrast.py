"""F3: same-target cost of external rather than local full-data training.

Regenerate with:

    uv run python -m src.plotting.training_source_contrast

Each target uses the reciprocal single-source experiments registered centrally
in ``src.plotting.experiments``. A target is skipped, without substitution, when
either complete-data direction is unavailable.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from matplotlib.figure import Figure

from src.config import config
from src.plotting.defaults import dataset_label, metric_label, set_plot_style, task_label
from src.plotting.experiments import (
    DATA_SOURCES,
    MAIN_TARGETS,
    PlotExperiment,
    reciprocal_single_source,
    tasks_for,
    warn_skipped,
)
from src.plotting.scientific_figstyle import BASELINE, WIDE, figure_grid, panel_labels, save
from src.plotting.utils import (
    MissingExperimentError,
    PlotArtifacts,
    SourceContrastEvaluation,
    load_plot_artifacts,
    prepare_source_contrast,
)
from src.plotting.utils.rendering import draw_model_forest, instance_plot_styles, interval_axis_limits


@dataclass(frozen=True)
class DataSettings:
    """Output location for reciprocal complete-data contrasts."""

    output_dir: Path = config.dir_plots / "training_source_contrast"


@dataclass(frozen=True)
class VisualSettings:
    """Locally editable presentation choices for F3."""

    metrics: tuple[str, ...] = ("roc_auc", "prc_auc")
    show_ci: bool = True
    ci_level: float = 0.95
    score_scale: float = 100.0
    figure_width: float = WIDE
    figure_height_ratio: float = 1.08
    axis_label: str = "Loss from external training (pp)"
    model_axis_label: str = "Model"
    shade_baselines: bool = True
    baseline_band_alpha: float = 0.13
    marker_size: float = 4.4
    ci_line_width: float = 1.0
    cap_size: float = 2.0
    axis_padding_fraction: float = 0.08
    output_formats: tuple[str, ...] = ("pdf",)


DATA = DataSettings()
VISUAL = VisualSettings()


def make_figure(data: SourceContrastEvaluation, visual: VisualSettings = VISUAL) -> Figure:
    """Draw metric rows and fixed evaluation-center columns."""
    set_plot_style()
    fig, axes = figure_grid(
        len(data.metrics),
        len(data.evaluation_centers),
        width=visual.figure_width,
        ratio=visual.figure_height_ratio,
        sharey=True,
        squeeze=False,
    )
    styles = instance_plot_styles(data.model_metadata)

    for metric_index, metric in enumerate(data.metrics):
        metric_rows = data.performance.loc[data.performance["metric"].eq(metric)]
        limits = interval_axis_limits(
            metric_rows,
            scale=visual.score_scale,
            include_zero=True,
            show_ci=visual.show_ci,
            padding_fraction=visual.axis_padding_fraction,
        )
        for center_index, center in enumerate(data.evaluation_centers):
            ax = axes[metric_index, center_index]
            rows = metric_rows.loc[metric_rows["evaluation_center"].eq(center)]
            draw_model_forest(
                ax,
                rows,
                data.model_instances,
                styles,
                scale=visual.score_scale,
                show_ci=visual.show_ci,
                show_model_labels=center_index == 0,
                shade_baselines=visual.shade_baselines,
                baseline_band_alpha=visual.baseline_band_alpha,
                marker_size=visual.marker_size,
                ci_line_width=visual.ci_line_width,
                cap_size=visual.cap_size,
            )
            ax.axvline(0, color=BASELINE, linewidth=0.8, linestyle="--", zorder=1)
            ax.set_xlim(limits)
            ax.set_xlabel(visual.axis_label)
            if center_index == 0:
                ax.set_ylabel(f"{metric_label(metric)}\n{visual.model_axis_label}")
            if metric_index == 0:
                ax.annotate(
                    f"Evaluation: {dataset_label(center)}",
                    xy=(0.5, 1),
                    xycoords="axes fraction",
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

    panel_labels(axes)
    return fig


def caption(data: SourceContrastEvaluation, visual: VisualSettings = VISUAL) -> str:
    """Return the self-contained F3 caption."""
    sources = data.evaluation_centers
    source_counts = " and ".join(
        f"{data.training_counts[source]:,} {dataset_label(source)} observations "
        f"({_repeat_text(data.run_count(source))})"
        for source in sources
    )
    centers = " and ".join(dataset_label(center) for center in data.evaluation_centers)
    metrics = " and ".join(metric_label(metric) for metric in data.metrics)
    confidence = round(100 * visual.ci_level)
    return (
        rf"\textbf{{Local rather than external full-data training has model-specific effects for "
        f"{task_label(data.target)}.}} Columns fix the held-out {centers} evaluation populations; rows show "
        f"{metrics}. Each point is score(the same model trained on the evaluation center's complete pool) minus "
        "score(the same model trained on the other center's complete pool), in percentage points, so positive "
        f"values favor local training and negative values favor external training. The complete pools contain "
        f"{source_counts}, selected only from runs explicitly configured with training fraction 1.0; this is an "
        f"all-available-data comparison, not an equal-budget source-effect estimate. Whiskers are {confidence}\\% "
        f"percentile intervals from {data.bootstrap_count:,} paired test-cohort bootstrap score differences after "
        "averaging repeated full-data runs within each training source by draw. They quantify resampling of each "
        "fixed held-out cohort, not training-repeat variation; cohort fingerprints, ordered cohort IDs and labels, "
        "split settings, bootstrap seed/count/IDs, and prediction-dataset order were required to match before "
        "pairing. Circles denote classical baselines and triangles tabular foundation models."
    )


def _repeat_text(count: int) -> str:
    return "one pipeline run" if count == 1 else f"{count} repeated pipeline runs"


def _pairing_report(data: SourceContrastEvaluation) -> None:
    print("Pairing validation:")
    print(
        f"  Models: the same {len(data.model_instances)} configured and successfully evaluated instances "
        "occur in every selected run from both sources; no exclusions or missing models."
    )
    for center in data.evaluation_centers:
        evidence = data.pairing_evidence[center]
        print(
            f"  {dataset_label(center)}: one cohort fingerprint across every reciprocal run "
            f"({evidence.cohort_fingerprint}); exact ordered IDs/labels for {evidence.cohort_size:,} held-out "
            f"observations; target={evidence.dataset_target}, dataset.random_state="
            f"{evidence.dataset_random_state}, dataset.train_size={evidence.dataset_train_size:g}, "
            f"evaluation_bootstrap_seed={evidence.evaluation_bootstrap_seed}; prediction schema "
            f"{evidence.prediction_schema_version} order={evidence.prediction_dataset_order}; exact "
            f"bootstrap IDs {evidence.bootstrap_id_first}..{evidence.bootstrap_id_last} "
            f"({evidence.bootstrap_count:,} draws) across both sources and metrics."
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        choices=MAIN_TARGETS,
        help="Prediction target to rebuild; repeat for several. Defaults to all three main targets.",
    )
    parser.add_argument("--output-dir", type=Path, default=DATA.output_dir)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    declared = tasks_for("training_source_contrast", args.targets)
    targets = args.targets or list(MAIN_TARGETS)
    for target in targets:
        tasks = tuple(task for task in declared if task.target == target)
        print(f"\n=== {task_label(target)} [reciprocal full-data source contrast]")
        pair = _reciprocal_pair(tasks)
        if pair is None:
            continue

        loaded = _load_pair(pair)
        if loaded is None:
            continue
        artifacts_by_source = loaded
        visual = replace(VISUAL, metrics=pair[0].metrics, score_scale=pair[0].score_scale)
        prepared = prepare_source_contrast(
            artifacts_by_source,
            metrics=visual.metrics,
            ci_level=visual.ci_level,
        )
        for source in DATA_SOURCES:
            print(
                f"Selected {dataset_label(source)} full-data runs "
                f"({prepared.training_counts[source]:,} observations): " + ", ".join(prepared.run_ids[source])
            )
        _pairing_report(prepared)

        output_dir = args.output_dir / prepared.target
        stem = output_dir / (
            f"training_source_contrast_reciprocal_{DATA_SOURCES[0]}_{DATA_SOURCES[1]}_{prepared.target}"
        )
        outputs = save(make_figure(prepared, visual), str(stem), formats=visual.output_formats)
        print("LaTeX caption:")
        print(f"\\caption{{{caption(prepared, visual)}}}")
        for output in outputs:
            print("figure: " + str(Path(output).relative_to(config.dir_root)))


def _reciprocal_pair(tasks: tuple[PlotExperiment, ...]) -> tuple[PlotExperiment, PlotExperiment] | None:
    if len(tasks) != len(DATA_SOURCES):
        raise RuntimeError(f"Expected two single-source declarations for F3; found {len(tasks)}")
    first = next((task for task in tasks if task.training_source == DATA_SOURCES[0]), None)
    if first is None:
        raise RuntimeError(f"No {DATA_SOURCES[0]!r} single-source declaration is available")
    reciprocal = reciprocal_single_source(first)
    if reciprocal not in tasks:
        raise RuntimeError(f"Central registry returned an undeclared reciprocal input: {reciprocal}")

    unavailable = [task for task in (first, reciprocal) if task.experiment_name is None]
    if unavailable:
        for task in unavailable:
            warn_skipped(
                task,
                "F3 reciprocal input is unregistered; skipping the target without substituting another run",
            )
        available = [task for task in (first, reciprocal) if task.experiment_name is not None]
        for task in available:
            print(
                f"warning: registered counterpart {task.experiment_name!r} is not plotted because its "
                "reciprocal source is unavailable",
                file=sys.stderr,
            )
        return None
    return first, reciprocal


def _load_pair(
    pair: tuple[PlotExperiment, PlotExperiment],
) -> dict[str, PlotArtifacts] | None:
    loaded = {}
    for task in pair:
        assert task.experiment_name is not None
        assert task.training_source is not None
        try:
            loaded[task.training_source] = load_plot_artifacts(
                task.experiment_name,
                full_training_only=True,
                expected_target=task.target,
                expected_training_source=task.training_source,
            )
        except MissingExperimentError as missing:
            warn_skipped(
                task,
                f"F3 reciprocal complete-data input is unavailable ({missing}); "
                "skipping the target without substitution",
            )
            return None
    return loaded


if __name__ == "__main__":
    main()
