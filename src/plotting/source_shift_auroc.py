"""Cross-source AUROC changes across the three thesis tasks.

Regenerate from the project MLflow store with:
    uv run python -m src.plotting.source_shift_auroc

The source experiments, model selection, pairing, aggregation, and uncertainty
calculation are explicit below. The script writes one publication PDF and
prints its LaTeX caption to the console.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.config import config
from src.mlflow.evaluation_data import list_pipeline_runs, load_evaluation_data
from src.plotting.defaults import (
    BASELINE_MARKER,
    TASK_COLORS,
    model_label,
    model_styles,
    ordered_models,
    ordered_tasks,
    set_plot_style,
    task_label,
)
from src.plotting.scientific_figstyle import (
    BASELINE,
    MUTED,
    WIDE,
    figure,
    mean_ci,
    save,
)

OUTPUT_STEM = config.dir_plots / "generalizability" / "source_shift_auroc"
RUN_COUNT = 5

# Experiment identity belongs to this figure; task ordering and presentation
# come from the canonical plotting defaults.
EXPERIMENT_BY_TASK = {
    "mortality": "tudd_baseline_mortality",
    "LOS7": "tudd_baseline_LOS7",
    "hours_to_readmit_72": "tudd_baseline_hours_to_readmit_72",
}
TASKS = tuple(ordered_tasks(EXPERIMENT_BY_TASK))

CAPTION_BODY = (
    r"\textbf{Cross-source AUROC degradation is task dependent and is not "
    r"uniformly reduced by tabular foundation models.} All models were trained "
    r"on EUH. Points show mean MIMIC-IV minus EUH AUROC across five repeated "
    r"pipeline runs; whiskers are 95\% t intervals over run-level differences. "
    r"Held-out cohorts were fixed, so intervals quantify run variation rather "
    r"than cohort-sampling uncertainty. Circles denote classical baselines and "
    r"triangles tabular foundation models."
)


def load_run_level_differences() -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Load paired scores and return models complete across all tasks/runs."""
    task_frames: list[pd.DataFrame] = []

    for task_key in TASKS:
        task_name = task_label(task_key)
        experiment_name = EXPERIMENT_BY_TASK[task_key]
        pipeline_runs = list_pipeline_runs(experiment_name)
        if len(pipeline_runs) != RUN_COUNT:
            raise ValueError(
                f"Expected {RUN_COUNT} baseline pipeline runs in "
                f"{experiment_name!r}, found {len(pipeline_runs)}. Select the "
                "intended run IDs explicitly before regenerating this figure."
            )

        data = load_evaluation_data(
            experiment_names=experiment_name,
            pipeline_runs=pipeline_runs["mlflow_run_id"],
        )
        selected = data.loc[
            data["scope"].eq("test")
            & data["statistic"].eq("point")
            & data["dataset"].isin(("tudd", "mimic")),
            [
                "pipeline_mlflow_run_id",
                "pipeline_run_name",
                "model_name",
                "model_instance",
                "trained_on",
                "dataset",
                "roc_auc",
            ],
        ].copy()

        if selected.empty or selected["roc_auc"].isna().any():
            raise ValueError(f"Incomplete AUROC measurements in {experiment_name!r}")
        if not selected["trained_on"].eq("tudd").all():
            raise ValueError(f"Figure requires EUH-trained models in {experiment_name!r}")
        index_columns = [
            "pipeline_mlflow_run_id",
            "pipeline_run_name",
            "model_name",
            "model_instance",
            "trained_on",
        ]
        paired = selected.pivot(
            index=index_columns,
            columns="dataset",
            values="roc_auc",
        ).reset_index()
        if not {"tudd", "mimic"}.issubset(paired.columns):
            raise ValueError(f"Both EUH and MIMIC-IV scores are required in {experiment_name!r}")
        if paired[["tudd", "mimic"]].isna().any().any():
            raise ValueError(f"Unpaired EUH/MIMIC-IV scores in {experiment_name!r}")

        paired["task"] = task_name
        paired["delta_auroc_points"] = 100.0 * (paired["mimic"] - paired["tudd"])
        task_frames.append(paired)

    differences = pd.concat(task_frames, ignore_index=True)
    task_names = [task_label(task_key) for task_key in TASKS]
    counts = (
        differences.groupby(["model_name", "task"], sort=False)
        .size()
        .unstack("task", fill_value=0)
        .reindex(columns=task_names, fill_value=0)
    )
    complete_models = counts.index[counts.eq(RUN_COUNT).all(axis=1)].tolist()
    models = tuple(ordered_models(complete_models))
    if not models:
        raise ValueError("No model has five paired EUH/MIMIC-IV runs for every task")

    omitted = ordered_models(sorted(set(differences["model_name"]) - set(models)))
    if omitted:
        print("Omitting models without five complete runs per task: " + ", ".join(omitted))
    differences = differences.loc[differences["model_name"].isin(models)].copy()
    return differences, models


def summarize_differences(
    differences: pd.DataFrame,
    models: tuple[str, ...],
) -> pd.DataFrame:
    """Calculate mean and 95% t interval across repeated pipeline runs."""
    rows: list[dict[str, str | int | float]] = []
    for task_key in TASKS:
        task_name = task_label(task_key)
        task_rows = differences.loc[differences["task"].eq(task_name)]
        for model_name in models:
            values = task_rows.loc[task_rows["model_name"].eq(model_name), "delta_auroc_points"].to_numpy(dtype=float)
            mean, lower, upper = mean_ci(values)
            rows.append(
                {
                    "task": task_name,
                    "model_name": model_name,
                    "model_label": model_label(model_name),
                    "n_runs": len(values),
                    "mean_delta_auroc_points": float(mean),
                    "ci95_lower": float(lower),
                    "ci95_upper": float(upper),
                }
            )
    return pd.DataFrame(rows)


def make_figure(summary: pd.DataFrame, models: tuple[str, ...]):
    """Draw one cross-task panel with stable model-family markers."""
    set_plot_style()
    fig, ax = figure(width=WIDE, ratio=0.58)

    model_y = np.arange(len(models), dtype=float)
    task_offsets = np.linspace(-0.22, 0.22, len(TASKS))
    styles = model_styles(models)
    task_handles: list[Line2D] = []

    for offset, task_key in zip(task_offsets, TASKS, strict=True):
        task_name = task_label(task_key)
        color = TASK_COLORS[task_key]
        rows = summary.loc[summary["task"].eq(task_name)].set_index("model_name").loc[list(models)]
        for model_index, model_name in enumerate(models):
            row = rows.loc[model_name]
            mean = float(row["mean_delta_auroc_points"])
            lower = float(row["ci95_lower"])
            upper = float(row["ci95_upper"])
            ax.errorbar(
                mean,
                model_y[model_index] + offset,
                xerr=np.array([[mean - lower], [upper - mean]]),
                fmt=styles[model_name].marker,
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=2.0,
                capthick=0.8,
                markersize=4.2,
                markeredgecolor="white",
                markeredgewidth=0.45,
                zorder=3,
            )
        task_handles.append(Line2D([], [], color=color, linewidth=1.8, label=task_name))

    ax.axvline(0, color=BASELINE, linewidth=0.8, linestyle="--", zorder=1)
    baseline_count = sum(styles[name].marker == BASELINE_MARKER for name in models)
    if 0 < baseline_count < len(models):
        ax.axhspan(-0.5, baseline_count - 0.5, color=MUTED, alpha=0.16, zorder=0)
        ax.axhline(
            baseline_count - 0.5,
            color=BASELINE,
            linewidth=0.5,
            alpha=0.65,
            zorder=1,
        )

    ax.set_yticks(model_y, [model_label(name) for name in models])
    ax.set_ylim(len(models) - 0.45, -0.55)
    ax.set_xlabel("Δ AUROC (MIMIC-IV − EUH, percentage points)")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    finite_bounds = summary[["ci95_lower", "ci95_upper"]].to_numpy(dtype=float)
    lower_limit = 5 * np.floor((np.nanmin(finite_bounds) - 0.5) / 5)
    upper_limit = 5 * np.ceil((np.nanmax(finite_bounds) + 0.5) / 5)
    ax.set_xlim(lower_limit, upper_limit)
    ax.set_xticks(np.arange(lower_limit, upper_limit + 0.1, 5))

    ax.legend(
        handles=task_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=len(TASKS),
        borderaxespad=0,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    return fig


def main() -> None:
    differences, models = load_run_level_differences()
    summary = summarize_differences(differences, models)

    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)

    figure_object = make_figure(summary, models)
    outputs = save(figure_object, str(OUTPUT_STEM), formats=("pdf",))
    print("LaTeX caption:")
    print(f"\\caption{{{CAPTION_BODY}}}")
    print("figure: " + str(Path(outputs[0]).relative_to(config.dir_root)))


if __name__ == "__main__":
    main()
