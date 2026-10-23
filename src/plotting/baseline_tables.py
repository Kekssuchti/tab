"""Full-sample reciprocal train/test tables, using the baseline plots' inputs.

Regenerate: uv run python -m src.plotting.baseline_tables
Select PLOT_METRIC=auroc|auprc as for figures. Always includes all models.
Use --ranges on|off|both (--ci is an alias); default: on.
Writes tables/baseline/<metric>/<target>_baseline.tex with ranges and
<target>_baseline_no_ranges.tex without ranges, with distinct LaTeX labels.
Only point metrics are loaded: ranges describe training repeats, not bootstrap CIs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mlflow import MlflowClient
from src.config import config
from src.mlflow.tracking_contract import ARTIFACT_CONFIG, ARTIFACT_DATASET_SUMMARY, ARTIFACT_TEST_PREDICTIONS
from src.plotting.defaults import (
    dataset_label,
    metric_label,
    metric_scale,
    ordered_datasets,
    task_label,
)
from src.plotting.experiments import tasks_for, warn_skipped
from src.plotting.model_sets import METRIC_SET, PLOT_METRIC
from src.plotting.utils.artifacts import MissingExperimentError, PlotArtifacts, load_plot_artifacts
from src.plotting.utils.rendering import instance_plot_styles
from src.plotting.utils.runs import read_json_artifact, single_value, validate_model_identity, validated_point_rows
from src.plotting.utils.sample_size import full_training_count
from src.utils.prediction_tables import PREDICTION_MANIFEST_FILENAME


@dataclass(frozen=True)
class BaselineTable:
    target: str
    sources: tuple[str, ...]
    model_labels: dict[str, str]
    performance: dict[str, pd.DataFrame]
    deltas: dict[str, pd.DataFrame]
    training_counts: dict[str, int]
    test_counts: dict[str, int]
    artifacts: dict[str, PlotArtifacts]


def prepare_table(artifacts: dict[str, PlotArtifacts], metric: str) -> BaselineTable:
    """Average complete logical repeats, preserving the canonical shard merging.

    Validate ordinary reciprocal cohorts and recorded cohort provenance without
    reading/exporting patient-level prediction rows. This is not an exact-ID or
    paired-bootstrap comparison between training sources.
    """
    sources = tuple(ordered_datasets(list(artifacts)))
    if set(sources) != {"mimic", "tudd"}:
        raise ValueError("A train/test matrix requires both MIMIC and TUDD training inputs")
    client = MlflowClient()
    performance, deltas, training_counts, test_counts = {}, {}, {}, {}
    targets, cohort_signatures = set(), {}
    model_labels, model_names = None, None
    for source in sources:
        data = artifacts[source]
        validated = validated_point_rows(data, metrics=(metric,), datasets=sources, description="baseline table")
        points = validated.points
        targets.add(validated.target)
        if single_value(points, "trained_on", "baseline table") != source:
            raise ValueError(f"Training input for {source} records another source")
        if points[["model_name", "model_instance"]].isna().any().any():
            raise ValueError("Baseline model identities must not be missing")
        validate_model_identity(points)
        names = points.drop_duplicates("model_instance").set_index("model_instance")["model_name"].to_dict()
        labels = {instance: label for instance, (_, label) in instance_plot_styles(points).items()}
        if model_labels is None:
            model_labels, model_names = labels, names
        elif model_names != names:
            raise ValueError("Reciprocal inputs must use the same model instances and names")
        keys = ["pipeline_mlflow_run_id", "model_instance", "dataset"]
        expected = pd.MultiIndex.from_product([data.run_ids, list(labels), sources], names=keys)
        observed = pd.MultiIndex.from_frame(points[keys])
        if observed.has_duplicates or len(expected.difference(observed)) or len(observed.difference(expected)):
            raise ValueError(f"Incomplete or duplicate run/model/test cells for {source}")
        points[metric] = pd.to_numeric(points[metric], errors="coerce")
        if not np.isfinite(points[metric]).all() or not points[metric].between(0, 1).all():
            raise ValueError(f"{metric} must contain finite scores between zero and one")
        training_counts[source] = full_training_count(data)
        for logical_id in data.run_ids:
            for physical_id in data.physical_run_ids(logical_id):
                summary = read_json_artifact(client, physical_id, ARTIFACT_DATASET_SUMMARY)
                recorded = read_json_artifact(client, physical_id, ARTIFACT_CONFIG)
                manifest = read_json_artifact(
                    client, physical_id, f"{ARTIFACT_TEST_PREDICTIONS}/{PREDICTION_MANIFEST_FILENAME}"
                )
                if str(manifest.get("version")) not in {"1", "2"} or set(manifest.get("datasets", sources)) != set(
                    sources
                ):
                    raise ValueError(f"Run {physical_id} has an unsupported ordinary-cohort manifest")
                dataset = recorded["dataset"]
                if dataset.get("custom_retriever") is not None:
                    raise ValueError("Retriever runs cannot supply ordinary baseline tables")
                if summary["target"] != validated.target or dataset["target"] != validated.target:
                    raise ValueError(f"Run {physical_id} records conflicting targets")
                if summary["train"]["row_count"] != training_counts[source]:
                    raise ValueError(f"Run {physical_id} records conflicting training counts")
                for center in sources:
                    count = summary[f"test_{center}"]["row_count"]
                    if type(count) is not int or count <= 0:
                        raise ValueError(f"Run {physical_id} records an invalid test count for {center}")
                    fingerprint = manifest["cohort_fingerprints"][center]
                    if not isinstance(fingerprint, str) or not fingerprint.strip():
                        raise ValueError(f"Run {physical_id} has no recorded {center} cohort fingerprint")
                    signature = (
                        count,
                        fingerprint,
                        dataset["random_state"],
                        dataset["train_size"],
                    )
                    if center in cohort_signatures and cohort_signatures[center] != signature:
                        raise ValueError(f"Reciprocal baseline runs disagree on recorded {center} test provenance")
                    cohort_signatures[center] = signature
                    test_counts[center] = count
        means = points.groupby(["model_instance", "dataset"], sort=False)[metric].agg(
            estimate="mean", lower="min", upper="max"
        )
        performance[source] = means
        external = next(center for center in sources if center != source)
        transferred = means.xs(external, level="dataset")["estimate"]
        # As in prepare_transfer_summary: best transferred mean in this direction
        # and selected roster, not the best local model or a per-repeat winner.
        deltas[source] = pd.DataFrame(
            {
                "delta_spec": transferred - means.xs(source, level="dataset")["estimate"],
                "delta_comp": transferred - transferred.max(),
            }
        )
    if len(targets) != 1:
        raise ValueError("Reciprocal baseline inputs contain different targets")
    return BaselineTable(
        target=targets.pop(),
        sources=sources,
        model_labels=model_labels,
        performance=performance,
        deltas=deltas,
        training_counts=training_counts,
        test_counts=test_counts,
        artifacts=artifacts,
    )


def caption(data: BaselineTable, metric: str, *, include_ranges: bool = True) -> str:
    counts = {source: len(data.artifacts[source].run_ids) for source in data.sources}
    if len(set(counts.values())) == 1:
        count = next(iter(counts.values()))
        number = {1: "one", 2: "two", 3: "three"}.get(count, str(count))
        repeats = f"{number} training repeat{'s' if count != 1 else ''}"
    else:
        repeats = "training repeats (" + "; ".join(f"{dataset_label(s)}: {n}" for s, n in counts.items()) + ")"
    scores = "mean [minimum, maximum]" if include_ranges and max(counts.values()) > 1 else "means"
    single_repeat = (
        " A single repeat is shown without a range."
        if include_ranges and min(counts.values()) == 1 < max(counts.values())
        else ""
    )
    return (
        rf"\textbf{{Full-sample {metric_label(metric)} train/test comparison for {task_label(data.target)}.}} "
        f"Scores are {scores} across {repeats}.{single_repeat} "
        "Grey cells denote external evaluations. "
        "Within each training block, bold marks the highest mean in each test column and the largest delta."
    )


def _number(value: float, metric: str) -> str:
    rounded = round(float(value) * metric_scale(metric), 2)
    return f"{0.0 if rounded == 0 else rounded:.2f}"


def _score(
    data: BaselineTable, source: str, center: str, instance: str, metric: str, *, include_ranges: bool = True
) -> str:
    row = data.performance[source].loc[(instance, center)]
    value = _number(row.estimate, metric)
    if include_ranges and len(data.artifacts[source].run_ids) > 1:
        value += f" [{_number(row.lower, metric)}, {_number(row.upper, metric)}]"
    return value


def _best(data: BaselineTable, source: str, column: str, instance: str, *, delta: bool = False) -> bool:
    values = data.deltas[source][column] if delta else data.performance[source].xs(column, level="dataset")["estimate"]
    return bool(np.isclose(values.loc[instance], values.max(), rtol=0, atol=1e-12))


def _escape(text: str) -> str:
    return "".join(
        {
            "\\": r"\textbackslash{}",
            "_": r"\_",
            "%": r"\%",
            "&": r"\&",
            "#": r"\#",
            "$": r"\$",
            "{": r"\{",
            "}": r"\}",
        }.get(char, char)
        for char in text
    )


def table_to_latex(data: BaselineTable, metric: str, *, include_ranges: bool = True) -> str:
    """Complete table fragment; no document class, preamble, or resizebox."""
    label_suffix = "" if include_ranges else "-no-ranges"
    lines = [
        r"% Requires \usepackage{booktabs,graphicx,multirow} and \usepackage[table]{xcolor}.",
        r"% Source: MLflow prediction_metrics.csv; only scope=test, statistic=point.",
        r"% Ranges, when shown, are training-repeat minima/maxima, not bootstrap confidence intervals.",
        r"% Scores are percentages; deltas are percentage points.",
    ]
    for source in data.sources:
        for run in data.artifacts[source].run_ids:
            lines.append(
                f"% {source} logical repeat: {run}; physical runs: "
                + ", ".join(data.artifacts[source].physical_run_ids(run))
            )
    lines += [
        r"\begin{table}[htb]",
        r"  \centering",
        rf"  \caption{{{caption(data, metric, include_ranges=include_ranges)}}}",
        rf"  \label{{tab:baseline-{data.target.replace('_', '-')}-{METRIC_SET}{label_suffix}}}",
        "",
        r"  \small",
        r"  \setlength{\tabcolsep}{2pt}",
        # r"  \renewcommand{\arraystretch}{1.18}",
        r"  \newcommand{\ood}{\cellcolor[gray]{0.92}}",
        "",
        # Tabular resets baselineskip to zero, so freeze the group offset first.
        r"  \begin{tabular}{clcccc}",
        r"    & & \multicolumn{2}{c}{Test set} & & \\",
        r"    \cmidrule(lr){3-4}",
        r"    Train set & Model & "
        "\n    "
        + " & ".join(
            rf"\shortstack{{{dataset_label(s)}\\($N={data.test_counts[s]:,}$)}}" + " \n    " for s in data.sources
        )
        + r" & $\Delta_{\mathrm{spec}}$ "
        + "\n    "
        + r" & $\Delta_{\mathrm{comp}}$ \\"
        + "\n",
        r"    \midrule",
    ]

    for block, source in enumerate(data.sources):
        if block:
            lines.append(r"    \midrule")
        # Center the rotated group label without adding a multirow dependency.
        train = (
            rf"\multirow{{{len(data.model_labels.values())}}}{{*}}"
            + rf"{{\rotatebox[origin=c]{{90}}{{\shortstack{{{dataset_label(source)}\\($N={data.training_counts[source]:,}$)}} }} }}"
            + "\n    "
        )
        for index, (instance, label) in enumerate(data.model_labels.items()):
            cells = [train if index == 0 else "", _escape(label)]
            for center in data.sources:
                value = _score(data, source, center, instance, metric, include_ranges=include_ranges)
                cell = rf"\textbf{{{value}}}" if _best(data, source, center, instance) else value
                if center != source:
                    cell = r"\ood " + cell
                cells.append(cell)
            for column in ("delta_spec", "delta_comp"):
                value = _number(data.deltas[source].loc[instance, column], metric)
                cells.append(rf"\textbf{{{value}}}" if _best(data, source, column, instance, delta=True) else value)
            lines.append("    " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append", dest="targets")
    parser.add_argument("--output-dir", type=Path, default=config.dir_root / "tables" / "baseline")
    parser.add_argument("--stdout", action="store_true", help="Also print generated LaTeX")
    parser.add_argument(
        "--ranges",
        "--ci",
        choices=("on", "off", "both"),
        default="on",
        help="Show training-repeat min/max ranges, omit them, or save both versions (default: on; not bootstrap CIs)",
    )
    args = parser.parse_args()
    tasks = tasks_for("baseline", args.targets)
    for target in dict.fromkeys(task.target for task in tasks):
        artifacts = {}
        for task in (task for task in tasks if task.target == target):
            if task.experiment_name is None:
                warn_skipped(task, "no experiment registered; a complete reciprocal table cannot be built")
                break
            try:
                artifacts[task.training_source] = load_plot_artifacts(
                    task.experiment_name,
                    full_training_only=True,
                    include_bootstrap=False,
                    apply_model_view=False,
                    expected_target=target,
                    expected_training_source=task.training_source,
                )
            except MissingExperimentError as missing:
                warn_skipped(task, missing)
                break
        if len(artifacts) != 2:
            continue
        data = prepare_table(artifacts, PLOT_METRIC)
        output = args.output_dir / METRIC_SET
        output.mkdir(parents=True, exist_ok=True)
        variants = (True, False) if args.ranges == "both" else (args.ranges == "on",)
        for include_ranges in variants:
            suffix = "" if include_ranges else "_no_ranges"
            path = output / f"{target}_baseline{suffix}.tex"
            latex = table_to_latex(data, PLOT_METRIC, include_ranges=include_ranges)
            path.write_text(latex, encoding="utf-8")
            print(f"Table: {path}")
            if args.stdout:
                print(latex)


if __name__ == "__main__":
    main()
