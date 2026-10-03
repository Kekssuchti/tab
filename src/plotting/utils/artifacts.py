"""Load and select canonical evaluation artifacts for figure scripts."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from mlflow.exceptions import MlflowException

from mlflow import MlflowClient
from src.mlflow.evaluation_data import DEFAULT_TRACKING_URI, load_bootstrap_data, load_evaluation_data
from src.mlflow.tracking_contract import ARTIFACT_CONFIG


class MissingExperimentError(ValueError):
    """Raised when a declared figure task has no evaluation artifacts yet."""


class MissingFullTrainingRunError(MissingExperimentError):
    """Raised when an experiment has no explicit single-source fraction-1.0 run."""


class TargetMismatchError(ValueError):
    """Raised when an experiment does not contain the prediction task it is declared under."""


@dataclass(frozen=True)
class PlotArtifacts:
    """Metric and bootstrap artifacts selected for one figure."""

    metrics: pd.DataFrame
    bootstrap_scores: pd.DataFrame
    experiment_name: str
    run_ids: tuple[str, ...]


def load_plot_artifacts(
    experiment_name: str,
    *,
    pipeline_runs: str | Sequence[str] | None = None,
    models: str | Sequence[str] | None = None,
    exclude_models: str | Sequence[str] | None = None,
    full_training_only: bool = False,
    include_bootstrap: bool = True,
    expected_target: str | None = None,
    expected_training_source: str | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> PlotArtifacts:
    """Load plotting artifacts and resolve their concrete pipeline run IDs.

    When ``full_training_only`` is true, every retained run must record exactly
    one ``dataset.train_on`` entry in ``config.json``, and that entry's fraction
    must be the JSON float ``1.0``. With no explicit selector, all such runs are
    selected. With ``pipeline_runs``, every selected run is validated against
    the same full-data contract; largest observed training size is never used as
    a substitute.

    ``exclude_models`` drops models by name from every returned frame, point
    metrics and bootstrap columns alike, so a figure can leave out a model
    without touching the run artifacts. It is the way to answer "the same figure
    without LimiX" for any script that loads artifacts through here.

    ``expected_target`` names the prediction task the caller intends to plot and
    is checked against the loaded artifacts, so a stale experiment fails loudly
    instead of quietly producing the same figure for another task.

    Complementary model-only runs with the same recorded data configuration and
    random states are combined in memory before the artifacts are returned.
    """
    metrics = load_evaluation_data(
        experiment_name,
        pipeline_runs=pipeline_runs,
        models=models,
        tracking_uri=tracking_uri,
    )
    if metrics.empty:
        raise MissingExperimentError(f"No evaluation data found for experiment {experiment_name!r}")
    if expected_target is not None:
        _require_target(metrics, experiment_name, expected_target)
    if expected_training_source is not None:
        _require_training_source(metrics, experiment_name, expected_training_source)

    dropped_instances: tuple[str, ...] = ()
    if exclude_models is not None:
        metrics, dropped_instances = _exclude_models(metrics, exclude_models)

    if full_training_only:
        candidate_run_ids = tuple(metrics["pipeline_mlflow_run_id"].astype(str).drop_duplicates())
        run_ids = select_single_source_full_data_run_ids(metrics, tracking_uri=tracking_uri)
        if pipeline_runs is not None:
            rejected = [run_id for run_id in candidate_run_ids if run_id not in set(run_ids)]
            if rejected:
                raise ValueError(
                    "Explicit full-data baseline run(s) do not record exactly one dataset.train_on "
                    "entry with fraction equal to float 1.0 in config.json: "
                    + ", ".join(rejected)
                )
        elif not run_ids:
            raise MissingFullTrainingRunError(
                f"Experiment {experiment_name!r} has no run whose config.json records exactly one "
                "dataset.train_on entry with fraction equal to float 1.0"
            )
        metrics = metrics.loc[metrics["pipeline_mlflow_run_id"].astype(str).isin(run_ids)].copy()
    else:
        run_ids = tuple(metrics["pipeline_mlflow_run_id"].astype(str).drop_duplicates())

    if not include_bootstrap:
        return combine_complementary_runs(
            PlotArtifacts(
                metrics=metrics,
                bootstrap_scores=pd.DataFrame(),
                experiment_name=experiment_name,
                run_ids=run_ids,
            ),
            tracking_uri=tracking_uri,
        )

    bootstrap_scores = load_bootstrap_data(
        experiment_name,
        pipeline_runs=run_ids,
        models=models,
        tracking_uri=tracking_uri,
    )
    if bootstrap_scores.empty:
        raise ValueError("No bootstrap_metrics.csv artifacts found for the selected pipeline runs")
    if dropped_instances:
        bootstrap_scores = _drop_instances(bootstrap_scores, dropped_instances)
    bootstrap_run_ids = set(bootstrap_scores["pipeline_mlflow_run_id"].astype(str))
    if bootstrap_run_ids != set(run_ids):
        raise ValueError(
            "Metric and bootstrap artifacts contain different pipeline runs: "
            f"metrics={list(run_ids)}, bootstrap={sorted(bootstrap_run_ids)}"
        )

    return combine_complementary_runs(
        PlotArtifacts(
            metrics=metrics,
            bootstrap_scores=bootstrap_scores,
            experiment_name=experiment_name,
            run_ids=run_ids,
        ),
        tracking_uri=tracking_uri,
    )


def combine_complementary_runs(
    artifacts: PlotArtifacts,
    *,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> PlotArtifacts:
    """Combine separately evaluated model subsets into complete logical runs.

    A later pipeline run may evaluate only a model that was absent from an older
    run. When both runs have the same data configuration and random states, they
    are shards of one logical repeat rather than independent repeats. This helper
    identifies those shards from their recorded ``config.json`` files, combines
    their point and bootstrap rows in memory, and keeps only logical repeats with
    the experiment's complete model roster.

    Physical runs are merged only when their model sets are disjoint. Exact
    reruns with the same model set remain separate repeats, while partially
    overlapping shards are rejected as ambiguous. No MLflow artifact is changed.
    """
    points = artifacts.metrics.loc[
        artifacts.metrics["scope"].eq("test") & artifacts.metrics["statistic"].eq("point")
    ].copy()
    if points.empty or len(artifacts.run_ids) < 2:
        return artifacts

    run_column = "pipeline_mlflow_run_id"
    points[run_column] = points[run_column].astype(str)
    model_sets = points.groupby(run_column, sort=False)["model_instance"].agg(lambda values: set(map(str, values)))
    if len({frozenset(models) for models in model_sets}) == 1:
        return artifacts

    client = MlflowClient(tracking_uri=tracking_uri)
    signatures = {run_id: _logical_run_signature(client, run_id) for run_id in artifacts.run_ids}
    groups: dict[str, list[str]] = {}
    for run_id in artifacts.run_ids:
        groups.setdefault(signatures[str(run_id)], []).append(str(run_id))

    replacement = {str(run_id): str(run_id) for run_id in artifacts.run_ids}
    merged_groups: list[tuple[str, ...]] = []
    for run_ids in groups.values():
        if len(run_ids) < 2:
            continue
        sets = [model_sets.get(run_id, set()) for run_id in run_ids]
        if all(model_set == sets[0] for model_set in sets[1:]):
            continue
        overlaps = [left & right for index, left in enumerate(sets) for right in sets[index + 1 :] if left & right]
        if overlaps:
            raise ValueError(
                "Runs with the same data configuration and random states have partially overlapping model sets; "
                f"cannot combine them unambiguously: {run_ids}"
            )
        _require_matching_bootstrap_plan(artifacts.bootstrap_scores, run_ids)
        representative = run_ids[0]
        replacement.update({run_id: representative for run_id in run_ids})
        merged_groups.append(tuple(run_ids))

    metrics = artifacts.metrics.copy()
    bootstraps = artifacts.bootstrap_scores.copy()
    for frame in (metrics, bootstraps):
        if not frame.empty:
            frame[run_column] = frame[run_column].astype(str).map(replacement)
    logical_run_ids = tuple(dict.fromkeys(replacement[str(run_id)] for run_id in artifacts.run_ids))

    logical_points = metrics.loc[metrics["scope"].eq("test") & metrics["statistic"].eq("point")].copy()
    roster = set(logical_points["model_instance"].astype(str))
    logical_sets = logical_points.groupby(run_column, sort=False)["model_instance"].agg(lambda values: set(map(str, values)))
    complete_run_ids = tuple(run_id for run_id in logical_run_ids if logical_sets.get(run_id, set()) == roster)
    if not complete_run_ids:
        raise ValueError(
            "No complete logical run remains after combining complementary model runs; "
            f"expected model roster {sorted(roster)}"
        )
    dropped = tuple(run_id for run_id in logical_run_ids if run_id not in set(complete_run_ids))
    if dropped:
        metrics = metrics.loc[metrics[run_column].astype(str).isin(complete_run_ids)].copy()
        if not bootstraps.empty:
            bootstraps = bootstraps.loc[bootstraps[run_column].astype(str).isin(complete_run_ids)].copy()

    if merged_groups:
        examples = [" + ".join(run_id[:8] for run_id in group) for group in merged_groups[:3]]
        suffix = f" (+{len(merged_groups) - len(examples)} more)" if len(merged_groups) > len(examples) else ""
        print(
            f"  note: combined {len(merged_groups)} complementary MLflow run group(s): "
            + "; ".join(examples)
            + suffix,
            file=sys.stderr,
        )
    if dropped:
        examples = [run_id[:8] for run_id in dropped[:3]]
        suffix = f" (+{len(dropped) - len(examples)} more)" if len(dropped) > len(examples) else ""
        print(
            f"  note: ignored {len(dropped)} incomplete logical repeat(s) without the full model roster: "
            + ", ".join(examples)
            + suffix,
            file=sys.stderr,
        )

    return PlotArtifacts(
        metrics=metrics,
        bootstrap_scores=bootstraps,
        experiment_name=artifacts.experiment_name,
        run_ids=complete_run_ids,
    )


def _logical_run_signature(client: MlflowClient, run_id: str) -> str:
    try:
        path = Path(client.download_artifacts(run_id, ARTIFACT_CONFIG))
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (MlflowException, OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Run {run_id} has no readable valid {ARTIFACT_CONFIG}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"Run {run_id} {ARTIFACT_CONFIG} must contain an object")
    identity = {key: value for key, value in payload.items() if key not in {"training", "mlflow", "run_id"}}
    return json.dumps(_without_none(identity), sort_keys=True, separators=(",", ":"))


def _without_none(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _without_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_without_none(item) for item in value]
    return value


def _require_matching_bootstrap_plan(bootstraps: pd.DataFrame, run_ids: Sequence[str]) -> None:
    if bootstraps.empty:
        return
    required = {"pipeline_mlflow_run_id", "dataset", "metric", "bootstrap_id", "model_instance"}
    _require_columns(bootstraps, required, "bootstrap scores")
    rows = bootstraps.loc[bootstraps["pipeline_mlflow_run_id"].astype(str).isin(run_ids)].copy()
    rows["bootstrap_id"] = pd.to_numeric(rows["bootstrap_id"], errors="coerce")
    if rows["bootstrap_id"].isna().any():
        raise ValueError("Complementary runs contain non-numeric bootstrap IDs")
    plans = {}
    for run_id, run_rows in rows.groupby("pipeline_mlflow_run_id", sort=False):
        cells = run_rows.groupby(["dataset", "metric", "model_instance"], sort=False)["bootstrap_id"].agg(
            lambda values: tuple(sorted(values.tolist()))
        )
        distinct = set(cells.tolist())
        if len(distinct) != 1:
            raise ValueError(f"Run {run_id} uses different bootstrap IDs across its model/evaluation cells")
        plans[str(run_id)] = next(iter(distinct))
    if set(plans) != set(run_ids) or len(set(plans.values())) != 1:
        raise ValueError(f"Complementary runs do not share one aligned bootstrap plan: {list(run_ids)}")


def _require_target(metrics: pd.DataFrame, experiment_name: str, expected_target: str) -> None:
    """Fail loudly when an experiment holds a different prediction task than declared."""
    if "target" not in metrics.columns:
        raise TargetMismatchError(
            f"Experiment {experiment_name!r} has no target column, so it cannot be checked "
            f"against the declared target {expected_target!r}"
        )
    found = sorted(set(metrics["target"].astype(str)))
    if found != [expected_target]:
        raise TargetMismatchError(
            f"Experiment {experiment_name!r} contains target {found}, but the figure declares "
            f"{expected_target!r}; point that figure task at its own experiment instead"
        )


def _require_training_source(metrics: pd.DataFrame, experiment_name: str, expected_source: str) -> None:
    """Fail loudly when an experiment is registered under the wrong training source."""
    if "trained_on" not in metrics.columns:
        raise ValueError(
            f"Experiment {experiment_name!r} has no trained_on column, so it cannot be checked "
            f"against the declared training source {expected_source!r}"
        )
    found = sorted(set(metrics["trained_on"].dropna().astype(str)))
    if found != [expected_source]:
        raise ValueError(
            f"Experiment {experiment_name!r} contains training source {found}, but the figure declares "
            f"{expected_source!r}; point that figure task at its own experiment instead"
        )


def _drop_instances(frame: pd.DataFrame, instances: Sequence[str]) -> pd.DataFrame:
    """Remove model instances from a bootstrap frame.

    The artifact is long (one row per run, dataset, metric, bootstrap, and model),
    so the rows are filtered; a wide frame, whose columns are named after the
    instances, is filtered by column.
    """
    if "model_instance" in frame.columns:
        frame = frame.loc[~frame["model_instance"].astype(str).isin(set(instances))]
    return frame.drop(columns=[name for name in instances if name in frame.columns])


def _exclude_models(metrics: pd.DataFrame, exclude_models: str | Sequence[str]) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Drop the named models and report the instances that were removed."""
    names = {exclude_models} if isinstance(exclude_models, str) else set(exclude_models)
    unknown = names - set(metrics["model_name"].astype(str))
    if unknown:
        available = sorted(set(metrics["model_name"].astype(str)))
        raise ValueError(f"Cannot exclude unknown models {sorted(unknown)}; available: {available}")
    excluded = metrics["model_name"].astype(str).isin(names)
    return metrics.loc[~excluded].copy(), tuple(metrics.loc[excluded, "model_instance"].astype(str).drop_duplicates())


def select_single_source_full_data_run_ids(
    metrics: pd.DataFrame,
    *,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> tuple[str, ...]:
    """Select runs explicitly configured with one source at float fraction ``1.0``.

    Every candidate run's recorded ``config.json`` is inspected. Integer ``1``,
    absolute row counts, and a largest observed ``training_size`` do not satisfy
    this full-data contract.
    """
    required = {"pipeline_mlflow_run_id", "scope", "statistic", "target", "trained_on"}
    _require_columns(metrics, required, "evaluation metrics")
    rows = metrics.loc[metrics["scope"].eq("test") & metrics["statistic"].eq("point")].copy()
    if rows.empty:
        raise ValueError("No scope='test', statistic='point' rows are available")

    for column in ("target", "trained_on"):
        values = rows[column].dropna().astype(str).unique().tolist()
        if len(values) != 1:
            raise ValueError(
                f"Automatic full-data selection requires exactly one {column}; found {values}. "
                "Select pipeline runs explicitly."
            )

    trained_on = str(rows["trained_on"].dropna().astype(str).iloc[0])
    run_ids = tuple(rows["pipeline_mlflow_run_id"].astype(str).drop_duplicates())
    client = MlflowClient(tracking_uri=tracking_uri)
    selected = []
    for run_id in run_ids:
        source, fraction = _read_single_source_training_config(client, run_id)
        if source != trained_on:
            raise ValueError(
                f"Run {run_id} config.json trains on {source!r}, but its evaluation artifact records "
                f"trained_on={trained_on!r}"
            )
        if type(fraction) is float and fraction == 1.0:
            selected.append(run_id)
    return tuple(selected)


def _read_single_source_training_config(client: MlflowClient, run_id: str) -> tuple[str, object]:
    """Read the source and fraction from one run's authoritative config artifact."""
    try:
        config_path = Path(client.download_artifacts(run_id, ARTIFACT_CONFIG))
    except MlflowException as error:
        raise ValueError(f"Run {run_id} has no readable {ARTIFACT_CONFIG} artifact") from error
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        train_on = payload["dataset"]["train_on"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(
            f"Run {run_id} has an invalid {ARTIFACT_CONFIG}; expected dataset.train_on"
        ) from error
    if not isinstance(train_on, list) or len(train_on) != 1:
        count = len(train_on) if isinstance(train_on, list) else "non-list"
        raise ValueError(
            f"Run {run_id} must record exactly one dataset.train_on entry for a single-source "
            f"full-data baseline; found {count}"
        )
    split = train_on[0]
    if not isinstance(split, dict) or "dataset" not in split or "fraction" not in split:
        raise ValueError(
            f"Run {run_id} has an invalid {ARTIFACT_CONFIG}; the dataset.train_on entry must "
            "contain dataset and fraction"
        )
    return str(split["dataset"]), split["fraction"]


def _require_columns(frame: pd.DataFrame, required: set[str], description: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing {description} columns: {', '.join(missing)}")
