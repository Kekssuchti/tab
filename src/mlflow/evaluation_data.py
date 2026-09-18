from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from mlflow import MlflowClient
from mlflow.entities import Experiment, Run
from mlflow.exceptions import MlflowException
from src.mlflow.tracking_contract import (
    ARTIFACT_CLASSIFICATION_METRICS,
    RUN_TYPE_PIPELINE,
    TAG_MODEL_INSTANCES,
    TAG_PIPELINE_ID,
    TAG_RUN_TYPE,
    TAG_TARGET,
    TAG_TASK_TYPE,
    TAG_TRACKING_SCHEMA_VERSION,
    TAG_TRAIN_SOURCES,
    TAG_TRAINED_ON,
    TRACKING_SCHEMA_VERSION,
)
from src.schemas.training_schemas import LOWER_IS_BETTER_SCORING

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT_NAME = "tab"

_PLOTTING_COLUMNS = {
    "pipeline_mlflow_run_id",
    "pipeline_id",
    "pipeline_run_name",
    "experiment_name",
    "model_instance",
    "model_name",
    "scope",
    "statistic",
    "dataset",
    "target",
    "task_type",
    "trained_on",
    "train_sources",
    "training_size",
    "cv_time",
    "fit_time",
    "predict_time_mimic",
    "predict_time_tudd",
    "training_time",
    "total_time",
}
_MODEL_SELECTOR_COLUMNS = ("model_mlflow_run_id", "model_instance", "model_name")


def list_pipeline_runs(
    experiment_names: str | Sequence[str] = DEFAULT_EXPERIMENT_NAME,
    *,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> pd.DataFrame:
    """List compact v1 parent runs."""
    client = MlflowClient(tracking_uri=tracking_uri)
    rows = []
    for experiment in _experiments(client, experiment_names):
        for run in _runs(client, experiment):
            rows.append(
                {
                    "mlflow_run_id": run.info.run_id,
                    "pipeline_id": run.data.tags.get(TAG_PIPELINE_ID),
                    "run_name": run.data.tags.get("mlflow.runName"),
                    "experiment_name": experiment.name,
                    "model_instances": tuple(filter(None, run.data.tags.get(TAG_MODEL_INSTANCES, "").split(","))),
                    "target": run.data.tags.get(TAG_TARGET),
                    "task_type": run.data.tags.get(TAG_TASK_TYPE),
                    "trained_on": run.data.tags.get(TAG_TRAINED_ON),
                    "train_sources": tuple(filter(None, run.data.tags.get(TAG_TRAIN_SOURCES, "").split(","))),
                }
            )
    return pd.DataFrame(rows)


def load_evaluation_data(
    experiment_names: str | Sequence[str] = DEFAULT_EXPERIMENT_NAME,
    *,
    pipeline_runs: str | Sequence[str] | None = None,
    models: str | Sequence[str] | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> pd.DataFrame:
    """Concatenate self-contained prediction metric CSVs for plotting."""
    client = MlflowClient(tracking_uri=tracking_uri)
    requested_runs = _selectors(pipeline_runs)
    requested_models = _selectors(models)
    frames = []

    for experiment in _experiments(client, experiment_names):
        for parent in _runs(client, experiment):
            if requested_runs and not requested_runs.intersection(_run_selectors(parent)):
                continue
            try:
                local_path = client.download_artifacts(parent.info.run_id, ARTIFACT_CLASSIFICATION_METRICS)
            except MlflowException:
                continue

            frame = pd.read_csv(Path(local_path))
            _validate_plotting_frame(frame, parent.info.run_id)
            if requested_models:
                selector_columns = [column for column in _MODEL_SELECTOR_COLUMNS if column in frame]
                selected = frame[selector_columns].astype(str).isin(requested_models).any(axis=1)
                frame = frame.loc[selected].copy()
            if not frame.empty:
                frames.append(frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["train_sources"] = combined["train_sources"].map(_parse_sources)
    return _add_generalizability_losses(combined)


def _experiments(client: MlflowClient, names: str | Sequence[str]) -> list[Experiment]:
    names = [names] if isinstance(names, str) else list(names)
    return [experiment for name in names if (experiment := client.get_experiment_by_name(name)) is not None]


def _runs(client: MlflowClient, experiment: Experiment) -> list[Run]:
    return list(
        client.search_runs(
            [experiment.experiment_id],
            filter_string=(
                f"tags.{TAG_RUN_TYPE} = '{RUN_TYPE_PIPELINE}' "
                f"and tags.{TAG_TRACKING_SCHEMA_VERSION} = '{TRACKING_SCHEMA_VERSION}'"
            ),
            order_by=["attributes.start_time ASC"],
        )
    )


def _run_selectors(run: Run) -> set[str]:
    return {
        value
        for value in (
            run.info.run_id,
            run.data.tags.get(TAG_PIPELINE_ID),
            run.data.tags.get("mlflow.runName"),
        )
        if value
    }


def _selectors(values: str | Sequence[str] | None) -> set[str]:
    if values is None:
        return set()
    return {values} if isinstance(values, str) else set(values)


def _validate_plotting_frame(frame: pd.DataFrame, run_id: str) -> None:
    missing = sorted(_PLOTTING_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            f"Run {run_id} has an outdated {ARTIFACT_CLASSIFICATION_METRICS} artifact; "
            f"missing plotting columns: {', '.join(missing)}"
        )
    recorded_ids = frame["pipeline_mlflow_run_id"].dropna().astype(str).unique().tolist()
    if recorded_ids != [run_id]:
        raise ValueError(
            f"Run {run_id} has mismatched pipeline_mlflow_run_id values in "
            f"{ARTIFACT_CLASSIFICATION_METRICS}: {recorded_ids}"
        )


def _parse_sources(value: object) -> tuple[str, ...]:
    if pd.isna(value):
        return ()
    return tuple(filter(None, str(value).split(",")))


def _add_generalizability_losses(frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column for column in ("roc_auc", "prc_auc", "f1", "accuracy", "sensitivity", "precision") if column in frame
    ]
    test_rows = frame["scope"].eq("test")
    external = test_rows & frame["dataset"].ne(frame["trained_on"])
    training = test_rows & frame["dataset"].eq(frame["trained_on"])
    identity_columns = ["pipeline_mlflow_run_id", "model_instance"]

    for metric in metric_columns:
        loss = f"generalizability_loss_{metric}"
        comparative = f"comparative_generalizability_loss_{metric}"
        frame[loss] = float("nan")
        frame[comparative] = float("nan")

        training_scores = frame.loc[training].set_index(identity_columns)[metric]
        external_index = pd.MultiIndex.from_frame(frame.loc[external, identity_columns])
        reference = training_scores.reindex(external_index).to_numpy()
        if metric in LOWER_IS_BETTER_SCORING:
            frame.loc[external, loss] = reference - frame.loc[external, metric].to_numpy()
            best = frame.loc[external].groupby(["target", "dataset"])[metric].transform("min")
            frame.loc[external, comparative] = best - frame.loc[external, metric]
        else:
            frame.loc[external, loss] = frame.loc[external, metric].to_numpy() - reference
            best = frame.loc[external].groupby(["target", "dataset"])[metric].transform("max")
            frame.loc[external, comparative] = frame.loc[external, metric] - best
    return frame
