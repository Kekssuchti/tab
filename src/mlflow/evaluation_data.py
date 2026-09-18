from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from mlflow import MlflowClient
from mlflow.entities import Experiment, Run
from mlflow.exceptions import MlflowException
from src.mlflow.tracking_contract import (
    ARTIFACT_CLASSIFICATION_METRICS,
    METRIC_TRAIN_FIT_TIME,
    RUN_TYPE_MODEL,
    RUN_TYPE_PIPELINE,
    TAG_MODEL_INSTANCE,
    TAG_MODEL_INSTANCES,
    TAG_MODEL_NAME,
    TAG_PIPELINE_ID,
    TAG_PIPELINE_MLFLOW_RUN_ID,
    TAG_RUN_TYPE,
    TAG_TARGET,
    TAG_TASK_TYPE,
    TAG_TRACKING_SCHEMA_VERSION,
    TAG_TRAIN_SOURCES,
    TAG_TRAINED_ON,
    TAG_TRAINING_SIZE,
    TRACKING_SCHEMA_VERSION,
    test_predict_time_metric,
)
from src.schemas.training_schemas import LOWER_IS_BETTER_SCORING

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT_NAME = "tab"


def list_pipeline_runs(
    experiment_names: str | Sequence[str] = DEFAULT_EXPERIMENT_NAME,
    *,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> pd.DataFrame:
    """List compact v1 parent runs."""

    client = MlflowClient(tracking_uri=tracking_uri)
    rows = []
    for experiment in _experiments(client, experiment_names):
        for run in _runs(client, experiment, RUN_TYPE_PIPELINE):
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
    """Load final post-processed metric artifacts into one plotting frame."""

    client = MlflowClient(tracking_uri=tracking_uri)
    requested_runs = _selectors(pipeline_runs)
    requested_models = _selectors(models)
    frames = []

    for experiment in _experiments(client, experiment_names):
        parents = _runs(client, experiment, RUN_TYPE_PIPELINE)
        children = _runs(client, experiment, RUN_TYPE_MODEL)
        children_by_parent = _children_by_parent(children)
        for parent in parents:
            if requested_runs and not requested_runs.intersection(_run_selectors(parent)):
                continue
            try:
                local_path = client.download_artifacts(parent.info.run_id, ARTIFACT_CLASSIFICATION_METRICS)
            except MlflowException:
                continue

            frame = pd.read_csv(Path(local_path))
            model_runs = {
                child.data.tags[TAG_MODEL_INSTANCE]: child for child in children_by_parent.get(parent.info.run_id, ())
            }
            if requested_models:
                selected_instances = {
                    instance
                    for instance, child in model_runs.items()
                    if requested_models.intersection(_model_selectors(child))
                }
                frame = frame.loc[frame["model_instance"].isin(selected_instances)].copy()
            if frame.empty:
                continue

            frame.insert(0, "pipeline_mlflow_run_id", parent.info.run_id)
            frame.insert(1, "pipeline_id", parent.data.tags.get(TAG_PIPELINE_ID))
            frame.insert(2, "pipeline_run_name", parent.data.tags.get("mlflow.runName"))
            frame.insert(3, "experiment_name", experiment.name)
            frame.insert(
                4,
                "model_mlflow_run_id",
                frame["model_instance"].map({instance: child.info.run_id for instance, child in model_runs.items()}),
            )
            frame.insert(
                5,
                "model_name",
                frame["model_instance"].map(
                    {instance: child.data.tags.get(TAG_MODEL_NAME) for instance, child in model_runs.items()}
                ),
            )
            timing_metrics = {
                "fit_time": METRIC_TRAIN_FIT_TIME,
                "predict_time_mimic": test_predict_time_metric("mimic"),
                "predict_time_tudd": test_predict_time_metric("tudd"),
            }
            for column, metric_name in timing_metrics.items():
                frame[column] = frame["model_instance"].map(
                    {instance: child.data.metrics.get(metric_name) for instance, child in model_runs.items()}
                )
            frame["total_time"] = frame[list(timing_metrics)].sum(axis=1, min_count=1)
            frame["target"] = parent.data.tags.get(TAG_TARGET)
            frame["task_type"] = parent.data.tags.get(TAG_TASK_TYPE)
            frame["trained_on"] = parent.data.tags.get(TAG_TRAINED_ON)
            frame["train_sources"] = [
                tuple(filter(None, parent.data.tags.get(TAG_TRAIN_SOURCES, "").split(",")))
            ] * len(frame)
            training_size = parent.data.tags.get(TAG_TRAINING_SIZE)
            frame["training_size"] = int(training_size) if training_size else None
            frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return _add_generalizability_losses(pd.concat(frames, ignore_index=True))


def _experiments(client: MlflowClient, names: str | Sequence[str]) -> list[Experiment]:
    names = [names] if isinstance(names, str) else list(names)
    return [experiment for name in names if (experiment := client.get_experiment_by_name(name)) is not None]


def _runs(client: MlflowClient, experiment: Experiment, run_type: str) -> list[Run]:
    return list(
        client.search_runs(
            [experiment.experiment_id],
            filter_string=(
                f"tags.{TAG_RUN_TYPE} = '{run_type}' "
                f"and tags.{TAG_TRACKING_SCHEMA_VERSION} = '{TRACKING_SCHEMA_VERSION}'"
            ),
            order_by=["attributes.start_time ASC"],
        )
    )


def _children_by_parent(children: list[Run]) -> dict[str, list[Run]]:
    grouped = {}
    for child in children:
        grouped.setdefault(child.data.tags.get(TAG_PIPELINE_MLFLOW_RUN_ID, ""), []).append(child)
    return grouped


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


def _model_selectors(run: Run) -> set[str]:
    return {
        value
        for value in (
            run.info.run_id,
            run.data.tags.get(TAG_MODEL_INSTANCE),
            run.data.tags.get(TAG_MODEL_NAME),
        )
        if value
    }


def _selectors(values: str | Sequence[str] | None) -> set[str]:
    if values is None:
        return set()
    return {values} if isinstance(values, str) else set(values)


def _add_generalizability_losses(frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column for column in ("roc_auc", "prc_auc", "f1", "accuracy", "sensitivity", "precision") if column in frame
    ]
    test_rows = frame["scope"].eq("test")
    external = test_rows & frame["dataset"].ne(frame["trained_on"])
    training = test_rows & frame["dataset"].eq(frame["trained_on"])

    for metric in metric_columns:
        loss = f"generalizability_loss_{metric}"
        comparative = f"comparative_generalizability_loss_{metric}"
        frame[loss] = float("nan")
        frame[comparative] = float("nan")
        training_scores = frame.loc[training].set_index("model_mlflow_run_id")[metric]
        reference = frame.loc[external, "model_mlflow_run_id"].map(training_scores)
        if metric in LOWER_IS_BETTER_SCORING:
            frame.loc[external, loss] = reference - frame.loc[external, metric]
            best = frame.loc[external].groupby(["target", "dataset"])[metric].transform("min")
            frame.loc[external, comparative] = best - frame.loc[external, metric]
        else:
            frame.loc[external, loss] = frame.loc[external, metric] - reference
            best = frame.loc[external].groupby(["target", "dataset"])[metric].transform("max")
            frame.loc[external, comparative] = frame.loc[external, metric] - best
    return frame
