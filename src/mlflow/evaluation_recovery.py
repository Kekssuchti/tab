from __future__ import annotations

import pandas as pd

from mlflow import MlflowClient
from src.mlflow.evaluation_data import _validate_plotting_frame
from src.mlflow.tracking_contract import (
    METRIC_TRAIN_CV_TIME,
    METRIC_TRAIN_FIT_TIME,
    RUN_TYPE_MODEL,
    TAG_MODEL_INSTANCE,
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


def add_run_metadata(
    metrics: pd.DataFrame,
    *,
    client: MlflowClient,
    pipeline_run_id: str,
) -> pd.DataFrame:
    """Enrich recovered metrics when the normal final CSV was not written."""
    parent = client.get_run(pipeline_run_id)
    experiment = client.get_experiment(parent.info.experiment_id)
    children = client.search_runs(
        [parent.info.experiment_id],
        filter_string=(
            f"tags.{TAG_RUN_TYPE} = '{RUN_TYPE_MODEL}' "
            f"and tags.{TAG_PIPELINE_MLFLOW_RUN_ID} = '{pipeline_run_id}' "
            f"and tags.{TAG_TRACKING_SCHEMA_VERSION} = '{TRACKING_SCHEMA_VERSION}'"
        ),
        order_by=["attributes.start_time ASC"],
    )
    model_metadata = pd.DataFrame(
        [
            {
                "model_instance": child.data.tags.get(TAG_MODEL_INSTANCE),
                "model_mlflow_run_id": child.info.run_id,
                "model_name": child.data.tags.get(TAG_MODEL_NAME),
                "cv_time": child.data.metrics.get(METRIC_TRAIN_CV_TIME),
                "fit_time": child.data.metrics.get(METRIC_TRAIN_FIT_TIME),
                "predict_time_mimic": child.data.metrics.get(test_predict_time_metric("mimic")),
                "predict_time_tudd": child.data.metrics.get(test_predict_time_metric("tudd")),
            }
            for child in children
        ]
    )
    if model_metadata.empty:
        raise ValueError(f"Run {pipeline_run_id} has no completed model runs")
    model_metadata = model_metadata.drop_duplicates("model_instance", keep="last")
    missing_models = set(metrics["model_instance"]) - set(model_metadata["model_instance"])
    if missing_models:
        raise ValueError("Missing model run metadata for: " + ", ".join(sorted(missing_models)))

    model_metadata["training_time"] = model_metadata[["cv_time", "fit_time"]].sum(axis=1, min_count=2)
    model_metadata["total_time"] = model_metadata[
        ["cv_time", "fit_time", "predict_time_mimic", "predict_time_tudd"]
    ].sum(axis=1, min_count=4)
    frame = metrics.merge(model_metadata, on="model_instance", how="left", validate="many_to_one")

    training_size = parent.data.tags.get(TAG_TRAINING_SIZE)
    pipeline_columns = (
        ("pipeline_mlflow_run_id", pipeline_run_id),
        ("pipeline_id", parent.data.tags.get(TAG_PIPELINE_ID)),
        ("pipeline_run_name", parent.data.tags.get("mlflow.runName")),
        ("experiment_name", experiment.name),
        ("target", parent.data.tags.get(TAG_TARGET)),
        ("task_type", parent.data.tags.get(TAG_TASK_TYPE)),
        ("trained_on", parent.data.tags.get(TAG_TRAINED_ON)),
        ("train_sources", parent.data.tags.get(TAG_TRAIN_SOURCES, "")),
        ("training_size", int(training_size) if training_size else None),
    )
    for index, (column, value) in enumerate(pipeline_columns):
        frame.insert(index, column, value)
    _validate_plotting_frame(frame, pipeline_run_id)
    return frame
