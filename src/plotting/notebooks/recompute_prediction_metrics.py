import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path, PurePosixPath
    from tempfile import TemporaryDirectory

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import marimo as mo

    from mlflow import MlflowClient
    from src.config import config
    from src.mlflow.evaluation_data import load_evaluation_data
    from src.mlflow.serialization import artifact_manifest_from_json
    from src.mlflow.tracking_contract import (
        ARTIFACT_MANIFEST,
        ARTIFACT_RECOMPUTED_METRICS,
        ARTIFACT_TEST_PREDICTIONS,
        TAG_PIPELINE_ID,
        TAG_TARGET,
        TAG_TASK_TYPE,
        TAG_TRACKING_SCHEMA_VERSION,
        TEST_DATASETS,
        test_predictions_artifact,
    )
    from src.utils.prediction_metrics import recompute_classification_metrics
    from src.utils.prediction_tables import load_prediction_snapshot

    return (
        ARTIFACT_MANIFEST,
        ARTIFACT_RECOMPUTED_METRICS,
        ARTIFACT_TEST_PREDICTIONS,
        MlflowClient,
        Path,
        PurePosixPath,
        TAG_PIPELINE_ID,
        TAG_TARGET,
        TAG_TASK_TYPE,
        TAG_TRACKING_SCHEMA_VERSION,
        TEST_DATASETS,
        TemporaryDirectory,
        artifact_manifest_from_json,
        config,
        load_evaluation_data,
        load_prediction_snapshot,
        mo,
        recompute_classification_metrics,
        test_predictions_artifact,
    )


@app.cell
def _(mo):
    mo.md("""
    # Recompute classification metrics from saved test predictions

    This notebook downloads the cumulative MIMIC and TUDD prediction CSVs
    from one parent pipeline run, recomputes the same six binary metrics and
    bootstrap intervals used during training, compares point estimates with
    the metrics already logged by the pipeline, and optionally writes a new
    CSV artifact back to that parent run.

    The y_pred columns contain the probability of encoded class 1. Each
    test_set_id is a non-PHI SHA-256 identifier derived from dataset origin,
    filtered-file fingerprint, and source row index. It is stable across runs
    that use the same filtered input file and changes when that file changes.
    """)
    return


@app.cell
def _(config):
    TRACKING_URI = "sqlite:///mlflow.db"
    PIPELINE_MLFLOW_RUN_ID = "2b7d2293b0df469790ec1e075a5dfa6b"
    N_BOOTSTRAP = 10_000
    RANDOM_STATE = config.seed

    # Safety switch: set to True only after filling in the parent run ID.
    RUN_RECOMPUTATION = True

    # If True, write prediction_metrics/classification_metrics.csv to the run.
    LOG_TO_MLFLOW = True
    return (
        LOG_TO_MLFLOW,
        N_BOOTSTRAP,
        PIPELINE_MLFLOW_RUN_ID,
        RANDOM_STATE,
        RUN_RECOMPUTATION,
        TRACKING_URI,
    )


@app.cell
def _(
    LOG_TO_MLFLOW,
    N_BOOTSTRAP,
    PIPELINE_MLFLOW_RUN_ID,
    RANDOM_STATE,
    RUN_RECOMPUTATION,
    TRACKING_URI,
    mo,
):
    mo.md(f"""
    **Tracking URI:** {TRACKING_URI}  
    **Parent run ID:** {PIPELINE_MLFLOW_RUN_ID or "not set"}  
    **Bootstrap samples:** {N_BOOTSTRAP:,}  
    **Random state:** {RANDOM_STATE}  
    **Run enabled:** {RUN_RECOMPUTATION}  
    **Log artifact:** {LOG_TO_MLFLOW}
    """)
    return


@app.cell
def _(
    ARTIFACT_MANIFEST,
    ARTIFACT_RECOMPUTED_METRICS,
    ARTIFACT_TEST_PREDICTIONS,
    LOG_TO_MLFLOW,
    MlflowClient,
    N_BOOTSTRAP,
    PIPELINE_MLFLOW_RUN_ID,
    Path,
    PurePosixPath,
    RANDOM_STATE,
    RUN_RECOMPUTATION,
    TAG_PIPELINE_ID,
    TAG_TARGET,
    TAG_TASK_TYPE,
    TAG_TRACKING_SCHEMA_VERSION,
    TEST_DATASETS,
    TRACKING_URI,
    TemporaryDirectory,
    artifact_manifest_from_json,
    load_evaluation_data,
    load_prediction_snapshot,
    mo,
    recompute_classification_metrics,
    test_predictions_artifact,
):
    mo.stop(
        not RUN_RECOMPUTATION,
        mo.md("Set RUN_RECOMPUTATION = True after entering a parent MLflow run ID."),
    )
    if not PIPELINE_MLFLOW_RUN_ID.strip():
        raise ValueError("PIPELINE_MLFLOW_RUN_ID must identify a parent pipeline run")
    if N_BOOTSTRAP < 1:
        raise ValueError("N_BOOTSTRAP must be at least 1")

    client = MlflowClient(tracking_uri=TRACKING_URI)
    pipeline_run = client.get_run(PIPELINE_MLFLOW_RUN_ID)
    task_type = pipeline_run.data.tags.get(TAG_TASK_TYPE)
    if task_type != "classification":
        raise ValueError(
            f"Prediction probability artifacts are only available for classification runs, got {task_type!r}"
        )

    manifest_path = Path(client.download_artifacts(PIPELINE_MLFLOW_RUN_ID, ARTIFACT_MANIFEST))
    manifest = artifact_manifest_from_json(manifest_path.read_text(encoding="utf-8"))
    expected_prediction_artifacts = {test_predictions_artifact(dataset) for dataset in TEST_DATASETS}
    missing_prediction_artifacts = expected_prediction_artifacts.difference(manifest.test_predictions)
    if missing_prediction_artifacts:
        missing = ", ".join(sorted(missing_prediction_artifacts))
        raise ValueError(f"Run manifest does not contain prediction artifacts: {missing}")

    snapshot_dir = client.download_artifacts(
        PIPELINE_MLFLOW_RUN_ID,
        ARTIFACT_TEST_PREDICTIONS,
    )
    prediction_snapshot = load_prediction_snapshot(snapshot_dir)
    prediction_tables = prediction_snapshot.tables

    recomputed_metrics = recompute_classification_metrics(
        prediction_tables,
        n_bootstrap=N_BOOTSTRAP,
        random_state=RANDOM_STATE,
    )
    context_columns = {
        "pipeline_mlflow_run_id": PIPELINE_MLFLOW_RUN_ID,
        "pipeline_id": pipeline_run.data.tags.get(TAG_PIPELINE_ID),
        "target": pipeline_run.data.tags.get(TAG_TARGET),
        "tracking_schema_version": pipeline_run.data.tags.get(TAG_TRACKING_SCHEMA_VERSION),
        "prediction_generation_id": prediction_snapshot.generation_id,
    }
    for position, (column, value) in enumerate(context_columns.items()):
        recomputed_metrics.insert(position, column, value)

    experiment = client.get_experiment(pipeline_run.info.experiment_id)
    if experiment is None:
        raise ValueError(f"MLflow experiment {pipeline_run.info.experiment_id!r} no longer exists")
    logged_metrics = load_evaluation_data(
        experiment_names=experiment.name,
        pipeline_runs=[PIPELINE_MLFLOW_RUN_ID],
        tracking_uri=TRACKING_URI,
    )
    metric_names = ["roc_auc", "prc_auc", "f1", "accuracy", "sensitivity", "precision"]
    comparison = recomputed_metrics.loc[
        :,
        ["model_instance", "scope", "dataset", *metric_names],
    ].merge(
        logged_metrics.loc[
            :,
            ["model_instance", "scope", "dataset", *metric_names],
        ],
        on=["model_instance", "scope", "dataset"],
        how="left",
        suffixes=("_recomputed", "_logged"),
        validate="one_to_one",
    )
    for metric in metric_names:
        comparison[f"{metric}_difference"] = comparison[f"{metric}_recomputed"] - comparison[f"{metric}_logged"]

    logged_artifact = None
    if LOG_TO_MLFLOW:
        artifact = PurePosixPath(ARTIFACT_RECOMPUTED_METRICS)
        with TemporaryDirectory() as temp_dir_name:
            local_path = Path(temp_dir_name) / artifact.name
            recomputed_metrics.to_csv(local_path, index=False)
            client.log_artifact(
                PIPELINE_MLFLOW_RUN_ID,
                str(local_path),
                artifact_path=str(artifact.parent),
            )
        logged_artifact = ARTIFACT_RECOMPUTED_METRICS
    return comparison, logged_artifact, recomputed_metrics


@app.cell
def _(comparison, logged_artifact, mo, recomputed_metrics):
    destination = logged_artifact or "logging disabled"
    mo.vstack(
        [
            mo.md(
                f"""## Recomputed metrics

                Artifact: **{destination}**
                """
            ),
            recomputed_metrics,
            mo.md(
                """## Point-estimate comparison

                Difference = recomputed minus pipeline-logged.
                """
            ),
            comparison,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
