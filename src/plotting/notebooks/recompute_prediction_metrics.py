import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import sys
    from pathlib import Path
    from tempfile import TemporaryDirectory

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import marimo as mo

    from mlflow import MlflowClient
    from src.config import config
    from src.mlflow.evaluation_recovery import add_run_metadata
    from src.mlflow.tracking_contract import (
        ARTIFACT_BOOTSTRAP_METRICS,
        ARTIFACT_CLASSIFICATION_METRICS,
        ARTIFACT_CONFIG,
        ARTIFACT_PAIRWISE_WINS,
        ARTIFACT_TEST_PREDICTIONS,
        TAG_TASK_TYPE,
        TAG_TRACKING_SCHEMA_VERSION,
        TRACKING_SCHEMA_VERSION,
    )
    from src.utils.prediction_metrics import evaluate_classification_models
    from src.utils.prediction_tables import load_prediction_snapshot

    return (
        ARTIFACT_BOOTSTRAP_METRICS,
        ARTIFACT_CLASSIFICATION_METRICS,
        ARTIFACT_CONFIG,
        ARTIFACT_PAIRWISE_WINS,
        ARTIFACT_TEST_PREDICTIONS,
        MlflowClient,
        Path,
        TAG_TASK_TYPE,
        TAG_TRACKING_SCHEMA_VERSION,
        TRACKING_SCHEMA_VERSION,
        TemporaryDirectory,
        add_run_metadata,
        config,
        json,
        load_prediction_snapshot,
        mo,
        evaluate_classification_models,
    )


@app.cell
def _(mo):
    mo.md("""
    # Rebuild classification metrics from saved predictions

    The pipeline normally performs this step automatically after every model has
    finished. Use this notebook to recover the final metrics artifact when that
    post-processing step was interrupted.

    It downloads the cumulative MIMIC and TUDD probability tables, validates the
    snapshot hashes, recomputes point metrics and 95% bootstrap intervals, and builds
    paired AUROC/AUPRC win matrices. It can upload all replacement artifacts.
    """)


@app.cell
def _(config):
    TRACKING_URI = "sqlite:///mlflow.db"
    PIPELINE_MLFLOW_RUN_ID = ""
    N_BOOTSTRAP = 10_000
    RANDOM_STATE = config.seed

    RUN_RECOMPUTATION = False
    LOG_TO_MLFLOW = False
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
    **Bootstrap seed:** read from the selected run's `config.json`, falling back to {RANDOM_STATE} for older runs  
    **Run enabled:** {RUN_RECOMPUTATION}  
    **Replace MLflow artifact:** {LOG_TO_MLFLOW}
    """)


@app.cell
def _(
    ARTIFACT_BOOTSTRAP_METRICS,
    ARTIFACT_CLASSIFICATION_METRICS,
    ARTIFACT_CONFIG,
    ARTIFACT_PAIRWISE_WINS,
    ARTIFACT_TEST_PREDICTIONS,
    LOG_TO_MLFLOW,
    MlflowClient,
    N_BOOTSTRAP,
    PIPELINE_MLFLOW_RUN_ID,
    Path,
    RANDOM_STATE,
    RUN_RECOMPUTATION,
    TAG_TASK_TYPE,
    TAG_TRACKING_SCHEMA_VERSION,
    TRACKING_SCHEMA_VERSION,
    TRACKING_URI,
    TemporaryDirectory,
    add_run_metadata,
    evaluate_classification_models,
    json,
    load_prediction_snapshot,
    mo,
):
    mo.stop(not RUN_RECOMPUTATION, mo.md("Set **RUN_RECOMPUTATION = True** to run."))
    mo.stop(not PIPELINE_MLFLOW_RUN_ID, mo.md("Set **PIPELINE_MLFLOW_RUN_ID** first."))

    client = MlflowClient(tracking_uri=TRACKING_URI)
    run = client.get_run(PIPELINE_MLFLOW_RUN_ID)
    if run.data.tags.get(TAG_TRACKING_SCHEMA_VERSION) != TRACKING_SCHEMA_VERSION:
        raise ValueError("The selected run does not use evaluation protocol v1")
    if run.data.tags.get(TAG_TASK_TYPE) != "classification":
        raise ValueError("Prediction post-processing only supports classification runs")

    with TemporaryDirectory() as temp_dir:
        snapshot_dir = Path(
            client.download_artifacts(
                PIPELINE_MLFLOW_RUN_ID,
                ARTIFACT_TEST_PREDICTIONS,
                dst_path=temp_dir,
            )
        )
        snapshot = load_prediction_snapshot(snapshot_dir)
        logged_config_path = Path(
            client.download_artifacts(
                PIPELINE_MLFLOW_RUN_ID,
                ARTIFACT_CONFIG,
                dst_path=temp_dir,
            )
        )
        logged_config = json.loads(logged_config_path.read_text(encoding="utf-8"))
        # Reuse the bootstrap seed the run was produced with, so recomputed
        # intervals and win matrices match the pipeline's own artifacts. Runs
        # logged before seeds became configurable fall back to the default.
        bootstrap_seed = logged_config.get("random_states", {}).get(
            "evaluation_bootstrap_seed",
            RANDOM_STATE,
        )
        classification_evaluation = evaluate_classification_models(
            snapshot.tables,
            n_bootstrap=N_BOOTSTRAP,
            random_state=bootstrap_seed,
        )

    plotting_metrics = classification_evaluation.metrics
    logged_artifacts = None
    if LOG_TO_MLFLOW:
        plotting_metrics = add_run_metadata(
            plotting_metrics,
            client=client,
            pipeline_run_id=PIPELINE_MLFLOW_RUN_ID,
        )
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metrics_path = root / ARTIFACT_CLASSIFICATION_METRICS
            bootstrap_path = root / ARTIFACT_BOOTSTRAP_METRICS
            plotting_metrics.to_csv(metrics_path, index=False)
            classification_evaluation.bootstrap_scores.to_csv(bootstrap_path, index=False)
            client.log_artifact(PIPELINE_MLFLOW_RUN_ID, str(metrics_path))
            client.log_artifact(PIPELINE_MLFLOW_RUN_ID, str(bootstrap_path))

            pairwise_dir = root / ARTIFACT_PAIRWISE_WINS
            pairwise_dir.mkdir()
            for name, matrix in classification_evaluation.pairwise_wins.items():
                matrix.to_csv(pairwise_dir / f"{name}.csv")
            client.log_artifacts(
                PIPELINE_MLFLOW_RUN_ID,
                str(pairwise_dir),
                artifact_path=ARTIFACT_PAIRWISE_WINS,
            )
        logged_artifacts = f"{ARTIFACT_CLASSIFICATION_METRICS}, {ARTIFACT_BOOTSTRAP_METRICS}, {ARTIFACT_PAIRWISE_WINS}/"
    return classification_evaluation, logged_artifacts, plotting_metrics, snapshot


@app.cell
def _(classification_evaluation, logged_artifacts, mo, plotting_metrics, snapshot):
    mo.vstack(
        [
            mo.md(
                f"""## Recomputed metrics

                Snapshot generation: **{snapshot.generation_id}**<br>
                MLflow artifacts: **{logged_artifacts or "not replaced"}**
                """
            ),
            plotting_metrics,
            *(
                mo.vstack([mo.md(f"### {name}"), matrix])
                for name, matrix in classification_evaluation.pairwise_wins.items()
            ),
        ]
    )


if __name__ == "__main__":
    app.run()
