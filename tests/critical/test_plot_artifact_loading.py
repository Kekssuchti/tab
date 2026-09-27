"""The sole plotting integration test: stored synthetic data, no figure rendering."""

import pandas as pd
import pytest

from mlflow import MlflowClient
from src.plotting.utils.artifacts import load_plot_artifacts


def test_plot_loader_preserves_run_model_and_bootstrap_identity_and_rejects_missing_run(tmp_path):
    """Catch cross-run/model swaps and partial comparisons hidden by plausible plotted values."""
    tracking_uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_name = "synthetic-artifact-consumer"
    experiment_id = client.create_experiment(experiment_name, artifact_location=(tmp_path / "artifacts").as_uri())
    names = {"slot_a": "forest", "slot_b": "forest", "slot_x": "exclude_me"}
    # Literal artifact rows are the oracle, independent of production writer/loader helpers.
    points = [
        ("mimic", "slot_a", .71, .41), ("mimic", "slot_b", .62, .32), ("mimic", "slot_x", .53, .23),
        ("tudd", "slot_a", .74, .44), ("tudd", "slot_b", .65, .35), ("tudd", "slot_x", .56, .26),
    ]
    draws = [
        ("mimic", "roc_auc", 0, .71, .62, .53), ("mimic", "roc_auc", 1, .72, .63, .54),
        ("mimic", "prc_auc", 0, .41, .32, .23), ("mimic", "prc_auc", 1, .42, .33, .24),
        ("tudd", "roc_auc", 0, .75, .66, .57), ("tudd", "roc_auc", 1, .76, .67, .58),
        ("tudd", "prc_auc", 0, .45, .36, .27), ("tudd", "prc_auc", 1, .46, .37, .28),
    ]
    expected_points, expected_draws = {}, {}

    def store_run(label, offset, *, bootstrap=True):
        run = client.create_run(experiment_id, tags={
            "run_type": "pipeline", "tracking_schema_version": "1", "pipeline_id": label,
            "mlflow.runName": label, "model_instances": "slot_a,slot_b,slot_x",
            "target": "mortality", "task_type": "classification", "trained_on": "mixed",
            "train_sources": "mimic,tudd", "training_size": "24",
        })
        run_id = run.info.run_id
        directory = tmp_path / label
        directory.mkdir()
        metadata = {
            "pipeline_mlflow_run_id": run_id, "pipeline_id": label, "pipeline_run_name": label,
            "experiment_name": experiment_name, "scope": "test", "statistic": "point",
            "target": "mortality", "task_type": "classification", "trained_on": "mixed",
            "train_sources": "mimic,tudd", "training_size": 24,
            "cv_time": .1, "fit_time": .2, "predict_time_mimic": .01, "predict_time_tudd": .02,
            "training_time": .3, "total_time": .33,
        }
        metric_rows = []
        for dataset, instance, auc, ap in points:
            metric_rows.append({
                **metadata, "dataset": dataset, "model_instance": instance, "model_name": names[instance],
                "roc_auc": auc + offset, "prc_auc": ap + offset,
            })
            if bootstrap and instance != "slot_x":
                expected_points[run_id, dataset, instance] = (auc + offset, ap + offset)
        metrics_path = directory / "prediction_metrics.csv"
        pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
        client.log_artifact(run_id, str(metrics_path))
        if bootstrap:
            bootstrap_rows = []
            for dataset, metric, draw, a, b, excluded in draws:
                bootstrap_rows.append({
                    "dataset": dataset, "metric": metric, "bootstrap_id": draw,
                    "slot_a": a + offset, "slot_b": b + offset, "slot_x": excluded + offset,
                })
                expected_draws[run_id, dataset, metric, draw, "slot_a"] = a + offset
                expected_draws[run_id, dataset, metric, draw, "slot_b"] = b + offset
            bootstrap_path = directory / "bootstrap_metrics.csv"
            pd.DataFrame(bootstrap_rows).to_csv(bootstrap_path, index=False)
            client.log_artifact(run_id, str(bootstrap_path))
        client.set_terminated(run_id)
        return run_id

    first = store_run("run-first", 0)
    second = store_run("run-second", .1)
    incomplete = store_run("run-incomplete", .2, bootstrap=False)
    artifacts = load_plot_artifacts(
        experiment_name, pipeline_runs=[first, second], exclude_models="exclude_me", tracking_uri=tracking_uri,
    )
    assert artifacts.experiment_name == experiment_name
    assert set(artifacts.run_ids) == {first, second}
    assert len(artifacts.run_ids) == 2
    point_keys = ["pipeline_mlflow_run_id", "dataset", "model_instance"]
    actual_points = artifacts.metrics.set_index(point_keys)
    assert actual_points.index.is_unique
    assert set(actual_points.index) == set(expected_points)
    for key, values in expected_points.items():
        assert tuple(actual_points.loc[key, ["roc_auc", "prc_auc"]]) == pytest.approx(values)

    draw_keys = ["pipeline_mlflow_run_id", "dataset", "metric", "bootstrap_id", "model_instance"]
    actual_draws = artifacts.bootstrap_scores.set_index(draw_keys)
    assert actual_draws.index.is_unique
    assert set(actual_draws.index) == set(expected_draws)
    for key, score in expected_draws.items():
        assert actual_draws.loc[key, "score"] == pytest.approx(score)
    for frame in (artifacts.metrics, artifacts.bootstrap_scores):
        assert set(frame.model_instance) == {"slot_a", "slot_b"}
        assert set(zip(frame.model_instance, frame.model_name)) == {("slot_a", "forest"), ("slot_b", "forest")}
        assert set(frame.train_sources) == {("mimic", "tudd")}
        assert set(zip(frame.pipeline_mlflow_run_id, frame.pipeline_id)) == {
            (first, "run-first"), (second, "run-second"),
        }

    # Both runs have metrics; only one has bootstrap data. Never silently compare that subset.
    with pytest.raises(ValueError, match="Metric and bootstrap artifacts contain different pipeline runs"):
        load_plot_artifacts(
            experiment_name, pipeline_runs=[first, incomplete], exclude_models="exclude_me", tracking_uri=tracking_uri,
        )
