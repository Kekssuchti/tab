"""Small synthetic file builders and independent output oracles; no pipeline fakes."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from mlflow import MlflowClient
from src.schemas.pipeline_schemas import PipelineConfig
from src.utils.config_io import dump_pipeline_config, load_pipeline_config
from src.utils.prediction_tables import load_prediction_snapshot

ROOT = Path(__file__).resolve().parents[2]


def write_sources(root, *, readmission=False):
    directory = root / "data" / "filtered"
    directory.mkdir(parents=True)
    frames = {}
    for origin, source_code in (("mimic", 0), ("tudd", 1)):
        rows = np.arange(64)
        labels = ((rows + source_code) % 4 == 0).astype(int)
        frame = pd.DataFrame(
            {
                # Deliberately overlapping row IDs: identity requires the source.
                "record_id": rows,
                "source": origin,
                "Age": 30 + rows % 50,
                "signal": np.sin(rows * 1.3) + labels * 0.7 + source_code,
                "partly_missing": np.where(rows % 7 == 0, np.nan, rows % 9),
                "ward": np.where(rows % 3 == 0, "medical", "surgical"),
                "mortality": labels,
                "LOS": 48 + rows * 3,
                "LOS3": 1 - labels,
                "LOS7": labels,
                "hours_to_readmit": np.resize([24.0, 72.0, 72.1, np.nan], len(rows)),
                # Deliberately incorrect prederived column must never be used.
                "hours_to_readmit_72": np.zeros(len(rows), dtype=int),
                f"{origin}_only": source_code + 100,
            }
        )
        prefix = "mimic4" if origin == "mimic" else "tudd"
        suffix = "readmission" if readmission else "mean_100_full"
        frame.to_csv(directory / f"{prefix}_{suffix}.csv", index=False)
        frames[origin] = frame
    return frames


def reduced_config(root, *, source="baseline.yaml"):
    data = load_pipeline_config(ROOT / "configs" / "pipeline" / source).model_dump(mode="json")
    data["run_id"] = "synthetic-e2e"
    data["random_states"].update(model_training_seed=17, training_sample_seed=23)
    data["dataset"].update(
        train_size=0.75,
        train_on=[{"dataset": "mimic", "fraction": 24}, {"dataset": "tudd", "fraction": 16}],
    )
    data["dataset"]["data_cleaner"]["outlier_limits_path"] = str(ROOT / "configs" / "data_limits.json")
    data["training"] = [cpu_model()]
    data["mlflow"].update(
        enabled=True,
        tracking_uri=f"sqlite:///{root / 'tracking.db'}",
        artifact_location=(root / "artifacts").as_uri(),
        experiment_name="synthetic-e2e",
    )
    return data


def cpu_model(*, method="grid", values=(0.5,)):
    return {
        "name": "logistic-regression",
        "preprocessing": {
            "imputer": {"imputation_method": "mean"},
            "scaler_encoder": {"type": "standardization"},
        },
        "tuning": {
            "method": method,
            "grid": {"C": list(values), "max_iter": [100], "solver": ["liblinear"]},
            "scoring": "roc_auc",
            "cv": {"n_splits": 2, "shuffle": True},
            "optuna": {"n_trials": 2, "n_startup_trials": 1, "sampler": "random"},
        },
    }


def write_config(data, path):
    config = PipelineConfig.model_validate(data)
    dump_pipeline_config(config, path)
    return config


def labels_for(frame, *, readmission=False):
    return (frame["hours_to_readmit"] <= 72).astype(int) if readmission else frame["mortality"]


def identities(X):
    return set(zip(X["source"], X["record_id"], strict=True))


def expected_holdouts(frames, config, *, readmission=False):
    return {
        origin: train_test_split(
            frame.index,
            test_size=1 - config.dataset.train_size,
            random_state=config.dataset.random_state,
            stratify=labels_for(frame, readmission=readmission),
        )[1].tolist()
        for origin, frame in frames.items()
    }


def assert_training_rows(observation, frames, holdouts, *, readmission=False):
    X, y = observation["X"], observation["y"]
    expected = [
        labels_for(frames[source], readmission=readmission).loc[row]
        for source, row in zip(X.source, X.record_id, strict=True)
    ]
    np.testing.assert_array_equal(y, expected)
    test_ids = {(source, row) for source, rows in holdouts.items() for row in rows}
    assert identities(X).isdisjoint(test_ids)
    excluded = {"mortality", "LOS3", "LOS7", "hours_to_readmit", "hours_to_readmit_72"}
    if not readmission:
        excluded.add("LOS")
    else:
        assert "LOS" in X
    assert excluded.isdisjoint(X.columns)
    assert {"mimic_only", "tudd_only"}.isdisjoint(X.columns)


def load_outputs(config):
    client = MlflowClient(tracking_uri=config.mlflow.tracking_uri)
    experiment = client.get_experiment_by_name(config.mlflow.experiment_name)
    assert experiment is not None
    parents = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.run_type = 'pipeline' and tags.pipeline_id = '{config.run_id}'",
    )
    assert len(parents) == 1
    run = parents[0]
    directory = Path(client.download_artifacts(run.info.run_id, "test_predictions"))
    snapshot = load_prediction_snapshot(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    metrics = pd.read_csv(client.download_artifacts(run.info.run_id, "prediction_metrics.csv"))
    bootstrap = pd.read_csv(client.download_artifacts(run.info.run_id, "bootstrap_metrics.csv"))
    return snapshot.tables, manifest, metrics, bootstrap


def assert_saved_cohort(table, manifest, cohort, row_ids, labels):
    fingerprint = manifest["cohort_fingerprints"][cohort]
    expected_ids = ["ts_" + hashlib.sha256(f"{fingerprint}:{row}".encode()).hexdigest() for row in row_ids]
    assert table.test_set_id.tolist() == expected_ids
    assert table.test_set_id.is_unique
    np.testing.assert_array_equal(table.y_true, labels)


def assert_metric_artifacts(tables, metrics, bootstrap, model_ids):
    points = metrics.loc[metrics.statistic == "point"]
    assert set(zip(points.dataset, points.model_instance, strict=True)) == {
        (cohort, model) for cohort in tables for model in model_ids
    }
    assert len(points) == len(tables) * len(model_ids)
    assert set(bootstrap.dataset) == set(tables)
    assert set(bootstrap.columns) == {"dataset", "metric", "bootstrap_id", *model_ids}
    assert set(bootstrap.metric) == {"roc_auc", "prc_auc"}
    assert np.isfinite(bootstrap[model_ids].to_numpy()).all()
    for _, group in bootstrap.groupby(["dataset", "metric"]):
        assert group.bootstrap_id.tolist() == list(range(32))
    for row in points.itertuples():
        table = tables[row.dataset]
        p = table[f"y_pred_{row.model_instance}"]
        y = table.y_true
        predicted = p >= 0.5
        expected = {
            "roc_auc": roc_auc_score(y, p),
            "prc_auc": average_precision_score(y, p),
            "accuracy": accuracy_score(y, predicted),
            "f1": f1_score(y, predicted, zero_division=0),
            "precision": precision_score(y, predicted, zero_division=0),
            "sensitivity": recall_score(y, predicted, zero_division=0),
        }
        for name, value in expected.items():
            assert getattr(row, name) == pytest.approx(value)
            assert np.isfinite(getattr(row, f"{name}_ci_lower"))
            assert np.isfinite(getattr(row, f"{name}_ci_upper"))
            assert getattr(row, f"{name}_ci_lower") <= getattr(row, f"{name}_ci_upper")
