from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.classes import dataset as dataset_module
from src.classes.pipeline import Pipeline
from src.schemas import pipeline_schemas
from src.utils.config_io import load_pipeline_params


def _make_dataset(start_id: int, n_rows: int) -> pd.DataFrame:
    labels = np.array([0, 1] * (n_rows // 2))
    signal = labels * 4.0 + np.linspace(0.0, 0.5, n_rows)
    return pd.DataFrame(
        {
            "record_id": np.arange(start_id, start_id + n_rows),
            "Age": 40 + np.arange(n_rows),
            "signal": signal,
            "noise": np.linspace(1.0, 2.0, n_rows),
            "mortality": labels,
            "LOS": 24.0 + np.arange(n_rows),
            "LOS7": labels,
            "hours_to_readmit": [12.0 if label else None for label in labels],
        }
    )


def _write_filtered_data(tmp_path) -> None:
    filtered_path = tmp_path / "filtered"
    filtered_path.mkdir()

    mimic = _make_dataset(1000, 20)
    tudd = _make_dataset(2000, 20)

    mimic.to_csv(filtered_path / "mimic4_mean_100_full.csv", index=False)
    tudd.to_csv(filtered_path / "tudd_mean_100_full.csv", index=False)
    mimic.to_csv(filtered_path / "mimic4_readmission.csv", index=False)
    tudd.to_csv(filtered_path / "tudd_readmission.csv", index=False)


def test_pipeline_runs_end_to_end_with_tuned_sklearn_and_tfm_models(
    tmp_path, monkeypatch
):
    _write_filtered_data(tmp_path)
    monkeypatch.setattr(dataset_module, "config", SimpleNamespace(dir_data=tmp_path))
    monkeypatch.setattr(
        pipeline_schemas,
        "config",
        SimpleNamespace(dir_run_results=tmp_path / "runs"),
    )

    params = load_pipeline_params("tests/integration/pipeline_integration_config.yaml")
    result = Pipeline(params).run()

    assert result.run_id == params.run_id
    assert result.total_time >= 0.0
    assert result.dataset_summary.train.row_count == 24
    assert result.dataset_summary.test_mimic.row_count == 8
    assert result.dataset_summary.test_tudd.row_count == 8
    assert {model.model_name for model in result.model_results} == {
        "xgboost",
        "tabpfn-3",
    }
    assert len(result.training_results) == 2

    for training_result in result.training_results:
        assert training_result.tuned
        assert training_result.tuning_result is not None
        assert training_result.tuning_result.best_score >= 0.0
        assert training_result.tuning_result.cv_results.params
        assert training_result.tuning_result.cv_results.mean_metrics
        assert len(training_result.tuning_result.fold_results) <= 4

    for model_result in result.model_results:
        assert set(model_result.metrics_by_test_set) == {"mimic", "tudd"}
        assert model_result.fit_time >= 0.0
        assert model_result.total_time >= model_result.fit_time
        assert (
            model_result.final_test_metrics.mimic_test
            is model_result.metrics_by_test_set["mimic"]
        )
        assert (
            model_result.final_test_metrics.tudd_test
            is model_result.metrics_by_test_set["tudd"]
        )
        for test_result in model_result.test_results:
            assert test_result.predict_time >= 0.0
            assert 0.0 <= test_result.metrics.accuracy <= 1.0
            assert 0.0 <= test_result.metrics.f1 <= 1.0
            assert test_result.metrics.roc_auc is not None
        assert -1.0 <= model_result.final_test_metrics.mimic_minus_tudd.accuracy <= 1.0
