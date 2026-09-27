from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
import yaml

from src.run_pipeline import run_pipeline, run_suite
from src.utils.config_io import load_pipeline_config
from tests.e2e.helpers import (
    ROOT,
    assert_metric_artifacts,
    assert_saved_cohort,
    assert_training_rows,
    cpu_model,
    expected_holdouts,
    identities,
    labels_for,
    load_outputs,
    reduced_config,
    write_config,
    write_sources,
)


def test_ordinary_yaml_runs_grid_optuna_and_failure_continuation_through_saved_evaluation(
    isolated_runtime, observed_adapter
):
    root = isolated_runtime
    frames = write_sources(root)
    data = reduced_config(root)
    invalid = cpu_model()
    invalid["tuning"]["grid"] = {"nonexistent_hyperparameter": [1]}
    data["training"] = [cpu_model(values=(0.1, 1.0)), invalid, cpu_model(method="optuna", values=(0.3, 3.0))]
    path = root / "ordinary.yaml"
    config = write_config(data, path)

    result = run_pipeline(path)

    assert [run.succeeded for run in result.model_runs] == [True, False, True]
    assert "nonexistent_hyperparameter" in result.model_runs[1].training_result.error
    successful = [result.model_runs[0], result.model_runs[2]]
    model_ids = [run.model_instance_id for run in successful]
    assert len({run.model_instance_id for run in result.model_runs}) == 3
    assert [run.training_result.tuning_result.method for run in successful] == ["grid", "optuna"]
    for run in successful:
        assert len(run.training_result.tuning_result.fold_results) == 4
        assert set(run.evaluation.metrics_by_test_set) == {"mimic", "tudd"}
    assert result.dataset_summary.train.row_count == 40
    assert result.dataset_summary.test_mimic.row_count == result.dataset_summary.test_tudd.row_count == 16
    holdouts = expected_holdouts(frames, config)
    final_fits = [fit for fit in observed_adapter["fits"] if len(fit["X"]) == 40]
    assert len(final_fits) == 2
    for fit in observed_adapter["fits"]:
        assert_training_rows(fit, frames, holdouts)
    for fit in final_fits:
        assert fit["X"].source.value_counts().to_dict() == {"mimic": 24, "tudd": 16}

    tables, manifest, metrics, bootstrap = load_outputs(config)
    assert set(tables) == {"mimic", "tudd"}
    for origin, table in tables.items():
        assert set(table.columns) == {"test_set_id", "y_true", *(f"y_pred_{model}" for model in model_ids)}
        assert_saved_cohort(table, manifest, origin, holdouts[origin], frames[origin].loc[holdouts[origin], "mortality"])
        predictions = [
            obs for obs in observed_adapter["predictions"]
            if len(obs["X"]) == 16 and set(obs["X"].source) == {origin}
        ]
        assert len(predictions) == 2
        for model_id, observation in zip(model_ids, predictions, strict=True):
            assert observation["X"].record_id.tolist() == holdouts[origin]
            np.testing.assert_allclose(table[f"y_pred_{model_id}"], observation["values"][:, 1])
    assert set(tables["mimic"].test_set_id).isdisjoint(tables["tudd"].test_set_id)
    assert_metric_artifacts(tables, metrics, bootstrap, model_ids)
    deltas = metrics.loc[metrics.statistic == "difference"].set_index("model_instance")
    assert set(deltas.index) == set(model_ids)
    for model in model_ids:
        points = metrics.loc[(metrics.statistic == "point") & (metrics.model_instance == model)].set_index("dataset")
        assert np.isclose(deltas.loc[model, "roc_auc"], points.loc["mimic", "roc_auc"] - points.loc["tudd", "roc_auc"])


def test_relative_yaml_suite_executes_cartesian_sizes_and_seeds_with_fixed_holdouts(isolated_runtime, observed_adapter):
    root = isolated_runtime
    frames = write_sources(root)
    data = reduced_config(root)
    data["dataset"]["train_on"] = [{"dataset": "tudd", "fraction": 16}]
    base_path = root / "pipeline" / "baseline.yaml"
    base = write_config(data, base_path)
    suite_data = yaml.safe_load((ROOT / "configs/suite/sample_size.yaml").read_text())
    suite_data["base_config"] = "../pipeline/baseline.yaml"
    for axis in suite_data["matrix"]:
        if axis["path"] == "dataset.train_on.0.fraction":
            axis["values"] = [16, 24]
        elif axis["path"] == "random_states":
            axis["values"] = [{"training_sample_seed": seed, "model_training_seed": seed} for seed in (31, 47)]
        elif axis["path"] == "mlflow.experiment_name":
            axis["values"] = ["synthetic-suite"]
    # A nested mapping inside a model-list entry must preserve unspecified grid keys.
    suite_data["matrix"].append({"path": "training.0.tuning.grid", "values": [{"C": [0.7]}]})
    suite_path = root / "suite" / "sample_size.yaml"
    suite_path.parent.mkdir()
    suite_path.write_text(yaml.safe_dump(suite_data))

    planned = run_suite(suite_path, dry_run=True)
    assert observed_adapter == {"fits": [], "predictions": []}
    result = run_suite(suite_path)

    expected_combinations = set(product((16, 24), (31, 47)))
    variants = result.summary.config_variants
    assert result.summary.config_count == planned.config_count == 4
    assert result.summary.total_model_runs == planned.total_model_runs == 4
    assert [variant.pipeline_config.model_dump() for variant in variants] == [
        variant.pipeline_config.model_dump() for variant in planned.config_variants
    ]
    assert {
        (v.pipeline_config.dataset.train_on[0].fraction, v.pipeline_config.random_states.training_sample_seed)
        for v in variants
    } == expected_combinations
    assert len({v.variant_id for v in variants}) == len({v.pipeline_config.run_id for v in variants}) == 4
    assert {record.run_id for record in result.results} == {v.pipeline_config.run_id for v in variants}
    assert len(observed_adapter["fits"]) == 4
    holdouts = expected_holdouts(frames, base)
    tables_by_cohort = {"mimic": [], "tudd": []}
    training_ids = {}
    for variant, record, fit in zip(variants, result.results, observed_adapter["fits"], strict=True):
        config = variant.pipeline_config
        size, seed = config.dataset.train_on[0].fraction, config.random_states.training_sample_seed
        assert all(run.succeeded for run in record.model_runs)
        assert record.dataset_summary.train.row_count == len(fit["X"]) == size
        assert fit["seed"] == seed == config.random_states.model_training_seed
        assert set(fit["X"].source) == {"tudd"}
        assert_training_rows(fit, frames, holdouts)
        training_ids[size, seed] = identities(fit["X"])
        assert config.dataset.random_state == base.dataset.random_state
        assert config.random_states.evaluation_bootstrap_seed == base.random_states.evaluation_bootstrap_seed
        assert config.random_states.cv_split_seed == base.random_states.cv_split_seed
        assert config.random_states.tuning_sampler_seed == base.random_states.tuning_sampler_seed
        assert config.training[0].tuning.grid == {**base.training[0].tuning.grid, "C": [0.7]}
        tables, manifest, metrics, bootstrap = load_outputs(config)
        for origin, table in tables.items():
            assert_saved_cohort(table, manifest, origin, holdouts[origin], frames[origin].loc[holdouts[origin], "mortality"])
            tables_by_cohort[origin].append(table[["test_set_id", "y_true"]])
        assert_metric_artifacts(tables, metrics, bootstrap, [record.model_runs[0].model_instance_id])
    for size in (16, 24):
        assert training_ids[size, 31] != training_ids[size, 47]
    for cohort_tables in tables_by_cohort.values():
        for table in cohort_tables[1:]:
            pd.testing.assert_frame_equal(table, cohort_tables[0])

    before = [deepcopy(v.pipeline_config.model_dump()) for v in variants]
    variants[0].pipeline_config.training[0].tuning.grid["C"].append(999)
    variants[0].pipeline_config.random_states.training_sample_seed = 999
    assert [v.pipeline_config.model_dump() for v in variants[1:]] == before[1:]
    assert load_pipeline_config(base_path).model_dump() == base.model_dump()
    assert planned.config_variants[0].pipeline_config.model_dump() == before[0]


def test_retriever_readmission_yaml_emits_only_aligned_query_cohort(isolated_runtime, observed_adapter):
    root = isolated_runtime
    frames = write_sources(root, readmission=True)
    data = reduced_config(root, source="retriever.yaml")
    data["dataset"]["target"] = "hours_to_readmit_72"
    data["dataset"]["custom_retriever"].update(
        train_size=20,
        test_on=[{"dataset": "mimic", "fraction": 6}, {"dataset": "tudd", "fraction": 8}],
        selection_strategy="random",
        test_sample_seed=53,
    )
    path = root / "retriever.yaml"
    config = write_config(data, path)

    result = run_pipeline(path)

    assert len(result.model_runs) == 1 and result.model_runs[0].succeeded
    assert result.dataset_summary.target == "hours_to_readmit_72"
    assert {file.file_name for file in result.dataset_summary.data_files} == {
        "mimic4_readmission.csv", "tudd_readmission.csv"
    }
    assert result.dataset_summary.train.row_count == 20
    assert len(observed_adapter["fits"]) == len(observed_adapter["predictions"]) == 1
    fit = observed_adapter["fits"][0]
    query = observed_adapter["predictions"][0]
    holdouts = expected_holdouts(frames, config, readmission=True)
    assert_training_rows(fit, frames, holdouts, readmission=True)
    assert len(fit["X"]) == len(identities(fit["X"])) == 20
    assert query["X"].source.value_counts().to_dict() == {"tudd": 8, "mimic": 6}
    assert identities(fit["X"]).isdisjoint(identities(query["X"]))
    for origin in frames:
        assert set(query["X"].loc[query["X"].source == origin, "record_id"]) <= set(holdouts[origin])
    labels = [
        labels_for(frames[source], readmission=True).loc[row]
        for source, row in zip(query["X"].source, query["X"].record_id, strict=True)
    ]
    assert query["X"].index.tolist() == [
        2 * row + (source == "tudd")
        for source, row in zip(query["X"].source, query["X"].record_id, strict=True)
    ]

    tables, manifest, metrics, bootstrap = load_outputs(config)
    assert set(tables) == {"retriever"}
    assert set(result.model_runs[0].evaluation.metrics_by_test_set) == {"retriever"}
    assert set(metrics.dataset) == {"retriever"}
    assert_saved_cohort(tables["retriever"], manifest, "retriever", query["X"].index, labels)
    model_id = result.model_runs[0].model_instance_id
    np.testing.assert_allclose(tables["retriever"][f"y_pred_{model_id}"], query["values"][:, 1])
    assert_metric_artifacts(tables, metrics, bootstrap, [model_id])
