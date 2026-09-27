"""Construction-only checks: no data access, trial execution, or model imports."""

import json
from math import prod

import pytest

from src.classes.data_registry import dataset_task_for_target
from src.classes.experiment_suite import ExperimentSuite
from src.schemas.pipeline_schemas import PipelineConfig
from src.utils.config_io import load_experiment_suite_config, load_pipeline_config
from src.utils.model_registry import ModelSpec, get_model_spec
from tests.e2e.helpers import ROOT


def test_all_supported_repository_yaml_configs_construct_with_registered_search_spaces(monkeypatch):
    # All current YAMLs are supported. If archives appear, classify them explicitly
    # rather than excluding broken examples by name.
    failures = []

    def forbid_model_construction(*args, **kwargs):
        pytest.fail("Supported-config validation must not instantiate model backends")

    monkeypatch.setattr(ModelSpec, "create", forbid_model_construction)

    def validate(config):
        PipelineConfig.model_validate(config.model_dump(mode="json"))
        assert dataset_task_for_target(config.dataset.target).task_type == "classification"
        assert config.training
        for model in config.training:
            spec = get_model_spec(model, "classification")
            assert model.tuning.scoring.task_type == "classification"
            # Check the named reference even when an explicit grid overrides it.
            spec.tuning_search_space(model.tuning.search_space, None)
            if model.tuning.method == "grid":
                assert spec.tuning_candidates(model.tuning.search_space, model.tuning.grid)
            else:
                assert spec.tuning_search_space(model.tuning.search_space, model.tuning.grid)

    pipeline_paths = sorted((ROOT / "configs/pipeline").glob("*.yaml"))
    suite_paths = sorted((ROOT / "configs/suite").glob("*.yaml"))
    assert pipeline_paths and suite_paths
    for path in pipeline_paths:
        try:
            validate(load_pipeline_config(path))
        except (ValueError, AssertionError) as exc:
            failures.append(f"{path.name}: {exc}")
    for path in suite_paths:
        try:
            suite_config = load_experiment_suite_config(path)
            summary = ExperimentSuite(suite_config, path).dry_run_summary()
            variants = summary.config_variants
            expected_count = prod(len(axis.expanded_values()) for axis in suite_config.matrix)
            assert summary.config_count == len(variants) == expected_count
            model_counts = [len(v.pipeline_config.training) for v in variants]
            assert summary.total_model_runs == sum(model_counts)
            if len(set(model_counts)) == 1:
                assert summary.models_per_config == model_counts[0]
            assert len({v.variant_id for v in variants}) == len(variants)
            assert len({v.pipeline_config.run_id for v in variants}) == len(variants)
            assert len({json.dumps(v.overrides, sort_keys=True) for v in variants}) == len(variants)
            for variant in variants:
                validate(variant.pipeline_config)
        except (ValueError, AssertionError) as exc:
            failures.append(f"{path.name}: {exc}")
    assert not failures, "Supported configuration defects:\n" + "\n".join(failures)
