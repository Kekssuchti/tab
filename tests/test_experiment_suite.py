from types import SimpleNamespace

import pytest

from src.classes.experiment_suite import ExperimentSuite
from src.run_pipeline import run_suite
from src.schemas.suite_schemas import OverrideRangeConfig
from src.utils.config_io import load_experiment_suite_config


def _write_base_config(path):
    path.write_text(
        """
run_id: base-run
dataset:
  target: mortality
  train_size: 0.8
  train_on:
    - dataset: mimic
      fraction: 1.0
training:
  - name: logistic-regression
mlflow:
  enabled: false
""".strip(),
        encoding="utf-8",
    )


def _write_suite_config(path, *, override_path="dataset.train_on.0.fraction"):
    path.write_text(
        f"""
name: training-size
base_config: base.yaml
matrix:
  - path: {override_path}
    range:
      start: 500
      stop: 1500
      step: 500
""".strip(),
        encoding="utf-8",
    )


def test_experiment_suite_expands_range_and_summarizes_dry_run(tmp_path):
    base_path = tmp_path / "base.yaml"
    suite_path = tmp_path / "suite.yaml"
    _write_base_config(base_path)
    _write_suite_config(suite_path)

    suite_params = load_experiment_suite_config(suite_path)
    summary = ExperimentSuite(suite_params, suite_path).dry_run_summary()

    assert summary.config_count == 3
    assert summary.models_per_config == 1
    assert summary.total_model_runs == 3
    assert summary.changed_parameters == ("dataset.train_on.0.fraction",)
    assert [variant.pipeline_config.dataset.train_on[0].fraction for variant in summary.config_variants] == [
        500,
        1000,
        1500,
    ]
    assert [variant.variant_id for variant in summary.config_variants] == [
        "fraction-500",
        "fraction-1000",
        "fraction-1500",
    ]
    assert summary.config_variants[0].pipeline_config.run_id == "base-run_training-size_fraction-500"
    assert summary.config_variants[0].pipeline_config.mlflow.run_name == "training-size/fraction-500"
    assert "Configs: 3" in summary.format()
    assert "Changed parameters: dataset.train_on.0.fraction" in summary.format()


def test_experiment_suite_rejects_invalid_override_path(tmp_path):
    base_path = tmp_path / "base.yaml"
    suite_path = tmp_path / "suite.yaml"
    _write_base_config(base_path)
    _write_suite_config(suite_path, override_path="dataset.train_on.3.fraction")

    suite_params = load_experiment_suite_config(suite_path)
    suite = ExperimentSuite(suite_params, suite_path)

    with pytest.raises(ValueError, match="index out of range"):
        suite.expand()


def test_run_suite_dry_run_and_execution_use_concrete_configs(
    tmp_path,
    monkeypatch,
    capsys,
):
    base_path = tmp_path / "base.yaml"
    suite_path = tmp_path / "suite.yaml"
    _write_base_config(base_path)
    _write_suite_config(suite_path)
    calls = []

    def _fake_run_pipeline_params(params, *, config_path=None):
        calls.append(
            {
                "fraction": params.dataset.train_on[0].fraction,
                "config_text": config_path.read_text(encoding="utf-8"),
            }
        )
        return SimpleNamespace(run_id=params.run_id)

    dry_run_summary = run_suite(suite_path, dry_run=True)
    captured = capsys.readouterr()

    assert dry_run_summary.config_count == 3
    assert "Configs: 3" in captured.out
    assert "Total model runs: 3" in captured.out

    monkeypatch.setattr(
        "src.run_pipeline.run_pipeline_params",
        _fake_run_pipeline_params,
    )
    result = run_suite(suite_path)

    assert len(result.results) == 3
    assert [call["fraction"] for call in calls] == [500, 1000, 1500]
    assert "fraction: 500" in calls[0]["config_text"]
    assert "run_id: base-run_training-size_fraction-500" in calls[0]["config_text"]


def test_override_range_values_include_stop_on_float_steps():
    values = OverrideRangeConfig(start=0.0, stop=0.3, step=0.1).values()

    assert len(values) == 4
    assert values[0] == pytest.approx(0.0)
    assert values[-1] == pytest.approx(0.3)
