from datetime import date

import pytest
from pydantic import ValidationError

from src.classes.pipeline import Pipeline
from src.config import config
from src.schemas.dataset_schemas import DataSplitParams, DatasetParams
from src.schemas.evaluation_schemas import EvaluationParams
from src.schemas.pipeline_schemas import PipelineParams
from src.schemas.plotting_schemas import PlottingParams
from src.schemas.training_schemas import ModelParams, TrainingParams
from src.utils.config_io import load_pipeline_params


def _make_pipeline_params(**overrides):
    values = {
        "dataset": DatasetParams(train_on=(DataSplitParams(dataset="mimic"),)),
        "training": TrainingParams(),
        "evaluation": EvaluationParams(),
        "plotting": PlottingParams(),
    }
    values.update(overrides)
    return PipelineParams(**values)


def test_pipeline_params_have_stable_default_name_for_given_date():
    params = _make_pipeline_params(run_number=12, run_date=date(2026, 6, 17))

    assert params.run_id == "0012_2026-06-17"
    assert params.run_dir == config.dir_run_results / "0012_2026-06-17"


def test_pipeline_params_accept_multiple_models():
    params = _make_pipeline_params(
        training=TrainingParams(
            models=(
                ModelParams(name="tabpfn-3"),
                ModelParams(name="tabicl-2"),
            )
        )
    )

    assert [model.name for model in params.training.models] == ["tabpfn-3", "tabicl-2"]


def test_pipeline_params_reject_unknown_config_keys():
    with pytest.raises(ValidationError):
        PipelineParams(unknown=True)


def test_pipeline_rejects_unknown_model_name(tmp_path):
    pipeline = Pipeline(
        _make_pipeline_params(
            training=TrainingParams(models=(ModelParams(name="unknown-model"),)),
        )
    )

    with pytest.raises(ValueError, match="Unknown classification model"):
        pipeline._create_model(pipeline.params.training.models[0])


def test_load_pipeline_params_from_yaml_example():
    params = load_pipeline_params("configs/example_pipeline.yaml")

    assert params.run_id == "0001_2026-06-17"
    assert [source.dataset for source in params.dataset.train_on] == ["mimic", "tudd"]
