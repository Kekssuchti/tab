import pytest
from pydantic import ValidationError

from src.schemas.training_schemas import (
    ClassificationScoring,
    RegressionScoring,
    TuningConfig,
    scoring_is_lower_better,
)


def test_tuning_config_parses_and_serializes_scoring_enums():
    classification = TuningConfig(scoring="roc_auc")
    regression = TuningConfig(scoring="rmse")

    assert classification.scoring is ClassificationScoring.ROC_AUC
    assert classification.scoring.task_type == "classification"
    assert classification.scoring.optimization_direction == "maximize"
    assert regression.scoring is RegressionScoring.RMSE
    assert regression.scoring.task_type == "regression"
    assert regression.scoring.optimization_direction == "minimize"
    assert classification.model_dump(mode="json")["scoring"] == "roc_auc"
    assert regression.model_dump(mode="json")["scoring"] == "rmse"


def test_tuning_config_rejects_unknown_scoring_metric():
    with pytest.raises(ValidationError, match="scoring"):
        TuningConfig(scoring="unknown")


def test_scoring_metadata_covers_every_metric():
    assert all(scoring.task_type == "classification" for scoring in ClassificationScoring)
    assert all(not scoring.lower_is_better for scoring in ClassificationScoring)
    assert all(scoring.task_type == "regression" for scoring in RegressionScoring)
    assert {scoring.value for scoring in RegressionScoring if scoring.lower_is_better} == {"mae", "mse", "rmse"}


def test_scoring_direction_helper_preserves_unknown_metric_default():
    assert scoring_is_lower_better("mae")
    assert not scoring_is_lower_better("accuracy")
    assert not scoring_is_lower_better("prc_auc")
