"""Override edge contracts not exercised by the valid Cartesian workflow."""

import pytest

from src.classes.experiment_suite import ExperimentSuite
from src.schemas.suite_schemas import ExperimentSuiteConfig, OverrideRangeConfig
from tests.e2e.helpers import ROOT


def test_out_of_range_list_override_rejects_mistyped_experiment_instead_of_silently_retargeting():
    suite = ExperimentSuite(
        ExperimentSuiteConfig.model_validate(
            {
                "name": "invalid-source",
                "base_config": "../pipeline/baseline.yaml",
                "matrix": [{"path": "dataset.train_on.3.fraction", "values": [12]}],
            }
        ),
        ROOT / "configs/suite/invalid.yaml",
    )
    with pytest.raises(ValueError, match="index out of range"):
        suite.expand()


def test_fractional_range_keeps_endpoint_despite_floating_point_drift():
    """A dropped endpoint silently omits an experiment; the E2E uses explicit values."""
    assert OverrideRangeConfig(start=0.0, stop=0.3, step=0.1).values() == pytest.approx((0.0, 0.1, 0.2, 0.3))
