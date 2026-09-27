"""Transform values, not sklearn object layout; trainer tests own fold-local fitting."""

import numpy as np
import pandas as pd
import pytest

from src.classes.preprocessor import Preprocessor
from src.schemas.preprocessing_schemas import ImputerConfig, ScalerEncoderConfig


@pytest.mark.parametrize(("method", "fill"), [("mean", 6.0), ("median", 7.0)])
def test_imputation_scaling_and_categories_learn_only_from_training_rows(method, fill):
    """Distribution shifts expose test-data fitting; unknown categories must remain usable."""
    training = pd.DataFrame(
        {"lab": [1.0, np.nan, 7.0, 10.0], "age": [10.0, 20.0, 30.0, 40.0], "ward": ["A", "B", "A", "B"]}
    )
    heldout = pd.DataFrame({"lab": [np.nan, 1000.0], "age": [100.0, 200.0], "ward": ["new", "B"]})
    pipeline = Preprocessor(
        ImputerConfig(imputation_method=method, flag_missing=True),
        ScalerEncoderConfig(type="standardization"),
    ).build_pipeline()

    transformed_training = pipeline.fit_transform(training)
    transformed_heldout = pipeline.transform(heldout)

    # Independent arithmetic oracle, including scaling AFTER imputation and indicators.
    numeric_train = np.array([[1, 10, 0], [fill, 20, 1], [7, 30, 0], [10, 40, 0]])
    numeric_test = np.array([[fill, 100, 1], [1000, 200, 0]])
    center, scale = numeric_train.mean(axis=0), numeric_train.std(axis=0)
    expected_train = np.column_stack(((numeric_train - center) / scale, [[1, 0], [0, 1], [1, 0], [0, 1]]))
    expected_test = np.column_stack(((numeric_test - center) / scale, [[0, 0], [0, 1]]))
    np.testing.assert_allclose(transformed_training, expected_train)
    np.testing.assert_allclose(transformed_heldout, expected_test)
    assert np.isfinite(transformed_heldout).all()
    # Transforming another cohort must not refit or make a row depend on its batchmates.
    extreme = heldout.assign(lab=1e9, age=-1e9, ward="another unseen ward")
    pipeline.transform(extreme)
    np.testing.assert_allclose(pipeline.transform(training), expected_train)
    np.testing.assert_allclose(pipeline.transform(heldout.iloc[:1]), expected_test[:1])


def test_knn_neighbor_configuration_uses_training_donors_not_other_query_rows():
    """A shape-only check would miss ignored neighbor counts and heldout donor leakage."""
    training = pd.DataFrame({"age": [0.0, 2.0, 10.0], "lab": [10.0, 30.0, 100.0]})
    heldout = pd.DataFrame({"age": [0.5, 0.5], "lab": [np.nan, 10000.0]})
    for neighbors, expected_fill in ((1, 10.0), (2, 15.0)):
        pipeline = Preprocessor(
            ImputerConfig(imputation_method="knn", knn_neighbors=neighbors),
            ScalerEncoderConfig(type="none"),
        ).build_pipeline()
        pipeline.fit(training)
        # Two distance-weighted donors at distances .5 and 1.5 give (3*10+30)/4.
        np.testing.assert_allclose(pipeline.transform(heldout), [[0.5, expected_fill], [0.5, 10000]])
        np.testing.assert_allclose(pipeline.transform(heldout.iloc[:1]), [[0.5, expected_fill]])
