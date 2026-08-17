import numpy as np
import pandas as pd

from src.classes.preprocessor import Preprocessor
from src.schemas.dataset_schemas import DatasetBundle, XYDataset
from src.schemas.preprocessing_schemas import ImputerConfig, ScalerEncoderConfig


def _bundle(X_train: pd.DataFrame, X_test: pd.DataFrame | None = None) -> DatasetBundle:
    if X_test is None:
        X_test = X_train.copy()

    return DatasetBundle(
        train_data=XYDataset(X=X_train, y=pd.Series([0] * len(X_train))),
        test_mimic=XYDataset(X=X_test, y=pd.Series([0] * len(X_test))),
        test_tudd=XYDataset(X=X_test.copy(), y=pd.Series([0] * len(X_test))),
    )


def _preprocessor(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
    imputation_method: str = "none",
    scaler_type: str = "none",
    flag_missing: bool = False,
    knn_neighbors: int = 2,
) -> Preprocessor:
    return Preprocessor(
        imputer_config=ImputerConfig(
            imputation_method=imputation_method,
            flag_missing=flag_missing,
            knn_neighbors=knn_neighbors,
        ),
        scaler_config=ScalerEncoderConfig(type=scaler_type),
    )


def test_no_preprocessing_returns_input_values_unchanged():
    X_train = pd.DataFrame(
        {
            "age": [40.0, 50.0, 60.0],
            "heart_rate": [80.0, 90.0, 100.0],
        }
    )
    pipeline = _preprocessor(X_train).build_pipeline()

    transformed = pipeline.fit_transform(X_train)

    np.testing.assert_allclose(np.asarray(transformed), X_train.to_numpy())


def test_categorical_columns_are_encoded_for_numeric_estimators():
    X_train = pd.DataFrame(
        {
            "age": [40.0, 50.0, 60.0],
            "Sex": ["F", "M", "F"],
        }
    )
    X_test = pd.DataFrame(
        {
            "age": [70.0],
            "Sex": ["unknown"],
        }
    )
    pipeline = _preprocessor(X_train, X_test).build_pipeline()

    transformed_train = pipeline.fit_transform(X_train)
    transformed_test = pipeline.transform(X_test)

    assert transformed_train.shape == (3, 3)
    assert transformed_test.shape == (1, 3)
    assert transformed_train.dtype.kind in {"f", "i"}
    assert transformed_test.dtype.kind in {"f", "i"}


def test_mean_imputation_uses_training_values_for_train_and_test_data():
    X_train = pd.DataFrame(
        {
            "lab_value": [1.0, np.nan, 7.0, 10.0],
            "age": [40.0, 50.0, 60.0, 70.0],
        }
    )
    X_test = pd.DataFrame(
        {
            "lab_value": [np.nan, 100.0],
            "age": [80.0, 90.0],
        }
    )
    pipeline = _preprocessor(X_train, X_test, imputation_method="mean").build_pipeline()

    transformed_train = pipeline.fit_transform(X_train)
    transformed_test = pipeline.transform(X_test)

    np.testing.assert_allclose(
        transformed_train,
        np.array(
            [
                [1.0, 40.0],
                [6.0, 50.0],
                [7.0, 60.0],
                [10.0, 70.0],
            ]
        ),
    )
    np.testing.assert_allclose(transformed_test, np.array([[6.0, 80.0], [100.0, 90.0]]))


def test_median_imputation_differs_from_mean_when_distribution_is_skewed():
    X_train = pd.DataFrame(
        {
            "lab_value": [1.0, np.nan, 7.0, 10.0],
            "age": [40.0, 50.0, 60.0, 70.0],
        }
    )
    pipeline = _preprocessor(X_train, imputation_method="median").build_pipeline()

    transformed = pipeline.fit_transform(X_train)

    assert transformed[1, 0] == 7.0


def test_missingness_flags_are_added_only_when_requested():
    X_train = pd.DataFrame(
        {
            "lab_value": [1.0, np.nan, 7.0],
            "age": [40.0, 50.0, 60.0],
        }
    )
    X_test = pd.DataFrame(
        {
            "lab_value": [np.nan, 10.0],
            "age": [70.0, 80.0],
        }
    )
    without_flags = _preprocessor(X_train, X_test, imputation_method="mean", flag_missing=False).build_pipeline()
    with_flags = _preprocessor(X_train, X_test, imputation_method="mean", flag_missing=True).build_pipeline()

    assert without_flags.fit_transform(X_train).shape == (3, 2)
    transformed_train = with_flags.fit_transform(X_train)
    transformed_test = with_flags.transform(X_test)

    assert transformed_train.shape == (3, 3)
    assert transformed_train[:, -1].tolist() == [0.0, 1.0, 0.0]
    assert transformed_test[:, -1].tolist() == [1.0, 0.0]


def test_standardization_learns_statistics_from_training_data_only():
    X_train = pd.DataFrame(
        {
            "lab_value": [1.0, 3.0, 5.0],
            "age": [10.0, 20.0, 30.0],
        }
    )
    X_test = pd.DataFrame(
        {
            "lab_value": [7.0],
            "age": [40.0],
        }
    )
    pipeline = _preprocessor(X_train, X_test, scaler_type="standardization").build_pipeline()

    transformed_train = pipeline.fit_transform(X_train)
    transformed_test = pipeline.transform(X_test)

    np.testing.assert_allclose(transformed_train.mean(axis=0), [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(transformed_train.std(axis=0), [1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(
        transformed_test,
        np.array([[(7.0 - 3.0) / np.sqrt(8.0 / 3.0), (40.0 - 20.0) / np.sqrt(200.0 / 3.0)]]),
    )


def test_knn_imputation_outputs_complete_data_with_original_shape():
    X_train = pd.DataFrame(
        {
            "lab_value": [1.0, 5.0, np.nan, 10.0],
            "age": [40.0, 50.0, 42.0, 80.0],
        }
    )
    pipeline = _preprocessor(X_train, imputation_method="knn", knn_neighbors=2).build_pipeline()

    transformed = pipeline.fit_transform(X_train)

    assert transformed.shape == X_train.shape
    assert np.isfinite(transformed).all()
