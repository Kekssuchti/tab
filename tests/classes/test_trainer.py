import numpy as np
import pandas as pd

from src.classes.preprocessor import Preprocessor
from src.classes.trainer import Trainer
from src.schemas.preprocessing_schemas import ImputerParams, ScalerEncoderParams
from src.schemas.training_schemas import CVParams, HPOParams, ModelParams, TrainingParams


def _classification_data():
    X = pd.DataFrame(
        {
            "age": [30, 35, 40, 45, 60, 65, 70, 75],
            "lab": [1.0, 1.2, 1.5, 1.7, 4.0, 4.2, 4.5, 4.7],
        }
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


def _regression_data():
    X = pd.DataFrame(
        {
            "age": [30, 35, 40, 45, 60, 65],
            "lab": [1.0, 1.5, 2.0, 2.5, 4.0, 4.5],
        }
    )
    y = np.array([10.0, 12.0, 14.0, 16.0, 24.0, 26.0])
    return X, y


def _preprocess_pipeline():
    return Preprocessor(
        params_imputer=ImputerParams(imputation_method="none"),
        params_scaler=ScalerEncoderParams(type="none"),
    ).build_pipeline()


def test_trainer_returns_adapter_that_predicts_after_pipeline_training():
    X, y = _classification_data()
    trainer = Trainer(
        params=TrainingParams(
            models=(
                ModelParams(
                    name="logistic-regression",
                    task_type="classification",
                    params={"max_iter": 200},
                ),
            )
        ),
        preprocess_pipeline=_preprocess_pipeline(),
    )

    result = trainer.train_models(X, y)[0]
    predictions, predict_time = result.trained_model.predict(X)

    assert result.model_name == "logistic-regression"
    assert not result.optimized_hyperparameters
    assert result.fit_time >= 0
    assert predict_time >= 0
    assert predictions.shape == (len(X), 2)
    assert np.isfinite(predictions).all()


def test_trainer_uses_model_specific_hpo_grid_and_returns_best_params():
    X, y = _classification_data()
    trainer = Trainer(
        params=TrainingParams(
            models=(
                ModelParams(
                    name="logistic-regression",
                    task_type="classification",
                    params={"max_iter": 200},
                    hpo=HPOParams(
                        search_grid={"C": [0.1, 1.0]},
                        scoring="accuracy",
                        cv=CVParams(n_splits=2, random_state=1),
                    ),
                ),
            )
        ),
        preprocess_pipeline=_preprocess_pipeline(),
    )

    result = trainer.train_models(X, y)[0]
    predictions, _ = result.trained_model.predict(X)

    assert result.optimized_hyperparameters
    assert result.hpo_result is not None
    assert set(result.hpo_result.best_params) == {"C"}
    assert result.hpo_result.best_score >= 0
    assert predictions.shape == (len(X), 2)


def test_trainer_can_fit_regression_adapter_behind_same_interface():
    X, y = _regression_data()
    trainer = Trainer(
        params=TrainingParams(
            models=(
                ModelParams(
                    name="linear-regression",
                    task_type="regression",
                ),
            )
        ),
        preprocess_pipeline=_preprocess_pipeline(),
    )

    result = trainer.train_models(X, y)[0]
    predictions, _ = result.trained_model.predict(X)

    assert result.model_name == "linear-regression"
    assert predictions.shape == (len(X),)
    assert np.isfinite(predictions).all()
