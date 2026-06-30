import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import os
    import sys
    from dataclasses import asdict
    from pathlib import Path

    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

    import marimo as mo
    import numpy as np
    import pandas as pd

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.classes.dataset import Dataset
    from src.classes.trainer import Trainer
    from src.config import config
    from src.schemas.dataset_schemas import DatasetParams, DataSplitParams
    from src.schemas.preprocessing_schemas import ImputerParams, ScalerEncoderParams
    from src.schemas.training_schemas import ModelParams
    from src.utils.evaluation_utils import evaluate_classification_predictions
    from src.utils.model_lifecycle import release_training_result_model

    return (
        DataSplitParams,
        Dataset,
        DatasetParams,
        ImputerParams,
        ModelParams,
        ScalerEncoderParams,
        Trainer,
        asdict,
        evaluate_classification_predictions,
        mo,
        np,
        pd,
        release_training_result_model,
    )


@app.cell
def _():
    # Single-model timing notebook. Edit this cell, then set RUN_TRAINING = True.
    # This uses Dataset and Trainer directly, without run_pipeline or MLflow.

    # Dataset variables. Use mimic/tudd for mortality or LOS7; use
    # mimic_readmission/tudd_readmission for hours_to_readmit.
    TARGET = "mortality"
    TRAIN_ON = (
        ("mimic", 1.0),
        ("tudd", 2500),
    )

    TEST_SETS = ("mimic", "tudd")
    RANDOM_STATE = 1337
    TRAIN_SIZE = 0.8
    FORCE_REPREPROCESS = False

    # Dataset-level preprocessing defaults used by Trainer unless overridden below.
    DATASET_IMPUTER = {
        "imputation_method": "mean",
        "flag_missing": False,
    }
    DATASET_SCALER = {"type": "none"}

    # Single model to train. Edit these params to profile foundation-model speed.
    MODEL_NAME = "tabpfn-2.5"
    MODEL_PARAMS = {
        "n_estimators": 8,
        "predict_batch_size": 2048,
        "inference_config": {"SUBSAMPLE_SAMPLES": 15_000},
    }
    # Optional model-specific preprocessing override. Set to None to use dataset defaults.
    MODEL_PREPROCESSING = None
    TASK_TYPE = "classification"

    # Execution controls.
    RUN_PREFLIGHT_VALIDATION = False
    RUN_TRAINING = True
    PREDICTION_REPEATS = 1
    RELEASE_AFTER_EVALUATION = False
    return (
        DATASET_IMPUTER,
        DATASET_SCALER,
        FORCE_REPREPROCESS,
        MODEL_NAME,
        MODEL_PARAMS,
        MODEL_PREPROCESSING,
        PREDICTION_REPEATS,
        RANDOM_STATE,
        RELEASE_AFTER_EVALUATION,
        RUN_PREFLIGHT_VALIDATION,
        RUN_TRAINING,
        TARGET,
        TASK_TYPE,
        TEST_SETS,
        TRAIN_ON,
        TRAIN_SIZE,
    )


@app.cell
def _(
    DATASET_IMPUTER,
    DATASET_SCALER,
    DataSplitParams,
    DatasetParams,
    FORCE_REPREPROCESS,
    ImputerParams,
    MODEL_NAME,
    MODEL_PARAMS,
    MODEL_PREPROCESSING,
    ModelParams,
    RANDOM_STATE,
    ScalerEncoderParams,
    TARGET,
    TASK_TYPE,
    TRAIN_ON,
    TRAIN_SIZE,
):
    dataset_params = DatasetParams(
        target=TARGET,
        random_state=RANDOM_STATE,
        train_size=TRAIN_SIZE,
        train_on=tuple(
            DataSplitParams(dataset=dataset_name, fraction=fraction)
            for dataset_name, fraction in TRAIN_ON
        ),
        classification=TASK_TYPE == "classification",
        force_repreprocess=FORCE_REPREPROCESS,
        imputer=ImputerParams(**DATASET_IMPUTER),
        scaler_encoder=ScalerEncoderParams(**DATASET_SCALER),
    )

    model_params = ModelParams(
        name=MODEL_NAME,
        task_type=TASK_TYPE,
        params=MODEL_PARAMS,
        preprocessing=MODEL_PREPROCESSING,
        tuning=None,
    )
    return dataset_params, model_params


@app.cell
def _(Dataset, asdict, dataset_params, mo, pd):
    dataset = Dataset(dataset_params)
    data = dataset.get_dataset()
    dataset_summary = dataset.summarize(data)

    dataset_table = pd.DataFrame(
        [
            {
                "part": "train",
                "rows": dataset_summary.train.row_count,
                "features": data.train_data.X.shape[1],
                "class_balance": dataset_summary.train.class_balance,
            },
            {
                "part": "test_mimic",
                "rows": dataset_summary.test_mimic.row_count,
                "features": data.test_mimic.X.shape[1],
                "class_balance": dataset_summary.test_mimic.class_balance,
            },
            {
                "part": "test_tudd",
                "rows": dataset_summary.test_tudd.row_count,
                "features": data.test_tudd.X.shape[1],
                "class_balance": dataset_summary.test_tudd.class_balance,
            },
        ]
    )

    mo.vstack(
        [
            mo.md("## Dataset"),
            dataset_table,
            pd.DataFrame(
                asdict(file_summary) for file_summary in dataset_summary.data_files
            ),
        ]
    )
    return (data,)


@app.cell
def _(RUN_PREFLIGHT_VALIDATION, Trainer, dataset_params, model_params):
    trainer = Trainer(
        params=(model_params,),
        default_imputer=dataset_params.imputer,
        default_scaler=dataset_params.scaler_encoder,
    )
    if RUN_PREFLIGHT_VALIDATION:
        trainer.validate_model_configs()
    model_config_checked = True
    return model_config_checked, trainer


@app.cell
def _(RUN_TRAINING, data, mo, model_config_checked, model_params, trainer):
    mo.stop(
        not RUN_TRAINING,
        mo.md("Set `RUN_TRAINING = True` in the variables cell to fit the model."),
    )
    _ = model_config_checked

    y_train = data.train_data.y.to_numpy()
    training_result = trainer.train_model(
        model_params=model_params,
        X_train=data.train_data.X,
        y_train=y_train,
    )
    return (training_result,)


@app.cell
def _(
    PREDICTION_REPEATS,
    RELEASE_AFTER_EVALUATION,
    TEST_SETS,
    asdict,
    data,
    evaluate_classification_predictions,
    model_params,
    np,
    pd,
    release_training_result_model,
    training_result,
):
    if model_params.task_type != "classification":
        raise NotImplementedError(
            "This notebook currently evaluates classification models only"
        )
    if PREDICTION_REPEATS < 1:
        raise ValueError("PREDICTION_REPEATS must be at least 1")
    if training_result.trained_model is None:
        raise RuntimeError("The trained model has already been released")

    available_test_sets = {
        "mimic": data.test_mimic,
        "tudd": data.test_tudd,
    }

    evaluation_rows = []
    try:
        for dataset_name in TEST_SETS:
            test_set = available_test_sets[dataset_name]
            timings = []
            predictions = None
            for _ in range(PREDICTION_REPEATS):
                predictions, predict_time = training_result.trained_model.predict(
                    test_set.X
                )
                timings.append(predict_time)

            metrics = evaluate_classification_predictions(
                predictions,
                test_set.y.to_numpy(),
            )
            mean_predict_time = float(np.mean(timings))
            evaluation_rows.append(
                {
                    "dataset": dataset_name,
                    "rows": len(test_set.y),
                    "predict_repeats": PREDICTION_REPEATS,
                    "predict_time_mean_s": mean_predict_time,
                    "rows_per_second_mean": len(test_set.y) / mean_predict_time,
                    **asdict(metrics),
                }
            )
    finally:
        if RELEASE_AFTER_EVALUATION:
            release_training_result_model(training_result)

    evaluation_table = pd.DataFrame(evaluation_rows)
    return (evaluation_table,)


@app.cell
def _(asdict, mo, pd, training_result):
    training_row = {
        "model": training_result.model_name,
        "fit_time_s": training_result.fit_time,
        "tuned": training_result.tuned,
    }
    if training_result.training_metrics is not None:
        training_row.update(
            {
                f"train_{key}": value
                for key, value in asdict(training_result.training_metrics).items()
            }
        )

    mo.vstack(
        [
            mo.md("## Training"),
            pd.DataFrame([training_row]),
        ]
    )
    return


@app.cell
def _(evaluation_table, mo):
    mo.vstack(
        [
            mo.md("## Test Timing And Metrics"),
            evaluation_table,
        ]
    )
    return


@app.cell
def _(MODEL_PARAMS, evaluation_table):
    print(
        f"& {MODEL_PARAMS.get('n_estimators', '')} & {round(evaluation_table['rows_per_second_mean'][0]):,} / {round(evaluation_table['rows_per_second_mean'][1]):,} \\\\"
    )
    return


if __name__ == "__main__":
    app.run()
