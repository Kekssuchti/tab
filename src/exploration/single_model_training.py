import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import os
    import sys
    from dataclasses import asdict
    from itertools import product
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
    from src.schemas.dataset_schemas import DatasetParams, DataSplitParams
    from src.schemas.preprocessing_schemas import ImputerParams, ScalerEncoderParams
    from src.schemas.training_schemas import ModelParams
    from src.utils.evaluation_utils import evaluate_classification_predictions
    from src.utils.model_lifecycle import release_model
    from src.utils.model_registry import get_model_spec

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
        get_model_spec,
        mo,
        np,
        pd,
        product,
        release_model,
    )


@app.cell
def _():
    # Model timing notebook. Edit this cell, then set RUN_TRAINING = True.
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

    MODEL_NAME = "tabpfn-2.5"
    MODEL_PARAMS = {
        "n_estimators": [4],
        "predict_batch_size": 2048,
    }
    # Optional model-specific preprocessing override. Set to None to use dataset defaults.
    MODEL_PREPROCESSING = None
    TASK_TYPE = "classification"

    # Execution controls.
    RUN_PREFLIGHT_VALIDATION = False
    RUN_TRAINING = True
    PREDICTION_REPEATS = 1
    return (
        DATASET_IMPUTER,
        DATASET_SCALER,
        FORCE_REPREPROCESS,
        MODEL_NAME,
        MODEL_PARAMS,
        MODEL_PREPROCESSING,
        PREDICTION_REPEATS,
        RANDOM_STATE,
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
    product,
):
    def expand_params(params):
        keys = list(params)
        value_lists = [
            value if isinstance(value, list) else [value] for value in params.values()
        ]
        return [dict(zip(keys, values)) for values in product(*value_lists)]

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

    model_param_sets = expand_params(MODEL_PARAMS)
    model_params_list = tuple(
        ModelParams(
            name=MODEL_NAME,
            task_type=TASK_TYPE,
            params=params,
            preprocessing=MODEL_PREPROCESSING,
            tuning=None,
        )
        for params in model_param_sets
    )
    return dataset_params, model_param_sets, model_params_list


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
def _(RUN_PREFLIGHT_VALIDATION, Trainer, dataset_params, model_params_list):
    trainer = Trainer(
        params=model_params_list,
        default_imputer=dataset_params.imputer,
        default_scaler=dataset_params.scaler_encoder,
    )
    if RUN_PREFLIGHT_VALIDATION:
        trainer.validate_model_configs()
    model_config_checked = True
    return model_config_checked, trainer


@app.cell
def _(
    PREDICTION_REPEATS,
    RUN_TRAINING,
    TEST_SETS,
    asdict,
    data,
    evaluate_classification_predictions,
    get_model_spec,
    mo,
    model_config_checked,
    model_param_sets,
    model_params_list,
    np,
    pd,
    release_model,
    trainer,
):
    mo.stop(
        not RUN_TRAINING,
        mo.md("Set `RUN_TRAINING = True` in the variables cell to fit the models."),
    )
    _ = model_config_checked
    if PREDICTION_REPEATS < 1:
        raise ValueError("PREDICTION_REPEATS must be at least 1")

    available_test_sets = {
        "mimic": data.test_mimic,
        "tudd": data.test_tudd,
    }

    evaluation_rows = []
    y_train = data.train_data.y.to_numpy()

    for config_index, (model_params, params) in enumerate(
        zip(model_params_list, model_param_sets),
        start=1,
    ):
        if model_params.task_type != "classification":
            raise NotImplementedError(
                "This notebook currently evaluates classification models only"
            )

        param_summary = ", ".join(f"{key}={value}" for key, value in params.items())
        trained_model = None
        try:
            spec = get_model_spec(model_params)
            trained_model, fit_time = trainer._fit_model(
                model_params,
                spec,
                model_params.params,
                data.train_data.X,
                y_train,
            )

            for dataset_name in TEST_SETS:
                test_set = available_test_sets[dataset_name]
                timings = []
                predictions = None
                for _ in range(PREDICTION_REPEATS):
                    predictions, predict_time = trained_model.predict(test_set.X)
                    timings.append(predict_time)

                metrics = evaluate_classification_predictions(
                    predictions,
                    test_set.y.to_numpy(),
                )
                mean_predict_time = float(np.mean(timings))
                evaluation_rows.append(
                    {
                        "config": config_index,
                        "model": model_params.name,
                        "params": param_summary,
                        "fit_time_s": fit_time,
                        "dataset": dataset_name,
                        "rows": len(test_set.y),
                        "predict_repeats": PREDICTION_REPEATS,
                        "predict_time_mean_s": mean_predict_time,
                        "rows_per_second_mean": len(test_set.y) / mean_predict_time,
                        **{f"param_{key}": value for key, value in params.items()},
                        **asdict(metrics),
                    }
                )
        finally:
            release_model(trained_model)

    evaluation_table = pd.DataFrame(evaluation_rows)
    speed_table = evaluation_table.pivot(
        index=["config", "model", "params", "fit_time_s"],
        columns="dataset",
        values="rows_per_second_mean",
    ).reset_index()
    speed_table.columns.name = None
    speed_table = speed_table.rename(
        columns={dataset: f"rows_per_second_{dataset}" for dataset in TEST_SETS}
    )
    return evaluation_table, speed_table


@app.cell
def _(evaluation_table, mo, speed_table):
    mo.vstack(
        [
            mo.md("## Speed Summary"),
            speed_table,
            mo.md("## Detailed Test Timing And Metrics"),
            evaluation_table,
        ]
    )
    return


@app.cell
def _(TEST_SETS, speed_table):
    for _, row in speed_table.iterrows():
        params_s = str(row["params"]).replace("_", r"\_")
        speeds = " / ".join(
            f"{round(row[f'rows_per_second_{dataset}']):,}" for dataset in TEST_SETS
        )
        latex_line = (
            f"{row['model']} & {params_s} & {row['fit_time_s']:.2f} & {speeds} " + r"\\"
        )
        print(latex_line)
    return


if __name__ == "__main__":
    app.run()
