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
    from src.classes.data_registry import dataset_task_for_target
    from src.classes.trainer import Trainer
    from src.schemas.dataset_schemas import DatasetConfig, DataSplitConfig
    from src.schemas.preprocessing_schemas import ImputerConfig, ScalerEncoderConfig
    from src.schemas.training_schemas import ModelConfig, TuningConfig
    from src.utils.evaluation_utils import evaluate_classification_predictions
    from src.utils.model_lifecycle import release_model
    from src.utils.model_registry import get_model_spec

    return (
        DataSplitConfig,
        Dataset,
        DatasetConfig,
        dataset_task_for_target,
        ImputerConfig,
        ModelConfig,
        ScalerEncoderConfig,
        Trainer,
        TuningConfig,
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

    # Training sources are always origins; the target selects normal/readmission files.
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

    MODEL_NAME = "tabpfn-2.6"
    MODEL_PARAMS = {
        "n_estimators": [2],
        "fit_mode": "fit_preprocessors",
    }
    # Optional model-specific preprocessing override. Set to None to use dataset defaults.
    MODEL_PREPROCESSING = None
    # Execution controls.
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
        RUN_TRAINING,
        TARGET,
        TEST_SETS,
        TRAIN_ON,
        TRAIN_SIZE,
    )


@app.cell
def _(
    DATASET_IMPUTER,
    DATASET_SCALER,
    DataSplitConfig,
    DatasetConfig,
    dataset_task_for_target,
    FORCE_REPREPROCESS,
    ImputerConfig,
    MODEL_NAME,
    MODEL_PARAMS,
    MODEL_PREPROCESSING,
    ModelConfig,
    RANDOM_STATE,
    ScalerEncoderConfig,
    TARGET,
    TRAIN_ON,
    TRAIN_SIZE,
    TuningConfig,
    product,
):
    def expand_params(params):
        keys = list(params)
        value_lists = [value if isinstance(value, list) else [value] for value in params.values()]
        return [dict(zip(keys, values)) for values in product(*value_lists)]

    dataset_params = DatasetConfig(
        target=TARGET,
        random_state=RANDOM_STATE,
        train_size=TRAIN_SIZE,
        train_on=tuple(DataSplitConfig(dataset=dataset_name, fraction=fraction) for dataset_name, fraction in TRAIN_ON),
        force_repreprocess=FORCE_REPREPROCESS,
        imputer=ImputerConfig(**DATASET_IMPUTER),
        scaler_encoder=ScalerEncoderConfig(**DATASET_SCALER),
    )

    model_param_sets = expand_params(MODEL_PARAMS)
    model_config = ModelConfig(
        name=MODEL_NAME,
        preprocessing=MODEL_PREPROCESSING,
        tuning=TuningConfig(method="grid"),
    )
    task_type = dataset_task_for_target(TARGET).task_type
    return dataset_params, model_config, model_param_sets, task_type


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
                "target_summary": asdict(dataset_summary.train.target_summary),
            },
            {
                "part": "test_mimic",
                "rows": dataset_summary.test_mimic.row_count,
                "features": data.test_mimic.X.shape[1],
                "target_summary": asdict(dataset_summary.test_mimic.target_summary),
            },
            {
                "part": "test_tudd",
                "rows": dataset_summary.test_tudd.row_count,
                "features": data.test_tudd.X.shape[1],
                "target_summary": asdict(dataset_summary.test_tudd.target_summary),
            },
        ]
    )

    mo.vstack(
        [
            mo.md("## Dataset"),
            dataset_table,
            pd.DataFrame(asdict(file_summary) for file_summary in dataset_summary.data_files),
        ]
    )
    return (data,)


@app.cell
def _(Trainer, dataset_params, model_config, task_type):
    trainer = Trainer(
        task_type=task_type,
        default_imputer=dataset_params.imputer,
        default_scaler=dataset_params.scaler_encoder,
    )
    return (trainer,)


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
    model_config,
    model_param_sets,
    np,
    pd,
    release_model,
    task_type,
    trainer,
):
    mo.stop(
        not RUN_TRAINING,
        mo.md("Set `RUN_TRAINING = True` in the variables cell to fit the models."),
    )
    if PREDICTION_REPEATS < 1:
        raise ValueError("PREDICTION_REPEATS must be at least 1")

    available_test_sets = {
        "mimic": data.test_mimic,
        "tudd": data.test_tudd,
    }

    evaluation_rows = []
    y_train = data.train_data.y.to_numpy()

    for config_index, params in enumerate(model_param_sets, start=1):
        if task_type != "classification":
            raise NotImplementedError("This notebook currently evaluates classification models only")

        param_summary = ", ".join(f"{key}={value}" for key, value in params.items())
        trained_model = None
        try:
            spec = get_model_spec(model_config, task_type)
            trained_model, fit_time = trainer._fit_model(
                model_config,
                spec,
                params,
                data.train_data.X,
                y_train,
            )

            for dataset_name in TEST_SETS:
                test_set = available_test_sets[dataset_name]
                timings = []
                prediction = None
                for _ in range(PREDICTION_REPEATS):
                    prediction = trained_model.predict(test_set.X)
                    timings.append(prediction.seconds)

                metrics = evaluate_classification_predictions(
                    prediction.values,
                    test_set.y.to_numpy(),
                )
                mean_predict_time = float(np.mean(timings))
                evaluation_rows.append(
                    {
                        "config": config_index,
                        "model": model_config.name,
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
    speed_table = speed_table.rename(columns={dataset: f"rows_per_second_{dataset}" for dataset in TEST_SETS})
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
        speeds = " / ".join(f"{round(row[f'rows_per_second_{dataset}']):,}" for dataset in TEST_SETS)
        latex_line = f"{row['model']} & {params_s} & {row['fit_time_s']:.2f} & {speeds} " + r"\\"
        print(latex_line)
    return


if __name__ == "__main__":
    app.run()
