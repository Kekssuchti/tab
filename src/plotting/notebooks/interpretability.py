import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

    import marimo as mo

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.classes.data_registry import dataset_task_for_target
    from src.classes.dataset import Dataset
    from src.classes.trainer import Trainer
    from src.plotting.interpretability import (
        comparison_summary,
        compute_interpretability_comparison,
        plot_interpretability_comparison,
    )
    from src.schemas.dataset_schemas import DatasetConfig, DataSplitConfig
    from src.schemas.preprocessing_schemas import ImputerConfig, ScalerEncoderConfig

    return (
        DataSplitConfig,
        Dataset,
        DatasetConfig,
        ImputerConfig,
        ScalerEncoderConfig,
        Trainer,
        comparison_summary,
        compute_interpretability_comparison,
        dataset_task_for_target,
        mo,
        plot_interpretability_comparison,
        project_root,
    )


@app.cell
def _():
    RUN_INTERPRETABILITY = True

    # Dataset settings. Training sources are origins; the target selects the file kind.
    TARGET = "mortality"
    IMPUTATION_METHOD = "none"
    TRAIN_ON = (("tudd", 0.3),)
    RANDOM_STATE = 1337
    TRAIN_SIZE = 0.8
    FORCE_REPREPROCESS = False
    DATASET_IMPUTER = {"imputation_method": IMPUTATION_METHOD, "flag_missing": False}
    DATASET_SCALER = {"type": "none"}

    # Replace the EBM and XGBoost entries with the exact fixed hyperparameters.
    # Keep EBM interactions at zero: only native main effects belong in these plots.
    MODEL_PARAMS = {
        "ebm": {
            "interactions": 0,
            "random_state": RANDOM_STATE,
            'learning_rate': 0.005, 
            'max_bins': 256, 
            'outer_bags': 8, 
            'inner_bags': 0, 
            'min_samples_leaf': 10
        },
        "xgboost": {
            "random_state": RANDOM_STATE,
            "colsample_bylevel": 0.8335403471179039, 
            "colsample_bytree": 0.516851692707462, 
            "eta": 0.003158291362058371, 
            "max_depth": 5, 
            "min_child_weight": 0.0035577147670846562, 
            "n_estimators": 2120, 
            "reg_alpha": 6.298800770211916e-05, 
            "reg_lambda": 0, 
            "subsample": 0.5591716150759687
        },
        "tabpfn-3": {
            "n_estimators": 4,
            "fit_mode": "fit_with_cache",
            "random_state": RANDOM_STATE,
        },
    }

    # Interpretability settings.
    TEST_SOURCE = "tudd"
    BACKGROUND_ROWS = 256
    EXPLANATION_ROWS = 1000
    CLASS_INDEX = 1
    TABPFN_SHAP_BUDGET = 2048

    # Plot settings. Use None to render every transformed feature.
    FEATURES_TO_PLOT = None
    SAVE_PLOTS = True
    PLOT_OUTPUT_DIR = f"plots/interpretability/{TARGET.lower()}/{IMPUTATION_METHOD}"
    return (
        BACKGROUND_ROWS,
        CLASS_INDEX,
        DATASET_IMPUTER,
        DATASET_SCALER,
        EXPLANATION_ROWS,
        FEATURES_TO_PLOT,
        FORCE_REPREPROCESS,
        MODEL_PARAMS,
        PLOT_OUTPUT_DIR,
        RANDOM_STATE,
        RUN_INTERPRETABILITY,
        SAVE_PLOTS,
        TABPFN_SHAP_BUDGET,
        TARGET,
        TEST_SOURCE,
        TRAIN_ON,
        TRAIN_SIZE,
    )


@app.cell
def _(
    DATASET_IMPUTER,
    DATASET_SCALER,
    DataSplitConfig,
    DatasetConfig,
    FORCE_REPREPROCESS,
    ImputerConfig,
    RANDOM_STATE,
    RUN_INTERPRETABILITY,
    ScalerEncoderConfig,
    TARGET,
    TRAIN_ON,
    TRAIN_SIZE,
    dataset_task_for_target,
    mo,
):
    mo.stop(
        not RUN_INTERPRETABILITY,
        mo.md("Set RUN_INTERPRETABILITY = True when the parameters are ready."),
    )

    task_type = dataset_task_for_target(TARGET).task_type
    dataset_config = DatasetConfig(
        target=TARGET,
        random_state=RANDOM_STATE,
        train_size=TRAIN_SIZE,
        train_on=tuple(DataSplitConfig(dataset=name, fraction=amount) for name, amount in TRAIN_ON),
        force_repreprocess=FORCE_REPREPROCESS,
        imputer=ImputerConfig(**DATASET_IMPUTER),
        scaler_encoder=ScalerEncoderConfig(**DATASET_SCALER),
    )
    return dataset_config, task_type


@app.cell
def _(Dataset, Trainer, dataset_config, task_type):
    dataset = Dataset(dataset_config)
    data = dataset.get_dataset()
    trainer = Trainer(
        task_type=task_type,
        default_imputer=dataset_config.imputer,
        default_scaler=dataset_config.scaler_encoder,
        log_transform_target=dataset_config.log_transform_target,
    )
    return data, trainer


@app.cell
def _(
    BACKGROUND_ROWS,
    CLASS_INDEX,
    EXPLANATION_ROWS,
    MODEL_PARAMS,
    RANDOM_STATE,
    TABPFN_SHAP_BUDGET,
    TEST_SOURCE,
    compute_interpretability_comparison,
    data,
    trainer,
):
    comparison = compute_interpretability_comparison(
        trainer=trainer,
        data=data,
        model_params=MODEL_PARAMS,
        test_source=TEST_SOURCE,
        background_rows=BACKGROUND_ROWS,
        explanation_rows=EXPLANATION_ROWS,
        class_index=CLASS_INDEX,
        tabpfn_shap_budget=TABPFN_SHAP_BUDGET,
        random_state=RANDOM_STATE,
    )
    return (comparison,)


@app.cell
def _(
    FEATURES_TO_PLOT,
    PLOT_OUTPUT_DIR,
    SAVE_PLOTS,
    comparison,
    plot_interpretability_comparison,
    project_root,
):
    output_dir = project_root / PLOT_OUTPUT_DIR if SAVE_PLOTS else None
    feature_figures = plot_interpretability_comparison(
        comparison,
        features=FEATURES_TO_PLOT,
        output_dir=output_dir,
    )
    return feature_figures, output_dir


@app.cell
def _(comparison, comparison_summary, feature_figures, mo, output_dir):
    saved_message = f"Plots saved to {output_dir}." if output_dir is not None else "Plots were not saved."
    mo.vstack(
        [
            mo.md("# EBM, XGBoost, and TabPFNv3 feature effects"),
            mo.md(
                "EBM is shown as its native main-effect line. XGBoost and TabPFNv3 "
                "are single-color SHAP point clouds; no secondary interaction feature is encoded."
            ),
            comparison_summary(comparison),
            mo.md(saved_message),
            *[
                mo.vstack([mo.md(f"## {feature_name}"), figure])
                for feature_name, figure in feature_figures
            ],
        ]
    )
    return


if __name__ == "__main__":
    app.run()
