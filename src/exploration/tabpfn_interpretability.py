import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import shap
    from tabpfn_extensions.interpretability import (
        shapiq as tabpfn_shapiq,
        shapiq_to_shap_explanation,
    )

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.classes.data_registry import dataset_task_for_target
    from src.classes.dataset import Dataset
    from src.classes.trainer import Trainer
    from src.schemas.dataset_schemas import DataSplitConfig, DatasetConfig
    from src.schemas.preprocessing_schemas import ImputerConfig, ScalerEncoderConfig
    from src.schemas.training_schemas import ModelConfig
    from src.utils.model_lifecycle import release_model
    from src.utils.model_registry import get_model_spec

    return (
        DataSplitConfig,
        Dataset,
        DatasetConfig,
        ImputerConfig,
        ModelConfig,
        Path,
        ScalerEncoderConfig,
        Trainer,
        dataset_task_for_target,
        get_model_spec,
        mo,
        np,
        pd,
        plt,
        project_root,
        release_model,
        shap,
        shapiq_to_shap_explanation,
        tabpfn_shapiq,
    )


@app.cell
def _():
    RUN_INTERPRETABILITY = True

    # Dataset settings. Training sources are origins; the target selects the file kind.
    TARGET = "mortality"
    imputation_method = "none"
    TRAIN_ON = (
        ("tudd", 1.0),
    )
    RANDOM_STATE = 1337
    TRAIN_SIZE = 0.8
    FORCE_REPREPROCESS = False
    DATASET_IMPUTER = {"imputation_method": imputation_method, "flag_missing": False}
    DATASET_SCALER = {"type": "none"}

    MODEL_NAME = "tabpfn-3"
    MODEL_PARAMS = {
        "n_estimators": 4,
        "fit_mode": "fit_with_cache",
        "random_state": RANDOM_STATE,
    }

    MODEL_PREPROCESSING = None

    # Interpretability settings.
    TEST_SOURCE = "tudd"
    BACKGROUND_ROWS = 256
    NUMBER_EXPLANATION_ROWS = 1000
    CLASS_INDEX = 1
    SHAP_BUDGET = 2048 
    return (
        BACKGROUND_ROWS,
        CLASS_INDEX,
        DATASET_IMPUTER,
        DATASET_SCALER,
        FORCE_REPREPROCESS,
        MODEL_NAME,
        MODEL_PARAMS,
        MODEL_PREPROCESSING,
        NUMBER_EXPLANATION_ROWS,
        RANDOM_STATE,
        RUN_INTERPRETABILITY,
        SHAP_BUDGET,
        TARGET,
        TEST_SOURCE,
        TRAIN_ON,
        TRAIN_SIZE,
        imputation_method,
    )


@app.cell
def _(
    DATASET_IMPUTER,
    DATASET_SCALER,
    DataSplitConfig,
    DatasetConfig,
    FORCE_REPREPROCESS,
    ImputerConfig,
    MODEL_NAME,
    MODEL_PREPROCESSING,
    ModelConfig,
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
        mo.md("Set `RUN_INTERPRETABILITY = True` in the settings cell when you are ready."),
    )

    task_type = dataset_task_for_target(TARGET).task_type

    dataset_params = DatasetConfig(
        target=TARGET,
        random_state=RANDOM_STATE,
        train_size=TRAIN_SIZE,
        train_on=tuple(DataSplitConfig(dataset=name, fraction=amount) for name, amount in TRAIN_ON),
        force_repreprocess=FORCE_REPREPROCESS,
        imputer=ImputerConfig(**DATASET_IMPUTER),
        scaler_encoder=ScalerEncoderConfig(**DATASET_SCALER),
    )
    model_config = ModelConfig(
        name=MODEL_NAME,
        preprocessing=MODEL_PREPROCESSING,
    )
    return dataset_params, model_config, task_type


@app.cell
def _(Dataset, dataset_params):
    dataset = Dataset(dataset_params)
    data = dataset.get_dataset()
    return (data,)


@app.cell
def _(Trainer, dataset_params, task_type):
    trainer = Trainer(
        task_type=task_type,
        default_imputer=dataset_params.imputer,
        default_scaler=dataset_params.scaler_encoder,
        log_transform_target=dataset_params.log_transform_target,
    )
    return (trainer,)


@app.cell
def _(
    BACKGROUND_ROWS,
    CLASS_INDEX,
    MODEL_PARAMS,
    NUMBER_EXPLANATION_ROWS,
    RANDOM_STATE,
    SHAP_BUDGET,
    TEST_SOURCE,
    data,
    get_model_spec,
    model_config,
    np,
    pd,
    release_model,
    shapiq_to_shap_explanation,
    tabpfn_shapiq,
    task_type,
    trainer,
):
    test_sets = {"mimic": data.test_mimic, "tudd": data.test_tudd}
    test_data = test_sets[TEST_SOURCE]
    background_raw = data.train_data.X.sample(n=BACKGROUND_ROWS, random_state=RANDOM_STATE)
    explain_raw = test_data.X.sample(n=NUMBER_EXPLANATION_ROWS, random_state=RANDOM_STATE + 1)

    trained_model = None
    try:
        spec = get_model_spec(model_config, task_type)
        trained_model, fit_time = trainer._fit_model(
            model_config,
            spec,
            MODEL_PARAMS,
            data.train_data.X,
            data.train_data.y.to_numpy(),
        )

        # Explain the transformed feature space consumed by the underlying TabPFN estimator,
        # not the raw columns that enter the project's preprocessing adapter.
        preprocess_pipeline = trained_model.preprocess_pipeline
        underlying_model = trained_model.adapter.model
        background = np.asarray(preprocess_pipeline.transform(background_raw))
        X_explain = np.asarray(preprocess_pipeline.transform(explain_raw))
        feature_names = [str(name) for name in preprocess_pipeline.get_feature_names_out()]

        explainer = tabpfn_shapiq.get_tabpfn_imputation_explainer(
            model=underlying_model,
            data=background,
            index="SV",
            max_order=1,
            imputer="baseline",
            class_index=CLASS_INDEX,
            random_state=RANDOM_STATE,
        )
        shap_explanation = shapiq_to_shap_explanation(
            explainer,
            X_explain,
            budget=SHAP_BUDGET,
            feature_names=feature_names,
        )
    finally:
        release_model(trained_model)

    shap_values = np.asarray(shap_explanation.values)
    importance_table = (
        pd.DataFrame(
            {
                "feature": list(shap_explanation.feature_names),
                "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False, ignore_index=True)
        .assign(rank=lambda frame: frame.index + 1)
        [["rank", "feature", "mean_abs_shap"]]
    )
    return fit_time, importance_table, shap_explanation, shap_values


@app.cell
def _(
    BACKGROUND_ROWS,
    CLASS_INDEX,
    MODEL_PARAMS,
    NUMBER_EXPLANATION_ROWS,
    Path,
    SHAP_BUDGET,
    TARGET,
    TEST_SOURCE,
    fit_time,
    importance_table,
    imputation_method,
    model_config,
    pd,
    plt,
    project_root,
    shap,
    shap_explanation,
    shap_values,
):
    MAX_DISPLAYED_FEATURES = 15

    # Optional plot output. Inline figures are always produced.
    SAVE_PLOTS = True
    PLOT_OUTPUT_DIR = f"plots/shap/{TARGET.lower()}/{imputation_method}"

    plot_height = min(8.0, 4.0 + 0.25 * MAX_DISPLAYED_FEATURES)

    bar_figure, bar_axis = plt.subplots(figsize=(10.5, plot_height), constrained_layout=True)
    shap.plots.bar(
        shap_explanation,
        max_display=MAX_DISPLAYED_FEATURES,
        ax=bar_axis,
        show=False,
    )

    beeswarm_figure, beeswarm_axis = plt.subplots(figsize=(10.5, plot_height), constrained_layout=True)
    shap.plots.beeswarm(
        shap_explanation,
        max_display=MAX_DISPLAYED_FEATURES,
        ax=beeswarm_axis,
        plot_size=None,
        show=False,
    )

    scatter_figures = []
    for feature_index, feature_name in enumerate(shap_explanation.feature_names):
        scatter_figure, scatter_axis = plt.subplots(figsize=(10.5, 6.0), constrained_layout=True)
        shap.plots.scatter(
            shap_explanation[:, feature_index],
            color=shap_explanation,
            title=f"SHAP dependence for {feature_name}",
            ax=scatter_axis,
            show=False,
        )
        scatter_figures.append((feature_name, scatter_figure))
    scatter_figures = tuple(scatter_figures)

    saved_to = None
    if SAVE_PLOTS:
        output_dir = Path(PLOT_OUTPUT_DIR)
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        bar_figure.savefig(output_dir / "global_bar.svg", bbox_inches="tight", pad_inches=0.2)
        beeswarm_figure.savefig(output_dir / "beeswarm.svg", bbox_inches="tight", pad_inches=0.2)
        for feature_index, (feature_name, scatter_figure) in enumerate(scatter_figures):
            feature_slug = "".join(
                character if character.isalnum() or character in "-_." else "_"
                for character in feature_name
            )
            scatter_figure.savefig(
                output_dir / f"scatter_{feature_index:02d}_{feature_slug}.svg",
                bbox_inches="tight",
                pad_inches=0.2,
            )
        saved_to = str(output_dir.resolve())

    run_summary = pd.DataFrame(
        [
            {
                "model": model_config.name,
                "fixed_params": repr(MODEL_PARAMS),
                "fit_time_s": fit_time,
                "test_source": TEST_SOURCE,
                "background_rows": BACKGROUND_ROWS,
                "explained_rows": NUMBER_EXPLANATION_ROWS,
                "transformed_features": shap_values.shape[1],
                "class_index": CLASS_INDEX,
                "budget_per_row": SHAP_BUDGET,
                "plots_saved_to": saved_to,
            }
        ]
    )
    plot_figures = (bar_figure, beeswarm_figure)

    if SAVE_PLOTS:
        run_summary.to_csv(output_dir / "run_meta.csv")
        importance_table.to_csv(output_dir / "importance_table.csv")    
    return plot_figures, run_summary, scatter_figures


@app.cell
def _(importance_table, mo, plot_figures, run_summary, scatter_figures):
    mo.vstack(
        [
            mo.md("## Run summary"),
            run_summary,
            mo.md("## Mean absolute SHAP feature importance"),
            importance_table,
            mo.md("## Global bar plot"),
            plot_figures[0],
            mo.md("## Beeswarm plot"),
            plot_figures[1],
            mo.md("## SHAP dependence scatter plots"),
            *[
                mo.vstack([mo.md(f"### `{feature_name}`"), scatter_figure])
                for feature_name, scatter_figure in scatter_figures
            ],
        ]
    )
    return


if __name__ == "__main__":
    app.run()
