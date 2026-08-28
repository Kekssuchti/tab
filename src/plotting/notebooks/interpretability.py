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
    import matplotlib.pyplot as plt
    import pandas as pd

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.classes.data_registry import dataset_task_for_target
    from src.classes.dataset import Dataset
    from src.classes.trainer import Trainer
    from src.plotting.interpretability import (
        comparison_summary,
        compute_interpretability_comparison,
        global_feature_importance,
        plot_global_ranking_correlations,
        plot_interpretability_comparison,
        ranking_correlation_table,
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
        global_feature_importance,
        mo,
        pd,
        plot_global_ranking_correlations,
        plt,
        plot_interpretability_comparison,
        project_root,
        ranking_correlation_table,
    )


@app.cell
def _():
    RUN_INTERPRETABILITY = True

    # Dataset settings. This seed fixes the train/test split across every run.
    TARGET = "mortality"
    IMPUTATION_METHOD = "none"
    TRAIN_ON = (("tudd", 1.0),)
    DATASET_RANDOM_STATE = 1337
    TRAIN_SIZE = 0.8
    FORCE_REPREPROCESS = False
    DATASET_IMPUTER = {"imputation_method": IMPUTATION_METHOD, "flag_missing": False}
    DATASET_SCALER = {"type": "none"}

    # Fixed model hyperparameters. The model seed is injected separately for each run.
    # EBM interactions remain disabled because only native main effects belong in these plots.
    MODEL_PARAMS = {
        "ebm": {
            "interactions": 0,
            "learning_rate": 0.005,
            "max_bins": 256,
            "outer_bags": 8,
            "inner_bags": 0,
            "min_samples_leaf": 10,
        },
        "xgboost": {
            "colsample_bylevel": 0.8335403471179039,
            "colsample_bytree": 0.516851692707462,
            "eta": 0.003158291362058371,
            "max_depth": 5,
            "min_child_weight": 0.0035577147670846562,
            "n_estimators": 2120,
            "reg_alpha": 6.298800770211916e-05,
            "reg_lambda": 0,
            "subsample": 0.5591716150759687,
        },
        "tabpfn-3": {
            "n_estimators": 32,
            "fit_mode": "fit_with_cache",
        },
    }

    # Interpretability settings. The explanation seed remains fixed so every run
    # explains exactly the same held-out rows. The other seeds vary by run.
    TEST_SOURCE = "tudd"
    BACKGROUND_ROWS = 256
    EXPLANATION_ROWS = 1000
    EXPLANATION_SAMPLE_SEED = 5337
    CLASS_INDEX = 1
    TABPFN_SHAP_BUDGET = 2048
    RUN_SEEDS = (
        {"model": 1337, "background": 2337, "explainer": 3337},
        {"model": 1338, "background": 2338, "explainer": 3338},
        {"model": 1339, "background": 2339, "explainer": 3339},
    )

    # Each run has its own cache at <PLOT_OUTPUT_DIR>/<run>/tabpfn_effects.npz.
    # The cache key also validates the data, preprocessing, parameters, and seeds.
    TABPFN_CACHE_FILENAME = "tabpfn_effects.npz"
    RECOMPUTE_TABPFN = False

    # Save every transformed feature for every run, but display only one run in
    # the notebook to keep the interactive output manageable.
    FEATURES_TO_PLOT = None
    DISPLAY_RUN = 1
    SAVE_PLOTS = True
    PLOT_OUTPUT_DIR = f"plots/interpretability/{TARGET.lower()}/{IMPUTATION_METHOD}"
    return (
        BACKGROUND_ROWS,
        CLASS_INDEX,
        DATASET_IMPUTER,
        DATASET_RANDOM_STATE,
        DATASET_SCALER,
        DISPLAY_RUN,
        EXPLANATION_ROWS,
        EXPLANATION_SAMPLE_SEED,
        FEATURES_TO_PLOT,
        FORCE_REPREPROCESS,
        MODEL_PARAMS,
        PLOT_OUTPUT_DIR,
        RECOMPUTE_TABPFN,
        RUN_INTERPRETABILITY,
        RUN_SEEDS,
        SAVE_PLOTS,
        TABPFN_CACHE_FILENAME,
        TABPFN_SHAP_BUDGET,
        TARGET,
        TEST_SOURCE,
        TRAIN_ON,
        TRAIN_SIZE,
    )


@app.cell
def _(
    DATASET_IMPUTER,
    DATASET_RANDOM_STATE,
    DATASET_SCALER,
    DataSplitConfig,
    DatasetConfig,
    FORCE_REPREPROCESS,
    ImputerConfig,
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
        random_state=DATASET_RANDOM_STATE,
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
    EXPLANATION_SAMPLE_SEED,
    MODEL_PARAMS,
    PLOT_OUTPUT_DIR,
    RECOMPUTE_TABPFN,
    RUN_SEEDS,
    TABPFN_CACHE_FILENAME,
    TABPFN_SHAP_BUDGET,
    TEST_SOURCE,
    compute_interpretability_comparison,
    data,
    project_root,
    trainer,
):
    _comparison_runs = []
    for _run_number, _run_seeds in enumerate(RUN_SEEDS, start=1):
        _run_output_dir = project_root / PLOT_OUTPUT_DIR / str(_run_number)
        _comparison_runs.append(
            compute_interpretability_comparison(
                trainer=trainer,
                data=data,
                model_params=MODEL_PARAMS,
                test_source=TEST_SOURCE,
                background_rows=BACKGROUND_ROWS,
                explanation_rows=EXPLANATION_ROWS,
                class_index=CLASS_INDEX,
                tabpfn_shap_budget=TABPFN_SHAP_BUDGET,
                model_random_state=_run_seeds["model"],
                background_random_state=_run_seeds["background"],
                explanation_random_state=EXPLANATION_SAMPLE_SEED,
                explainer_random_state=_run_seeds["explainer"],
                tabpfn_cache_path=_run_output_dir / TABPFN_CACHE_FILENAME,
                recompute_tabpfn=RECOMPUTE_TABPFN,
            )
        )
    comparisons = tuple(_comparison_runs)
    return (comparisons,)


@app.cell
def _(
    PLOT_OUTPUT_DIR,
    SAVE_PLOTS,
    comparisons,
    global_feature_importance,
    plot_global_ranking_correlations,
    project_root,
    ranking_correlation_table,
):
    ranking_output_dir = project_root / PLOT_OUTPUT_DIR if SAVE_PLOTS else None
    correlation_output_path = (
        ranking_output_dir / "global_ranking_correlations.svg" if ranking_output_dir is not None else None
    )
    global_rankings = global_feature_importance(comparisons)
    rank_correlations = ranking_correlation_table(comparisons)
    correlation_figure, correlation_matrix = plot_global_ranking_correlations(
        comparisons,
        output_path=correlation_output_path,
    )

    run_manifest = global_rankings[
        ["run", "model_seed", "background_seed", "explanation_seed", "explainer_seed"]
    ].drop_duplicates(ignore_index=True)
    stability_summary = rank_correlations[
        rank_correlations["scope"] == "run stability"
    ].groupby("comparison", as_index=False).agg(
        median_rho=("spearman_rho", "median"),
        minimum_rho=("spearman_rho", "min"),
        maximum_rho=("spearman_rho", "max"),
    )
    agreement_summary = rank_correlations[
        rank_correlations["scope"] == "model agreement"
    ].groupby("comparison", as_index=False).agg(
        median_rho=("spearman_rho", "median"),
        minimum_rho=("spearman_rho", "min"),
        maximum_rho=("spearman_rho", "max"),
    )

    if ranking_output_dir is not None:
        ranking_output_dir.mkdir(parents=True, exist_ok=True)
        run_manifest.to_csv(ranking_output_dir / "run_manifest.csv", index=False)
        global_rankings.to_csv(ranking_output_dir / "global_feature_rankings.csv", index=False)
        rank_correlations.to_csv(ranking_output_dir / "global_ranking_correlation_details.csv", index=False)
        stability_summary.to_csv(ranking_output_dir / "global_ranking_stability_summary.csv", index=False)
        agreement_summary.to_csv(ranking_output_dir / "global_ranking_agreement_summary.csv", index=False)
        correlation_matrix.to_csv(ranking_output_dir / "global_ranking_correlation_matrix.csv")

    top_rankings = global_rankings[global_rankings["rank"] <= 10].pivot_table(
        index="feature_label",
        columns=["model_label", "run"],
        values="rank",
    )
    return (
        agreement_summary,
        correlation_figure,
        correlation_matrix,
        global_rankings,
        rank_correlations,
        stability_summary,
        top_rankings,
    )


@app.cell
def _(
    DISPLAY_RUN,
    FEATURES_TO_PLOT,
    PLOT_OUTPUT_DIR,
    SAVE_PLOTS,
    comparisons,
    plot_interpretability_comparison,
    plt,
    project_root,
):
    if not 1 <= DISPLAY_RUN <= len(comparisons):
        raise ValueError(f"DISPLAY_RUN must be between 1 and {len(comparisons)}")

    _display_feature_figures = None
    _run_output_dirs = []
    for _run_number, _comparison in enumerate(comparisons, start=1):
        _output_dir = project_root / PLOT_OUTPUT_DIR / str(_run_number) if SAVE_PLOTS else None
        _feature_figures = plot_interpretability_comparison(
            _comparison,
            features=FEATURES_TO_PLOT,
            output_dir=_output_dir,
            figsize=(10, 6),
        )
        _run_output_dirs.append((_run_number, _output_dir))
        if _run_number == DISPLAY_RUN:
            _display_feature_figures = _feature_figures
        else:
            for _, _figure in _feature_figures:
                plt.close(_figure)

    assert _display_feature_figures is not None
    display_feature_figures = _display_feature_figures
    run_output_dirs = tuple(_run_output_dirs)
    return display_feature_figures, run_output_dirs


@app.cell
def _(
    DISPLAY_RUN,
    agreement_summary,
    comparison_summary,
    comparisons,
    correlation_figure,
    display_feature_figures,
    mo,
    pd,
    run_output_dirs,
    stability_summary,
    top_rankings,
):
    run_summary = pd.concat(
        [comparison_summary(comparison).assign(run=run) for run, comparison in enumerate(comparisons, start=1)],
        ignore_index=True,
    )
    saved_locations = [
        f"Run {run}: {output_dir}" if output_dir is not None else f"Run {run}: plots not saved"
        for run, output_dir in run_output_dirs
    ]

    mo.vstack(
        [
            mo.md("# EBM, XGBoost, and TabPFNv3 feature effects"),
            mo.md(
                "Every run uses the same train/test split and the same explained test rows. "
                "Model, background-sample, and explainer seeds vary between runs. EBM is shown "
                "as its native main-effect line; XGBoost and TabPFNv3 are single-color SHAP point clouds. "
                "The methods explain model-specific margin/logit outputs with different centering and scales; "
                "compare directions, shapes, and rankings rather than raw magnitudes."
            ),
            run_summary,
            mo.md("## Global mean-absolute feature rankings"),
            mo.md(
                "Ranks are calculated within each model and run from the mean absolute row-level "
                "feature contribution. Spearman correlations compare rankings without treating the "
                "different explanation scales as directly comparable."
            ),
            top_rankings,
            mo.md("### Within-model global-ranking stability across runs"),
            stability_summary,
            mo.md("### Between-model agreement within runs"),
            agreement_summary,
            correlation_figure,
            mo.md("## Saved outputs\n" + "\n".join(f"- {location}" for location in saved_locations)),
            mo.md(f"# Feature plots displayed for run {DISPLAY_RUN}"),
            *[
                mo.vstack([mo.md(f"## {feature_name}"), figure])
                for feature_name, figure in display_feature_figures
            ],
        ]
    )


if __name__ == "__main__":
    app.run()
