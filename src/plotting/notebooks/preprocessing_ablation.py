import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import matplotlib.pyplot as plt
    import pandas as pd

    from src.mlflow.evaluation_data import (
        list_pipeline_runs,
        load_evaluation_data,
    )
    from src.plotting.ablations import plot_model_setting_performance, plot_model_setting_performance_vs_runtime

    return (
        list_pipeline_runs,
        load_evaluation_data,
        pd,
        plot_model_setting_performance,
        plot_model_setting_performance_vs_runtime,
        plt,
    )


@app.cell
def _(list_pipeline_runs):
    # has KNN
    exploration_exp_name_knn = "tudd_baseline_mean_standardize_mortality"

    # all other settings
    exploration_exp_name = "tudd_ablation_imputer"

    setting_run_ids = {
        "KNN (5)": (
            "00eb43db187c44a7b91b21fa5ec05191",
            "28f1e7530b3b4140a629889d2ffdae8e",
        ),
        "Mean": ("35174cd2e5414f04a8e0ac59e7f553ad"),
        "Median": ("2ac501a577254560af7ca01d8706206f"),
        "Most Frequent": ("a14ec50990da4cf087b9c66e7958e672"),
        "Native": ("ade91770433244aaa2877ecf7d0d331e"),
    }
    selected_run_ids = {
        run_id
        for run_ids in setting_run_ids.values()
        for run_id in ((run_ids,) if isinstance(run_ids, str) else run_ids)
    }
    runs_exploration = list_pipeline_runs(exploration_exp_name)
    runs_exploration = runs_exploration.loc[runs_exploration["mlflow_run_id"].isin(selected_run_ids)].copy()

    runs_exploration_knn = list_pipeline_runs(exploration_exp_name_knn)
    runs_exploration_knn = runs_exploration_knn.loc[runs_exploration_knn["mlflow_run_id"].isin(selected_run_ids)].copy()

    runs_exploration
    return (
        exploration_exp_name,
        exploration_exp_name_knn,
        runs_exploration,
        runs_exploration_knn,
        setting_run_ids,
    )


@app.cell
def _(
    exploration_exp_name,
    exploration_exp_name_knn,
    load_evaluation_data,
    pd,
    runs_exploration,
    runs_exploration_knn,
):
    data_imputation = load_evaluation_data(
        experiment_names=exploration_exp_name,
        pipeline_runs=runs_exploration["mlflow_run_id"],
    )

    data_imputation_knn = load_evaluation_data(
        experiment_names=exploration_exp_name_knn,
        pipeline_runs=runs_exploration_knn["mlflow_run_id"],
    )

    data_imputation_all = pd.concat([data_imputation, data_imputation_knn])
    return (data_imputation_all,)


@app.cell
def _(setting_run_ids):
    model_setups = {
        "all": [
            "ebm",
            "orion-msp",
            "limix-16m",
            "tabswift",
            "tabpfn-3",
            "xgboost",
            "logistic-regression",
            "tabicl-2",
            "tabfm",
        ],
        "main": ["tabpfn-3", "xgboost", "logistic-regression", "tabicl-2", "tabfm"],
    }
    common_kwargs = {
        "show_ci": False,
        "setting_run_ids": setting_run_ids,
        "excluded_models_by_setting": {
            "Native": ["tabswift", "orion-msp", "tabfm", "logistic-regression"],
        },
        "dataset": "tudd",
    }
    save_figs = True
    return common_kwargs, model_setups, save_figs


@app.cell
def _(
    common_kwargs,
    data_imputation_all,
    model_setups,
    plot_model_setting_performance,
    plt,
    save_figs,
):
    for model_group, models in model_setups.items():
        fig = plot_model_setting_performance(
            data_imputation_all,
            include_models=models,
            legend_title="Missing Value Imputation",
            title="",
            y_limits="auto",
            **common_kwargs,
        )
        if save_figs:
            fig.savefig(f"plots/preprocessing_comparision/imputer/performance_bar_{model_group}.svg")
        plt.show()


@app.cell
def _(
    common_kwargs,
    data_imputation_all,
    model_setups,
    plot_model_setting_performance_vs_runtime,
    plt,
    save_figs,
):
    for model_group2, models2 in model_setups.items():
        fig2 = plot_model_setting_performance_vs_runtime(
            data_imputation_all,
            include_models=models2,
            legend_title="Preprocessing",
            title="",
            log_x=True,
            **common_kwargs,
        )
        if save_figs:
            fig2.savefig(f"plots/preprocessing_comparision/imputer/time_performance_{model_group2}.svg")
        plt.show()


@app.cell
def _(list_pipeline_runs):
    scaler_exp_name = "tudd_ablation_scaler_mortality"
    scaler_setting_run_ids = {
        "None": ("5999b4b649ab4033824da3f43898e42b", "90a1808f717d4a529c723536f828dc57"),
        "standardization": ("6bb1c0c0a7c64087b3193217dc18393d", "a68a825193b343cfb001caa47508e324"),
        "robust": ("8af8da8ee87e40ccaf8b2346bb55f376", "20c35b97b88f4429bb8126ca9417d519"),
        "power": ("6303c0526493488990c5df6e4d854cb4", "b576c6260ee64e8aaf98a7876c1083a7"),
    }
    selected_run_ids_scaler = {
        run_id
        for run_ids in scaler_setting_run_ids.values()
        for run_id in ((run_ids,) if isinstance(run_ids, str) else run_ids)
    }
    runs_scaler = list_pipeline_runs(scaler_exp_name)
    runs_scaler = runs_scaler.loc[runs_scaler["mlflow_run_id"].isin(selected_run_ids_scaler)].copy()

    runs_scaler
    return runs_scaler, scaler_exp_name, scaler_setting_run_ids


@app.cell
def _(load_evaluation_data, runs_scaler, scaler_exp_name):
    data_scaler = load_evaluation_data(
        experiment_names=scaler_exp_name,
        pipeline_runs=runs_scaler["mlflow_run_id"],
    )
    data_scaler
    return (data_scaler,)


@app.cell
def _(data_scaler):
    data_scaler["pipeline_mlflow_run_id"].unique()


@app.cell
def _(scaler_setting_run_ids):
    common_kwargs_scaler = {
        "show_ci": False,
        "setting_run_ids": scaler_setting_run_ids,
        "excluded_models_by_setting": None,
        "dataset": "tudd",
    }
    return (common_kwargs_scaler,)


@app.cell
def _(
    common_kwargs_scaler,
    data_scaler,
    model_setups,
    plot_model_setting_performance,
    plt,
    save_figs,
):
    for model_group_s, models_s in model_setups.items():
        fig_s = plot_model_setting_performance(
            data_scaler,
            include_models=models_s,
            legend_title="Feature Scaler",
            title="",
            y_limits="auto",
            **common_kwargs_scaler,
        )
        if save_figs:
            fig_s.savefig(f"plots/preprocessing_comparision/scaler/performance_bar_{model_group_s}.svg")
        plt.show()


@app.cell
def _(
    common_kwargs_scaler,
    data_scaler,
    model_setups,
    plot_model_setting_performance_vs_runtime,
    plt,
    save_figs,
):
    for model_group_s2, models_s2 in model_setups.items():
        fig_s2 = plot_model_setting_performance_vs_runtime(
            data_scaler,
            include_models=models_s2,
            legend_title="Feature Scaler",
            title="",
            log_x=True,
            **common_kwargs_scaler,
        )
        if save_figs:
            fig_s2.savefig(f"plots/preprocessing_comparision/scaler/time_performance_{model_group_s2}.svg")
        plt.show()


if __name__ == "__main__":
    app.run()
