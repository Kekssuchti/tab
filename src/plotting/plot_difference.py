import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import sys

    sys.path.insert(0, ".")

    from src.mlflow.evaluation_data import (
        list_pipeline_runs,
        load_evaluation_data,
    )
    from src.plotting.evaluation import plot_model_setting_performance, plot_model_setting_performance_vs_runtime
    import matplotlib.pyplot as plt

    return (
        list_pipeline_runs,
        load_evaluation_data,
        plot_model_setting_performance,
        plot_model_setting_performance_vs_runtime,
        plt,
    )


@app.cell
def _(list_pipeline_runs):
    exploration_exp_name = "tudd_baseline_mean_standardize_mortality"
    runs_exploration = list_pipeline_runs(exploration_exp_name)

    runs_exploration
    return exploration_exp_name, runs_exploration


@app.cell
def _(exploration_exp_name, load_evaluation_data, runs_exploration):
    data_imputation = load_evaluation_data(
        experiment_names=exploration_exp_name,
        pipeline_runs=runs_exploration["mlflow_run_id"],
    )
    data_imputation
    return (data_imputation,)


@app.cell
def _():
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
        "setting_labels": ["Mean", "KNN (5)", "Native/Mean"],
        "dataset": "tudd",
    }
    save_figs = True
    return common_kwargs, model_setups, save_figs


@app.cell
def _(
    common_kwargs,
    data_imputation,
    model_setups,
    plot_model_setting_performance,
    plt,
    save_figs,
):
    for model_group, models in model_setups.items():
        fig = plot_model_setting_performance(
            data_imputation,
            include_models=models,
            legend_title="Missing Value Imputation",
            title="Preprocessing comparison",
            y_limits="auto",
            **common_kwargs,
        )
        if save_figs:
            fig.savefig(f"plots/preprocessing_comparision/performance_bar_{model_group}.svg")
        plt.show()
    return


@app.cell
def _(
    common_kwargs,
    data_imputation,
    model_setups,
    plot_model_setting_performance_vs_runtime,
    plt,
    save_figs,
):
    for model_group2, models2 in model_setups.items():
        fig2 = plot_model_setting_performance_vs_runtime(
            data_imputation,
            include_models=models2,
            legend_title="Preprocessing",
            log_x=True,
            **common_kwargs,
        )
        if save_figs:
            fig2.savefig(f"plots/preprocessing_comparision/time_performance_{model_group2}.svg")
        plt.show()
    return


if __name__ == "__main__":
    app.run()
