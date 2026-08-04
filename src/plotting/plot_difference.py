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
    return (model_setups,)


@app.cell
def _(data_imputation, model_setups, plot_model_setting_performance, plt):
    for model_group, models in model_setups.items():
        fig = plot_model_setting_performance(
            data_imputation,
            include_models=models,
            dataset="tudd",
            setting_labels=["Mean", "KNN (5)", "Native/Mean"],
            legend_title="Missing Value Imputation",
            title="Preprocessing comparison",
        )
        fig.savefig(f"plots/preprocessing_comparision/performance_bar_{model_group}.svg")
        plt.show()
    return


@app.cell
def _(data_imputation, plot_model_setting_performance_vs_runtime, plt):
    fig2 = plot_model_setting_performance_vs_runtime(
        data_imputation,
        dataset="tudd",
        setting_labels=["Mean", "KNN (5)", "Native / Mean"],
        legend_title="Preprocessing",
        log_x=True,
        show_ci=False
    )
    fig2.savefig("plots/preprocessing_comparision/time_performance.svg")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
