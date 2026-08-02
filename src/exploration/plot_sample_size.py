import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import sys

    sys.path.insert(0, ".")

    from src.plotting.latex_table import performance_table_to_latex
    from src.mlflow.evaluation_data import (
        list_pipeline_runs,
        load_evaluation_data,
    )

    return list_pipeline_runs, load_evaluation_data


@app.cell
def _(list_pipeline_runs):
    mortality_exp_name = "tudd_sample_size_mortality"
    readmission_exp_name = "tudd_sample_size_hours_to_readmit"
    readmission_72_exp_name = "tudd_sample_size_hours_to_readmit_72"
    los_exp_name = "tudd_sample_size_LOS7"

    runs_moratlity = list_pipeline_runs(mortality_exp_name)
    runs_readmission = list_pipeline_runs(readmission_exp_name)
    runs_readmission_72 = list_pipeline_runs(readmission_72_exp_name)
    runs_los7 = list_pipeline_runs(los_exp_name)
    runs_moratlity = runs_moratlity[runs_moratlity["run_name"].str.contains(r"training-size", na=False)]
    runs_moratlity
    return (
        los_exp_name,
        mortality_exp_name,
        readmission_72_exp_name,
        readmission_exp_name,
        runs_los7,
        runs_moratlity,
        runs_readmission,
        runs_readmission_72,
    )


@app.cell
def _(
    load_evaluation_data,
    los_exp_name,
    mortality_exp_name,
    readmission_72_exp_name,
    readmission_exp_name,
    runs_los7,
    runs_moratlity,
    runs_readmission,
    runs_readmission_72,
):
    data_mortality = load_evaluation_data(
        experiment_names=mortality_exp_name,
        pipeline_runs=runs_moratlity["mlflow_run_id"],
    )
    data_readmission = load_evaluation_data(
        experiment_names=readmission_exp_name,
        pipeline_runs=runs_readmission["mlflow_run_id"],
    )
    data_readmission_72 = load_evaluation_data(
        experiment_names=readmission_72_exp_name,
        pipeline_runs=runs_readmission_72["mlflow_run_id"],
    )
    data_los7 = load_evaluation_data(
        experiment_names=los_exp_name,
        pipeline_runs=runs_los7["mlflow_run_id"],
    )
    data_mortality
    return data_los7, data_mortality, data_readmission, data_readmission_72


@app.cell
def _(data_readmission):
    data_readmission
    return


@app.cell
def _(data_los7, data_mortality, data_readmission, data_readmission_72):
    # change sample size plots
    import matplotlib.pyplot as plt

    from src.plotting.evaluation import plot_over_training_size, plot_performance_vs_runtime, plot_roc_auc

    save_figs = True
    setups = {
        "readmission": (data_readmission, "./plots/sample_size/readmission/"),
        "readmission": (data_readmission_72, "./plots/sample_size/readmission_72/"),
        "mortality": (data_mortality, "./plots/sample_size/mortality/"),
        "los7": (data_los7, "./plots/sample_size/LOS7/"),   
    }



    model_setups = {
        "all": ["ebm","orion-msp","limix-16m","tabswift","tabpfn-3", "xgboost", "logistic-regression", "tabicl-2", "tabfm"],
        "main": ["tabpfn-3", "xgboost", "logistic-regression", "tabicl-2", "tabfm"]
    }

    datasets_to_plot = ("tudd",)
    return (
        datasets_to_plot,
        model_setups,
        plot_over_training_size,
        plot_performance_vs_runtime,
        plt,
        save_figs,
        setups,
    )


@app.cell
def _(
    datasets_to_plot,
    model_setups,
    plot_over_training_size,
    plt,
    save_figs,
    setups,
):
    for (exp_data, base_save_path) in setups.values():
        for setting, included_models in model_setups.items():
            fig = plot_over_training_size(data=exp_data, include_models = included_models, datasets=datasets_to_plot, show_title=False)
            if save_figs:
                fig.savefig(f"{base_save_path}{setting}_models.png")
            plt.show()
    return


@app.cell
def _(base_path_mort, data_mortality, plot_over_training_size, plt):
    fig_sample_size_mort = plot_over_training_size(data=data_mortality, ignore_models=[], datasets=("tudd",))

    fig_sample_size_mort.savefig(f"{base_path_mort}all_models.png")
    plt.show()

    fig_ss_main_mort = plot_over_training_size(
        data=data_mortality,
        include_models=["tabpfn-3", "xgboost", "logistic-regression", "tabicl-2", "tabfm"],
        datasets=("tudd",),
    )
    fig_ss_main_mort.savefig(f"{base_path_mort}main_models.png")
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _(data_readmission, plot_performance_vs_runtime, plt):
    fig = plot_performance_vs_runtime(data=data_readmission, test_dataset="tudd")

    plt.show()
    return


if __name__ == "__main__":
    app.run()
