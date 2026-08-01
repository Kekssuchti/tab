import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import sys

    sys.path.insert(0, ".")

    from src.mlflow.evaluation_data import (
        list_pipeline_runs,
        load_evaluation_data,
    )
    from src.exploration.latex_table import performance_table_to_latex

    return list_pipeline_runs, load_evaluation_data, performance_table_to_latex


@app.cell
def _(list_pipeline_runs):
    mortality_exp_name = "tudd_baseline_mortality"
    readmission_exp_name = "tudd_sample_size_hours_to_readmit"

    runs_moratlity = list_pipeline_runs(mortality_exp_name)
    runs_readmission = list_pipeline_runs(readmission_exp_name)
    runs_moratlity = runs_moratlity[runs_moratlity["run_name"].str.contains(r"training-size", na=False)]
    runs_moratlity
    return (
        mortality_exp_name,
        readmission_exp_name,
        runs_moratlity,
        runs_readmission,
    )


@app.cell
def _(
    load_evaluation_data,
    mortality_exp_name,
    readmission_exp_name,
    runs_moratlity,
    runs_readmission,
):
    data_mortality = load_evaluation_data(
        experiment_names=mortality_exp_name,
        pipeline_runs=runs_moratlity["mlflow_run_id"],
    )
    data_readmission = load_evaluation_data(
        experiment_names=readmission_exp_name,
        pipeline_runs=runs_readmission["mlflow_run_id"],
    )
    data_mortality
    return data_mortality, data_readmission


@app.cell
def _(data_readmission):
    data_readmission
    return


@app.cell
def _(data_mortality, data_readmission, performance_table_to_latex):
    latex_mortality = performance_table_to_latex(results=data_mortality, metric="roc_auc", include_ci=False)

    latex_readmission = performance_table_to_latex(results=data_readmission, metric="roc_auc", include_ci=False)
    return latex_mortality, latex_readmission


@app.cell
def _(latex_mortality, latex_readmission):
    print(latex_mortality)
    print(latex_readmission)
    return


@app.cell
def _(data_readmission):
    # change sample size plots
    from src.utils.plot_eval import plot_roc_auc, plot_over_training_size, plot_performance_vs_runtime
    import matplotlib.pyplot as plt

    fig_sample_size = plot_over_training_size(data=data_readmission, ignore_models=[], datasets=("tudd",))

    base_path_read = "./plots/sample_size/readmission/"
    fig_sample_size.savefig(f"{base_path_read}all_models.png")
    plt.show()

    fig_ss_main = plot_over_training_size(data=data_readmission, include_models=["tabpfn-3", "xgboost", "logistic-regression", "tabicl-2", "tabfm"], datasets=("tudd",))
    fig_ss_main.savefig(f"{base_path_read}main_models.png")
    plt.show()
    return plot_over_training_size, plot_performance_vs_runtime, plt


@app.cell
def _(data_mortality, plot_over_training_size, plt):
    fig_sample_size_mort = plot_over_training_size(data=data_mortality, ignore_models=[], datasets=("tudd",))

    base_path_mort = "./plots/sample_size/mortality/"
    fig_sample_size_mort.savefig(f"{base_path_mort}all_models.png")
    plt.show()

    fig_ss_main_mort = plot_over_training_size(data=data_mortality, include_models=["tabpfn-3", "xgboost", "logistic-regression", "tabicl-2", "tabfm"], datasets=("tudd",))
    fig_ss_main_mort.savefig(f"{base_path_mort}main_models.png")
    plt.show()
    return


@app.cell
def _(data_readmission, plot_performance_vs_runtime, plt):
    fig = plot_performance_vs_runtime(data=data_readmission, test_dataset="tudd")

    plt.show()
    return


if __name__ == "__main__":
    app.run()
