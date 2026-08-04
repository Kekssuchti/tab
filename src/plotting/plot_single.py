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
    from src.plotting.latex_table import performance_table_to_latex, multiple_latex_tables

    return (
        list_pipeline_runs,
        load_evaluation_data,
        multiple_latex_tables,
        performance_table_to_latex,
    )


@app.cell
def _(list_pipeline_runs):
    mortality_exp_name = "tudd_baseline_mortality"
    readmission_exp_name = "tudd_baseline_hours_to_readmit_72"
    los7_exp_name = "tudd_baseline_LOS7"

    runs_moratlity = list_pipeline_runs(mortality_exp_name)
    runs_readmission = list_pipeline_runs(readmission_exp_name)
    runs_los7 = list_pipeline_runs(los7_exp_name)
    runs_moratlity
    return (
        los7_exp_name,
        mortality_exp_name,
        readmission_exp_name,
        runs_los7,
        runs_moratlity,
        runs_readmission,
    )


@app.cell
def _(
    load_evaluation_data,
    los7_exp_name,
    mortality_exp_name,
    readmission_exp_name,
    runs_los7,
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
    data_los7 = load_evaluation_data(
        experiment_names=los7_exp_name,
        pipeline_runs=runs_los7["mlflow_run_id"],
    )
    data_mortality
    return data_los7, data_mortality, data_readmission


@app.cell
def _(data_readmission):
    data_readmission
    return


@app.cell
def _(data_mortality, performance_table_to_latex):
    common_kwargs = {
        "metric": "roc_auc",
        "include_ci": True,
        "run_aggregation": "average",
        "dataset_order": ("tudd",),
        "include_generalizability": False,
        "line_width_percent": 0.5,
    }

    latex_mortality = performance_table_to_latex(results=data_mortality, **common_kwargs)
    print(latex_mortality)
    return common_kwargs, latex_mortality


@app.cell
def _(common_kwargs, data_los7, data_readmission, performance_table_to_latex):
    latex_readmission = performance_table_to_latex(results=data_readmission, **common_kwargs)
    latex_los7 = performance_table_to_latex(results=data_los7, **common_kwargs)
    return latex_los7, latex_readmission


@app.cell
def _(latex_los7, latex_mortality, latex_readmission):
    print(latex_mortality)
    print(latex_readmission)
    print(latex_los7)
    return


@app.cell
def _(
    common_kwargs,
    data_los7,
    data_mortality,
    data_readmission,
    multiple_latex_tables,
):
    results = [data_mortality, data_los7, data_readmission]
    names = ["Mortality", "Length of Stay", "Readmission"]

    multi_table = multiple_latex_tables(results, names, kwargs=common_kwargs)
    print(multi_table)
    return


@app.cell
def _(data_mortality, data_readmission):
    # change sample size plots
    from src.plotting.evaluation import plot_over_training_size, plot_performance_vs_runtime
    import matplotlib.pyplot as plt

    main_models = ["tabpfn-3", "xgboost", "logistic-regression", "tabicl-2", "tabfm"]
    for plot_data, base_path in (
        (data_readmission, "./plots/sample_size/readmission/"),
        (data_mortality, "./plots/sample_size/mortality/"),
    ):
        for filename, included_models in (("all_models.png", None), ("main_models.png", main_models)):
            fig_sample_size = plot_over_training_size(
                data=plot_data, include_models=included_models, datasets=("tudd",)
            )
            fig_sample_size.savefig(f"{base_path}{filename}")
            plt.show()
    return plot_over_training_size, plot_performance_vs_runtime, plt


@app.cell
def _(data_readmission, plot_performance_vs_runtime, plt):
    plot_performance_vs_runtime(data=data_readmission, test_dataset="tudd")

    plt.show()
    return


if __name__ == "__main__":
    app.run()
