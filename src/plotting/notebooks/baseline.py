import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.mlflow.evaluation_data import (
        list_pipeline_runs,
        load_evaluation_data,
    )
    from src.plotting.latex_table import performance_table_to_latex, multiple_latex_tables
    from src.plotting.ablations import plot_model_setting_performance_vs_runtime

    return (
        list_pipeline_runs,
        load_evaluation_data,
        multiple_latex_tables,
        performance_table_to_latex,
        plot_model_setting_performance_vs_runtime,
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
def _(data_los7, data_mortality, data_readmission):
    for data in [data_mortality, data_los7, data_readmission]:
        print(data["experiment_name"][0])
        print(len(data["pipeline_id"].unique())) 
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
def _():
    import matplotlib.pyplot as plt

    from src.plotting.baseline import plot_performance_vs_runtime

    return plot_performance_vs_runtime, plt


@app.cell
def _(data_readmission, plot_performance_vs_runtime, plt):
    plot_performance_vs_runtime(data=data_readmission, test_dataset="tudd", show_ci=False)

    plt.show()
    return


@app.cell
def _(
    data_los7,
    data_mortality,
    data_readmission,
    plot_model_setting_performance_vs_runtime,
    plt,
):
    for df in [data_mortality, data_los7, data_readmission]: 
        fig = plot_model_setting_performance_vs_runtime(
            df,
            run_aggregation="average",
            runtime_metric="total_time",
            show_ci=False,
            ignore_models=["tabpfn-2.6"]
        )
    
        plt.show()
    return


if __name__ == "__main__":
    app.run()
