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
    from src.utils.evaluation_plot import plot_roc_auc
    from src.exploration.latex_table import performance_table_to_latex

    return (
        list_pipeline_runs,
        load_evaluation_data,
        performance_table_to_latex,
        plot_roc_auc,
    )


@app.cell
def _(list_pipeline_runs, load_evaluation_data):
    mortality_exp_name = "tudd_baseline_mortality"
    readmission_exp_name = "tudd_baseline_hours_to_readmit"

    runs_moratlity = list_pipeline_runs(mortality_exp_name)
    runs_readmission = list_pipeline_runs(readmission_exp_name)

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
def _(data_mortality, plot_roc_auc):
    plot_roc_auc(data_mortality)
    return


@app.cell
def _(data_readmission, plot_roc_auc):
    plot_roc_auc(data_readmission)
    return


if __name__ == "__main__":
    app.run()
