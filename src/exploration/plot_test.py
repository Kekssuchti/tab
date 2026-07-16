import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys

    sys.path.insert(0, ".")

    from src.mlflow.evaluation_data import (
        list_pipeline_runs,
        load_evaluation_data,
    )
    from src.utils.evaluation_plot import (
        plot_generalization_gaps,
        plot_performance_vs_runtime,
        plot_roc_auc,
    )

    return (
        list_pipeline_runs,
        load_evaluation_data,
        plot_generalization_gaps,
        plot_performance_vs_runtime,
        plot_roc_auc,
    )


@app.cell
def _(list_pipeline_runs, load_evaluation_data):
    def list_run_id(experiment_name: str = "tab"):
        runs = list_pipeline_runs(experiment_name)
        print(
            runs[
                ["run_name", "mlflow_run_id", "model_instances"]
            ].to_string(index=False)
        )
        return runs


    experiment_name = "tab"
    # list_run_id(experiment_name)
    runs = list_pipeline_runs(experiment_name)

    selected_ids = runs.loc[
        runs["run_name"].str.startswith("2026-07-16"), "mlflow_run_id"
    ].tolist()

    print(selected_ids)

    data = load_evaluation_data(
        experiment_names=experiment_name,
        pipeline_runs=selected_ids,
    )
    return (data,)


@app.cell
def _(data):
    data
    return


@app.cell
def _(data, plot_roc_auc):
    plot_roc_auc(data)
    return


@app.cell
def _(data, plot_generalization_gaps):
    plot_generalization_gaps(data, loss="comparative")
    return


@app.cell
def _(data, plot_performance_vs_runtime):
    plot_performance_vs_runtime(data)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
