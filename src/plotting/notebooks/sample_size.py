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
def _(data_los7, data_mortality, data_readmission_72):
    for data in [data_mortality, data_los7, data_readmission_72]:
        print(data["experiment_name"][0])
        print(len(data["pipeline_id"].unique()) / 10)
    return


@app.cell
def _(data_los7, data_mortality, data_readmission_72):
    # change sample size plots
    import matplotlib.pyplot as plt

    from src.plotting.sample_size import plot_over_training_size
    from src.plotting.sample_size_difference import plot_difference_training_size

    save_figs = True
    setups = {
        # "readmission": (data_readmission, "./plots/sample_size/readmission/"),
        "mortality": (data_mortality, "./plots/sample_size/mortality/"),
        "los7": (data_los7, "./plots/sample_size/LOS7/"),
        "readmission_72": (data_readmission_72, "./plots/sample_size/readmission_72/"),
    }

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

    datasets_to_plot = ("tudd",)

    for ds, _ in setups.values():
        ds["training_time"] = ds["cv_time"] + ds["fit_time"]

    return (
        datasets_to_plot,
        model_setups,
        plot_difference_training_size,
        plot_over_training_size,
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
    y_axis_metric="training_time"
    for exp_data, base_save_path in setups.values():
        for setting, included_models in model_setups.items():
            fig = plot_over_training_size(
                data=exp_data,
                include_models=included_models,
                datasets=datasets_to_plot,
                run_aggregation="average",
                show_title=False,
                metric=y_axis_metric,
                y_label= "Training time (s, log scale)"
            )
            if save_figs:
                fig.tight_layout()
                fig.savefig(f"{base_save_path}{setting}_{y_axis_metric}.svg")
            plt.show()
    return


@app.cell
def _(
    datasets_to_plot,
    model_setups,
    plot_difference_training_size,
    plt,
    save_figs,
    setups,
):
    _baseline_model = "xgboost"
    for _exp_data, _base_save_path in setups.values():
        for _setting, _included_models in model_setups.items():
            _comparison_models = [_model for _model in _included_models if _model != _baseline_model]
            _difference_fig = plot_difference_training_size(
                data=_exp_data,
                baseline_model=_baseline_model,
                compare_models=_comparison_models,
                datasets=datasets_to_plot,
                run_aggregation="average",
                show_title=False,
                metric="prc_auc",
            )
            if save_figs:
                _difference_fig.tight_layout()
                _difference_fig.savefig(f"{_base_save_path}{_setting}_difference_xgboost.svg")
            plt.show()
    return


if __name__ == "__main__":
    app.run()
