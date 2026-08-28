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
    from src.plotting.ablations import plot_model_setting_performance_vs_runtime

    return (
        list_pipeline_runs,
        load_evaluation_data,
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
    for daa in [data_mortality, data_los7, data_readmission]:
        print(daa["experiment_name"][0])
        print(len(daa["pipeline_id"].unique()))
    return


@app.cell
def _(data_los7, data_mortality, data_readmission):
    import matplotlib.pyplot as plt

    save_figs = True
    figure_size = (6, 6)
    setups = {
        "mortality": (data_mortality, "./plots/baseline/mortality/"),
        "los7": (data_los7, "./plots/baseline/LOS7/"),
        "readmission": (data_readmission, "./plots/baseline/readmission/"),
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

    # Adjust total time since MIMIC prediction is not relevant right now.
    for ds, _ in setups.values():
        ds["total_time"] = ds["total_time"] - ds["predict_time_mimic"]
        ds["training_time"] = ds["cv_time"] + ds["fit_time"]

    return figure_size, model_setups, plt, save_figs, setups


@app.cell
def _(
    figure_size,
    model_setups,
    plot_model_setting_performance_vs_runtime,
    plt,
    save_figs,
    setups,
):
    for data, save_path in setups.values():
        for setting, included_models in model_setups.items():
            fig = plot_model_setting_performance_vs_runtime(
                data,
                include_models=included_models,
                run_aggregation="average",
                metric="prc_auc",
                runtime_metric="training_time",
                show_ci=False,
                x_axis_label="Model preparation time (s, log scale)"
            )
            fig.set_size_inches(*figure_size, forward=True)
            if save_figs:
                fig.savefig(f"{save_path}{setting}_performance_training_time.svg")
            plt.show()
    return


@app.cell
def _(
    figure_size,
    model_setups,
    plot_model_setting_performance_vs_runtime,
    plt,
    save_figs,
    setups,
):
    for _data, _save_path in setups.values():
        for _setting, _included_models in model_setups.items():
            _fig = plot_model_setting_performance_vs_runtime(
                _data,
                include_models=_included_models,
                run_aggregation="average",
                metric="prc_auc",
                runtime_metric="predict_time_tudd",
                show_ci=False,
                x_axis_label="Predict time (s, log scale)"
            )
            _fig.set_size_inches(*figure_size, forward=True)
            if save_figs:
                _fig.savefig(f"{_save_path}{_setting}_performance_test_time.svg")
            plt.show()
    return


@app.cell
def _():
    """Render a compact cross-task model performance table."""

    import collections.abc

    import pandas as pd

    from src.plotting.defaults import metric_label, model_label, ordered_models

    __all__ = ["performance_table_to_latex"]

    _SUPPORTED_METRICS = ("roc_auc", "prc_auc")
    _EVALUATION_COLUMNS = ("model_name", "scope", "statistic", "dataset")

    def _format_score(score: pd.Series, metric: str, show_ci: bool, *, bold: bool) -> str:
        point = score[metric]
        if pd.isna(point):
            return "--"

        formatted = f"{point * 100:.2f}"
        if bold:
            formatted = rf"\textbf{{{formatted}}}"
        if not show_ci:
            return formatted

        lower, upper = (score[f"{metric}_ci_{bound}"] for bound in ("lower", "upper"))
        if pd.isna(lower) or pd.isna(upper):
            return "--"
        return f"{formatted} [{lower * 100:.2f}, {upper * 100:.2f}]"

    def _escape_latex(value: object) -> str:
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        return "".join(replacements.get(character, character) for character in str(value))

    def performance_table_to_latex(
        task_frames: collections.abc.Mapping[str, pd.DataFrame],
        show_ci: bool = True,
        exclude_models: list[str] | None = None,
        metrics: list[str] = ["roc_auc"],
    ) -> str:
        """Return a LaTeX table of TUDD test performance for the supplied tasks.

        Repeated evaluation runs are averaged per model within each task. Model
        rows are shared across tasks, so a model absent from one task is shown as
        ``--`` in that cell.
        """
        if not metrics or len(metrics) != len(set(metrics)) or not set(metrics) <= set(_SUPPORTED_METRICS):
            raise ValueError(f"metrics must contain one or both of: {', '.join(_SUPPORTED_METRICS)}")

        excluded = set(exclude_models or [])
        ci_columns = [f"{metric}_ci_{bound}" for metric in metrics for bound in ("lower", "upper")]
        value_columns = [*metrics, *(ci_columns if show_ci else ())]
        required_columns = [*_EVALUATION_COLUMNS, *value_columns]
        scores_by_task: dict[str, pd.DataFrame] = {}
        model_names: list[str] = []

        for task_name, frame in task_frames.items():
            missing = [column for column in required_columns if column not in frame.columns]
            if missing:
                raise ValueError(f"Task {task_name!r} is missing required columns: {', '.join(missing)}")

            selected = frame.loc[
                frame["scope"].eq("test")
                & frame["statistic"].eq("point")
                & frame["dataset"].eq("tudd")
                & ~frame["model_name"].isin(excluded),
                ["model_name", *value_columns],
            ]
            model_names.extend(selected["model_name"].drop_duplicates().tolist())
            scores_by_task[task_name] = selected.groupby("model_name", sort=False)[value_columns].mean()

        tasks = list(task_frames)
        best_by_task = {
            (task_name, metric): scores[metric].max()
            for task_name, scores in scores_by_task.items()
            for metric in metrics
        }

        column_spec = "l" + "c" * (len(tasks) * len(metrics))
        if len(metrics) == 1:
            header = " & ".join([r"\textbf{Model}", *[rf"\textbf{{{_escape_latex(task)}}}" for task in tasks]])
            metric_header = []
        else:
            header = " & ".join(
                [r"\textbf{Model}", *[rf"\multicolumn{{2}}{{c}}{{\textbf{{{_escape_latex(task)}}}}}" for task in tasks]]
            )
            metric_header = [
                " & ".join(["", *[rf"\textbf{{{metric_label(metric)}}}" for _ in tasks for metric in metrics]])
                + " "
                + r"\\"
            ]
        display_metrics = " and ".join(metric_label(metric) for metric in metrics)
        lines = [
            r"\begin{table*}[htbp]",
            r"\centering",
            rf"\caption{{Model {display_metrics} performance across Mortality, Length of Stay, and Readmission.}}",
            r"\label{tab:results_baseline_performance}",
            rf"\begin{{tabular}}{{{column_spec}}}",
            r"\toprule",
            header + " \\\\",
            *metric_header,
            r"\midrule",
        ]

        for model_name in ordered_models(model_names):
            cells = [_escape_latex(model_label(model_name))]
            for task_name in tasks:
                scores = scores_by_task[task_name]
                if model_name not in scores.index:
                    cells.extend(["--"] * len(metrics))
                    continue

                score = scores.loc[model_name]
                for metric in metrics:
                    point = score[metric]
                    is_best = not pd.isna(point) and point == best_by_task[task_name, metric]
                    cells.append(_format_score(score, metric, show_ci, bold=is_best))
            lines.append(" & ".join(cells) + " \\\\")
            if model_name == "xgboost":
                lines.append(r"\midrule")

        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table*}",
            ]
        )
        return "\n".join(lines)

    return (performance_table_to_latex,)


@app.cell
def _(data_los7, data_mortality, data_readmission, performance_table_to_latex):
    table = performance_table_to_latex(
        task_frames={
            "Mortality": data_mortality,
            "Length of Stay": data_los7,
            "Readmission": data_readmission,
        },
        exclude_models=["tabpfn-2.6"],
        show_ci=True,
        metrics=["prc_auc"],
    )
    print(table)
    return


if __name__ == "__main__":
    app.run()
