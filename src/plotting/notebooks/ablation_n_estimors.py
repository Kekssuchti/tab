import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import marimo as mo
    import pandas as pd

    from mlflow import MlflowClient
    from src.config import config
    from src.mlflow.evaluation_data import DEFAULT_TRACKING_URI, list_pipeline_runs, load_evaluation_data
    from src.mlflow.tracking_contract import ARTIFACT_BOOTSTRAP_METRICS
    from src.plotting.ablations import (
        plot_model_setting_performance,
        plot_model_setting_performance_vs_runtime,
    )
    from src.plotting.defaults import model_label, ordered_models
    from src.utils.prediction_metrics import BOOTSTRAP_METADATA_COLUMNS, pairwise_win_matrices

    return (
        ARTIFACT_BOOTSTRAP_METRICS,
        BOOTSTRAP_METADATA_COLUMNS,
        DEFAULT_TRACKING_URI,
        MlflowClient,
        Path,
        config,
        list_pipeline_runs,
        load_evaluation_data,
        mo,
        model_label,
        ordered_models,
        pairwise_win_matrices,
        pd,
        plot_model_setting_performance,
        plot_model_setting_performance_vs_runtime,
    )


@app.cell
def _(list_pipeline_runs, ordered_models):
    experiment_name = "ablation_tudd_n_estimators"
    runs = list_pipeline_runs(experiment_name)
    if runs.empty:
        raise ValueError(f"No pipeline runs found in MLflow experiment {experiment_name!r}")

    # Suite-generated run names contain the synchronized estimator count. Build
    # the setting map from MLflow so newly completed suite runs appear without
    # copying run IDs into the notebook.
    runs = runs.assign(n_estimators=runs["run_name"].str.extract(r"grid-default-(\d+)-roc-auc", expand=False))
    if runs["n_estimators"].isna().any():
        invalid_names = runs.loc[runs["n_estimators"].isna(), "run_name"].tolist()
        raise ValueError(f"Could not infer n_estimators from run names: {invalid_names}")
    runs["n_estimators"] = runs["n_estimators"].astype(int)
    runs = runs.sort_values("n_estimators", kind="stable").reset_index(drop=True)
    runs["repeat"] = runs.groupby("n_estimators", sort=False).cumcount() + 1

    # Plotting helpers identify settings by pipeline ID. Averaged settings use a
    # synthetic ID because several real pipeline runs contribute to each one.
    _estimator_sizes = runs["n_estimators"].drop_duplicates().tolist()
    setting_run_ids = {str(size): f"n_estimators_{size}" for size in _estimator_sizes}
    present_models = {}
    for _n_estimators, group in runs.groupby("n_estimators", sort=False):
        model_sets = [set(models) for models in group["model_instances"]]
        if any(models != model_sets[0] for models in model_sets[1:]):
            raise ValueError(f"Model sets differ between repeats for n_estimators={_n_estimators}")
        present_models[str(_n_estimators)] = model_sets[0]
    all_models = ordered_models([model for models in runs["model_instances"] for model in models])
    excluded_models_by_setting = {
        setting: [model for model in all_models if model not in models]
        for setting, models in present_models.items()
        if set(all_models) - models
    }
    common_models = [model for model in all_models if all(model in models for models in present_models.values())]

    runs[["mlflow_run_id", "n_estimators", "repeat", "model_instances"]]
    return (
        all_models,
        common_models,
        excluded_models_by_setting,
        experiment_name,
        runs,
        setting_run_ids,
    )


@app.cell
def _(
    averaged_bootstrap_intervals,
    experiment_name,
    load_evaluation_data,
    metric,
    runs,
):
    raw_data = load_evaluation_data(
        experiment_names=experiment_name,
        pipeline_runs=runs["mlflow_run_id"],
    )
    if raw_data.empty:
        raise ValueError(f"No evaluation artifacts found in MLflow experiment {experiment_name!r}")

    _run_estimators = runs.set_index("mlflow_run_id")["n_estimators"]
    raw_data["n_estimators"] = raw_data["pipeline_mlflow_run_id"].map(_run_estimators)
    point_data = raw_data.loc[raw_data["scope"].eq("test") & raw_data["statistic"].eq("point")].copy()
    group_columns = [
        "n_estimators",
        "model_name",
        "model_instance",
        "scope",
        "statistic",
        "dataset",
    ]
    numeric_columns = [column for column in point_data.select_dtypes("number").columns if column != "n_estimators"]
    data = point_data.groupby(group_columns, as_index=False, sort=False)[numeric_columns].mean()
    data["pipeline_mlflow_run_id"] = data["n_estimators"].map(lambda value: f"n_estimators_{value}")
    data["pipeline_run_name"] = data["n_estimators"].map(lambda value: f"Average n_estimators={value}")
    _repeat_counts = runs.groupby("n_estimators").size()
    data["repeat_count"] = data["n_estimators"].map(_repeat_counts)

    # Replace the mean of per-run interval endpoints with the interval of the
    # bootstrap scores averaged across repeats.
    lower_column = f"{metric}_ci_lower"
    upper_column = f"{metric}_ci_upper"
    for index, row in data.iterrows():
        interval = averaged_bootstrap_intervals.get((row["model_name"], row["dataset"], row["n_estimators"]))
        if interval is not None:
            data.loc[index, [lower_column, upper_column]] = interval

    data[
        [
            "n_estimators",
            "repeat_count",
            "model_name",
            "dataset",
            "roc_auc",
            "roc_auc_ci_lower",
            "roc_auc_ci_upper",
            "total_time",
        ]
    ]
    return (data,)


@app.cell
def _(Path, config, excluded_models_by_setting, setting_run_ids):
    dataset = "mimic"
    metric = "roc_auc"
    save_figures = True
    output_dir = Path(config.dir_plots) / "ablation_n_estimators"

    common_kwargs = {
        "dataset": dataset,
        "metric": metric,
        "setting_run_ids": setting_run_ids,
        "excluded_models_by_setting": excluded_models_by_setting,
        "legend_title": "Number of estimators",
    }
    return common_kwargs, dataset, metric, output_dir, save_figures


@app.cell
def _(
    all_models,
    common_kwargs,
    data,
    dataset,
    output_dir,
    plot_model_setting_performance,
    save_figures,
):
    performance_figure = plot_model_setting_performance(
        data,
        include_models=all_models,
        title=None,
        y_limits="auto",
        x_axis_label=None,
        show_ci=True,
        **common_kwargs,
    )
    if save_figures:
        output_dir.mkdir(parents=True, exist_ok=True)
        performance_figure.savefig(output_dir / f"{dataset}_performance.svg", bbox_inches="tight")
        performance_figure.savefig(output_dir / f"{dataset}_performance.pdf", bbox_inches="tight")
    performance_figure


@app.cell
def _(
    common_kwargs,
    common_models,
    data,
    dataset,
    output_dir,
    plot_model_setting_performance_vs_runtime,
    save_figures,
):
    # Keep the runtime view readable by comparing only models present at every
    # estimator count; the performance figure above still includes all models.
    runtime_figure = plot_model_setting_performance_vs_runtime(
        data,
        include_models=common_models,
        runtime_metric="total_time",
        log_x=True,
        show_ci=False,
        title=None,
        **common_kwargs,
    )
    if save_figures:
        output_dir.mkdir(parents=True, exist_ok=True)
        runtime_figure.savefig(output_dir / f"{dataset}_performance_vs_runtime.svg", bbox_inches="tight")
        runtime_figure.savefig(output_dir / f"{dataset}_performance_vs_runtime.pdf", bbox_inches="tight")
    runtime_figure


@app.cell
def _(
    ARTIFACT_BOOTSTRAP_METRICS,
    BOOTSTRAP_METADATA_COLUMNS,
    DEFAULT_TRACKING_URI,
    MlflowClient,
    all_models,
    metric,
    mo,
    model_label,
    pairwise_win_matrices,
    pd,
    runs,
):
    # Average each bootstrap ID across repeated runs before comparing estimator
    # counts. This preserves the shared resampling plan and incorporates
    # run-to-run variation without generating new bootstrap samples.
    client = MlflowClient(tracking_uri=DEFAULT_TRACKING_URI)
    bootstrap_runs = [
        (
            run.n_estimators,
            run.repeat,
            pd.read_csv(client.download_artifacts(run.mlflow_run_id, ARTIFACT_BOOTSTRAP_METRICS)),
        )
        for run in runs.itertuples()
    ]

    averaged_bootstrap_intervals = {}
    estimator_win_matrices = {}
    matrix_metadata = {}
    metadata_columns = list(BOOTSTRAP_METADATA_COLUMNS)
    _estimator_sizes = runs["n_estimators"].drop_duplicates().tolist()
    for model in all_models:
        estimator_win_matrices[model] = {}
        for local_dataset in ["tudd", "mimic"]:
            averaged_estimator_scores = []
            _repeat_counts = {}
            reference_index = None
            for _n_estimators in _estimator_sizes:
                repeat_scores = []
                for _run_estimators, repeat, bootstrap_scores in bootstrap_runs:
                    if _run_estimators != _n_estimators or model not in bootstrap_scores:
                        continue
                    scores = (
                        bootstrap_scores.loc[
                            bootstrap_scores["dataset"].eq(local_dataset) & bootstrap_scores["metric"].eq(metric),
                            [*metadata_columns, model],
                        ]
                        .set_index(metadata_columns)[model]
                        .rename(f"repeat_{repeat}")
                    )
                    if reference_index is None:
                        reference_index = scores.index
                    elif not scores.index.equals(reference_index):
                        raise ValueError(f"Bootstrap IDs differ between repeated runs for {model}")
                    repeat_scores.append(scores)

                if not repeat_scores:
                    continue
                repeat_frame = pd.concat(repeat_scores, axis=1)
                averaged_scores = repeat_frame.mean(axis=1)
                averaged_estimator_scores.append(averaged_scores.rename(str(_n_estimators)))
                lower, upper = averaged_scores.quantile([0.025, 0.975])
                averaged_bootstrap_intervals[(model, local_dataset, _n_estimators)] = (
                    float(lower),
                    float(upper),
                )
                _repeat_counts[str(_n_estimators)] = len(repeat_scores)

            if not averaged_estimator_scores:
                raise ValueError(f"No logged bootstrap scores found for {model}")
            combined_scores = pd.concat(averaged_estimator_scores, axis=1).reset_index()
            matrix = pairwise_win_matrices(combined_scores)[f"{local_dataset}_{metric}"].copy()
            matrix.index.name = "row n_estimators"
            matrix.columns.name = "column n_estimators"
            estimator_win_matrices[model][local_dataset] = matrix
            matrix_metadata[(model, local_dataset)] = {
                "bootstrap_count": len(combined_scores),
                "repeat_counts": _repeat_counts,
            }

    comparison_display = mo.vstack(
        [
            mo.md(
                f"""## Paired bootstrap comparisons by estimator count

                **Metric:** {metric.upper()}  
                Bootstrap scores are averaged by bootstrap ID across repeated runs before counting row-versus-column wins; ties contribute 0.5.
                """
            ),
            *(
                mo.vstack(
                    [
                        mo.md(f"### {model_label(model)}"),
                        *(
                            mo.vstack(
                                [
                                    mo.md(
                                        f"#### {local_dataset.upper()} "
                                        f"({matrix_metadata[(model, local_dataset)]['bootstrap_count']:,} bootstraps; "
                                        f"repeats by estimator: {matrix_metadata[(model, local_dataset)]['repeat_counts']})"
                                    ),
                                    matrix,
                                ]
                            )
                            for local_dataset, matrix in matrices.items()
                        ),
                    ]
                )
                for model, matrices in estimator_win_matrices.items()
            ),
        ]
    )
    comparison_display
    return (averaged_bootstrap_intervals,)


if __name__ == "__main__":
    app.run()
