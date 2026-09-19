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
    runs = runs.assign(
        n_estimators=runs["run_name"].str.extract(r"grid-default-(\d+)-roc-auc", expand=False)
    )
    if runs["n_estimators"].isna().any():
        invalid_names = runs.loc[runs["n_estimators"].isna(), "run_name"].tolist()
        raise ValueError(f"Could not infer n_estimators from run names: {invalid_names}")
    runs["n_estimators"] = runs["n_estimators"].astype(int)
    runs = runs.sort_values("n_estimators", kind="stable").reset_index(drop=True)
    if runs["n_estimators"].duplicated().any():
        duplicates = runs.loc[runs["n_estimators"].duplicated(keep=False), "n_estimators"].tolist()
        raise ValueError(f"Expected one pipeline run per estimator count; duplicates: {duplicates}")

    setting_run_ids = {
        str(row.n_estimators): row.mlflow_run_id
        for row in runs.itertuples()
    }
    present_models = {
        str(row.n_estimators): set(row.model_instances)
        for row in runs.itertuples()
    }
    all_models = ordered_models(
        [model for models in runs["model_instances"] for model in models]
    )
    excluded_models_by_setting = {
        setting: [model for model in all_models if model not in models]
        for setting, models in present_models.items()
        if set(all_models) - models
    }
    common_models = [
        model for model in all_models if all(model in models for models in present_models.values())
    ]

    runs[["mlflow_run_id", "n_estimators", "model_instances"]]
    return (
        all_models,
        common_models,
        excluded_models_by_setting,
        experiment_name,
        runs,
        setting_run_ids,
    )


@app.cell
def _(experiment_name, load_evaluation_data, runs):
    data = load_evaluation_data(
        experiment_names=experiment_name,
        pipeline_runs=runs["mlflow_run_id"],
    )
    if data.empty:
        raise ValueError(f"No evaluation artifacts found in MLflow experiment {experiment_name!r}")

    point_results = data.loc[
        data["scope"].eq("test") & data["statistic"].eq("point"),
        [
            "pipeline_mlflow_run_id",
            "model_name",
            "dataset",
            "roc_auc",
            "roc_auc_ci_lower",
            "roc_auc_ci_upper",
            "total_time",
        ],
    ]
    point_results
    return (data,)


@app.cell
def _(Path, config, excluded_models_by_setting, setting_run_ids):
    dataset = "tudd"
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
    return


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
    return


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
    # Reuse the paired bootstrap scores already logged by each suite run. The
    # shared bootstrap IDs preserve the original resampling plan and avoid
    # downloading probabilities or recomputing any bootstrap samples.
    client = MlflowClient(tracking_uri=DEFAULT_TRACKING_URI)
    bootstrap_by_estimators = {
        run.n_estimators: pd.read_csv(
            client.download_artifacts(run.mlflow_run_id, ARTIFACT_BOOTSTRAP_METRICS)
        )
        for run in runs.itertuples()
    }

    estimator_win_matrices = {}
    bootstrap_counts = {}
    metadata_columns = list(BOOTSTRAP_METADATA_COLUMNS)
    for model in all_models:
        for local_dataset in ["tudd", "mimic"]:
            model_scores = []
            reference_index = None
            for n_estimators, bootstrap_scores in bootstrap_by_estimators.items():
                if model not in bootstrap_scores:
                    continue
                scores = (
                    bootstrap_scores.loc[
                        bootstrap_scores["dataset"].eq(local_dataset)
                        & bootstrap_scores["metric"].eq(metric),
                        [*metadata_columns, model],
                    ]
                    .set_index(metadata_columns)[model]
                    .rename(str(n_estimators))
                )
                if reference_index is None:
                    reference_index = scores.index
                elif not scores.index.equals(reference_index):
                    raise ValueError(f"Bootstrap IDs differ between estimator runs for {model}")
                model_scores.append(scores)
    
            if not model_scores:
                raise ValueError(f"No logged bootstrap scores found for {model}")
            combined_scores = pd.concat(model_scores, axis=1).reset_index()
            matrix = pairwise_win_matrices(combined_scores)[f"{local_dataset}_{metric}"].copy()
            matrix.index.name = "row n_estimators"
            matrix.columns.name = "column n_estimators"
            estimator_win_matrices[f"{model}_{local_dataset}"] = matrix
            bootstrap_counts[f"{model}_{local_dataset}"] = len(combined_scores)

    mo.vstack(
        [
            mo.md(
                f"""## Paired bootstrap comparisons by estimator count

                **Dataset:** {local_dataset.upper()} · **Metric:** {metric.upper()}  
                Each cell counts how often the row estimator count beats the column estimator count for the same model; ties contribute 0.5.
                """
            ),
            *(
                mo.vstack(
                    [
                        mo.md(f"### {model_label(model)} ({bootstrap_counts[model]:,} bootstraps)"),
                        matrix,
                    ]
                )
                for model, matrix in estimator_win_matrices.items()
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
