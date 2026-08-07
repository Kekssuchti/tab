import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.config import config
    from src.plotting.plot_distributions import plot_feature_distributions

    return config, mo, pd, plot_feature_distributions


@app.cell
def _(config, pd):
    filtered_data_dir = config.dir_data / "filtered"
    tudd_data = pd.read_csv(filtered_data_dir / "tudd_mean_100_full.csv")
    return (tudd_data,)


@app.cell
def _(config):
    include_features = None
    exclude_features = None
    bins = "auto"
    kde = True
    save_plots=True
    plots_output_dir = config.dir_plots / "feature_distributions"
    return (
        bins,
        exclude_features,
        include_features,
        kde,
        plots_output_dir,
        save_plots,
    )


@app.cell
def _(
    bins,
    exclude_features,
    include_features,
    kde,
    plot_feature_distributions,
    tudd_data,
):
    plots = plot_feature_distributions(
        tudd=tudd_data,
        include_features=include_features,
        exclude_features=exclude_features,
        bins=bins,
        kde=kde,
    )
    return (plots,)


@app.cell
def _(mo, plots):
    mo.vstack(
        [
            mo.vstack([mo.md(f"### `{feature_name}`"), figure])
            for feature_name, figure in plots.items()
        ]
    )
    return


@app.cell
def _(plots, plots_output_dir, save_plots):
    if save_plots:
        for feature_name, plot in plots.items():
            plot.savefig(plots_output_dir / f"{feature_name}.svg")
    return


if __name__ == "__main__":
    app.run()
