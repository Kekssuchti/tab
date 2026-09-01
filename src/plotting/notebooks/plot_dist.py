import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path

    import pandas as pd

    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))

    import matplotlib.pyplot as plt

    from src.config import config
    from src.plotting.plot_distributions import plot_feature_distributions

    return config, pd, plot_feature_distributions, plt


@app.cell
def _(config, pd):
    filtered_data_dir = config.dir_data / "filtered"
    tudd_data = pd.read_csv(filtered_data_dir / "tudd_mean_100_full.csv")
    mimic_data = pd.read_csv(filtered_data_dir / "mimic4_mean_100_full.csv")
    return filtered_data_dir, tudd_data


@app.cell
def _(tudd_data):
    tudd_data
    return


@app.cell
def _(config):
    exclude_features = None
    save_plots = True
    plots_output_dir = config.dir_plots / "feature_distributions" / "tudd"
    return exclude_features, plots_output_dir, save_plots


@app.cell
def _(
    exclude_features,
    plot_feature_distributions,
    plots_output_dir,
    plt,
    save_plots,
    tudd_data,
):
    figs = plot_feature_distributions(
        tudd=tudd_data,
        # mimic=mimic_data,
        exclude_features=exclude_features,
    )

    for feature_name, fig in figs.items():
        if save_plots:
            fig.savefig(f"{plots_output_dir}/{feature_name}.svg")
        plt.show()
    return


@app.cell
def _(config, filtered_data_dir, pd):
    read_tudd_data = pd.read_csv(filtered_data_dir / "tudd_readmission.csv")
    read_mimic_data = pd.read_csv(filtered_data_dir / "mimic4_readmission.csv")

    read_tudd_data.columns
    exclude_cols = list(read_tudd_data.columns)
    exclude_cols.remove("hours_to_readmit")

    plots_output_dir_read = config.dir_plots / "feature_distributions" / "tudd"
    return exclude_cols, plots_output_dir_read, read_mimic_data, read_tudd_data


@app.cell
def _(read_mimic_data):
    read_mimic_data
    return


@app.cell
def _(read_mimic_data):
    read_mimic_data["hours_to_readmit"].gt(3*24).sum()
    return


@app.cell
def _(read_mimic_data):
    read_mimic_data_c = read_mimic_data[read_mimic_data["hours_to_readmit"] < 24000]
    return


@app.cell
def _(
    exclude_cols,
    plot_feature_distributions,
    plots_output_dir_read,
    plt,
    read_tudd_data,
    save_plots,
):
    figs_r = plot_feature_distributions(
        tudd=read_tudd_data,
        #mimic=read_mimic_data_c,
        exclude_features=exclude_cols,
    )

    for feature_name_r, fig_r in figs_r.items():
        if save_plots:
            fig_r.savefig(f"{plots_output_dir_read}/{feature_name_r}.svg")
        plt.show()
    return


if __name__ == "__main__":
    app.run()
