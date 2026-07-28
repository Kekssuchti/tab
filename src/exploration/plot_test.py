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
    from src.utils.evaluation_plot import (
        plot_generalization_gaps,
        plot_performance_vs_runtime,
        plot_roc_auc,
    )
    from src.exploration.latex_table import performance_table_to_latex
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    return (
        list_pipeline_runs,
        load_evaluation_data,
        mticker,
        performance_table_to_latex,
        plot_generalization_gaps,
        plot_performance_vs_runtime,
        plot_roc_auc,
        plt,
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
    latex_mortality = performance_table_to_latex(
        results=data_mortality, metric="roc_auc", include_ci=False
    )

    latex_readmission = performance_table_to_latex(
        results=data_readmission, metric="roc_auc", include_ci=False
    )
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


@app.cell
def _(data, plot_generalization_gaps):
    plot_generalization_gaps(data, loss="comparative")
    return


@app.cell
def _(data, plot_performance_vs_runtime):
    plot_performance_vs_runtime(data)
    return


@app.cell
def _(data):
    data["metric"].unique()
    return


@app.cell
def _(data, plt):
    df = data.copy()
    for metric in ["roc_auc", "prc_auc", "f1"]:
        for test_set in ["tudd", "mimic"]:
            plot_df = df[(df["metric"] == metric) & (df["dataset"] == test_set)].copy()

            for model in df["model_name"].unique():
                model_df = plot_df[plot_df["model_name"] == model].sort_values(
                    "training_size"
                )

                (line,) = plt.plot(
                    model_df["training_size"],
                    model_df["value"],
                    marker="o",
                    label=model,
                )

                plt.fill_between(
                    model_df["training_size"],
                    model_df["ci_lower"],
                    model_df["ci_upper"],
                    alpha=0.15,
                )

            plt.xlabel("Training size")
            plt.ylabel(f"{metric} on {test_set}")
            plt.legend(title="Model")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            # plt.savefig(config.dir_plots / f"trainsize_{test_set}_{metric}")
            plt.show()
    return


@app.cell
def _(data, mticker, plt):
    import numpy as np

    def plot_sample_size_growth():
        df = data.copy()

        models = df["model_name"].unique()
        metrics = ["roc_auc"]  # , "prc_auc", "f1"
        test_sets = ["tudd", "mimic"]

        metric_labels = {
            "roc_auc": "ROC AUC",
            "prc_auc": "PRC AUC",
            "f1": "F1 score",
        }

        palette = plt.colormaps["tab10"].colors
        model_colors = {
            model: palette[i % len(palette)] for i, model in enumerate(models)
        }

        fig, axes = plt.subplots(
            nrows=len(metrics),
            ncols=len(test_sets),
            figsize=(11, 11),
            sharex=True,
            squeeze=False,
        )

        for row, metric in enumerate(metrics):
            for col, test_set in enumerate(test_sets):
                ax = axes[row, col]

                plot_df = df[
                    (df["metric"] == metric) & (df["dataset"] == test_set)
                ].copy()

                for model in models:
                    model_df = plot_df[plot_df["model_name"] == model].sort_values(
                        "training_size"
                    )

                    color = model_colors[model]

                    ax.plot(
                        model_df["training_size"],
                        model_df["value"],
                        color=color,
                        marker="o",
                        markersize=5,
                        linewidth=2,
                        markeredgecolor="white",
                        markeredgewidth=0.8,
                        label=model,
                        zorder=3,
                    )

                    ax.fill_between(
                        model_df["training_size"],
                        model_df["ci_lower"],
                        model_df["ci_upper"],
                        color=color,
                        alpha=0.12,
                        linewidth=0,
                        zorder=2,
                    )

                ax.set_title(
                    test_set.upper(),
                    fontsize=12,
                    fontweight="semibold",
                    pad=10,
                )

                """ax.set_xscale(
                    "function",
                    functions=(
                        lambda x: np.sqrt(x),
                        lambda x: x**2,
                    ),
                )"""

                ax.set_xscale("linear")

                # Display 100, 1000, etc. rather than 10², 10³
                ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
                ax.xaxis.set_minor_formatter(mticker.NullFormatter())

                # A horizontal grid is usually sufficient for this plot
                ax.grid(
                    axis="y",
                    linewidth=0.8,
                    alpha=0.2,
                )
                ax.grid(axis="x", visible=True)

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_alpha(0.4)
                ax.spines["bottom"].set_alpha(0.4)

                ax.tick_params(length=0, pad=6)

                if col == 0:
                    ax.set_ylabel(metric_labels[metric], fontsize=11)

                if row == len(metrics) - 1:
                    ax.set_xlabel("Training size", fontsize=11)

        # One shared legend instead of six repeated legends
        handles, labels = axes[0, 0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            title="Model",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=min(len(labels), 5),
            frameon=False,
        )

        fig.suptitle(
            "Model performance by training-set size",
            fontsize=15,
            fontweight="semibold",
            y=1.05,
        )

        fig.tight_layout(rect=(0, 0, 1, 0.97))
        plt.show()

    plot_sample_size_growth()
    return (np,)


@app.cell
def _(data, np, plt):
    def plot_2():
        df = data.copy()

        models = df["model_name"].unique()
        metrics = ["roc_auc"]  # , "prc_auc", "f1"
        test_sets = ["tudd", "mimic"]

        metric_labels = {
            "roc_auc": "ROC AUC",
            "prc_auc": "PRC AUC",
            "f1": "F1 score",
        }

        palette = plt.colormaps["tab10"].colors
        model_colors = {
            model: palette[i % len(palette)] for i, model in enumerate(models)
        }

        for metric in ["roc_auc", "prc_auc", "f1"]:
            for test_set in ["tudd", "mimic"]:
                fig, ax = plt.subplots(figsize=(7, 5))

                plot_df = df[
                    (df["metric"] == metric) & (df["dataset"] == test_set)
                ].copy()

                # Shared positions across all models in this panel
                training_sizes = np.sort(plot_df["training_size"].unique())

                x_positions = {
                    size: position for position, size in enumerate(training_sizes)
                }

                for model in plot_df["model_name"].unique():
                    model_df = plot_df[plot_df["model_name"] == model].sort_values(
                        "training_size"
                    )

                    x = model_df["training_size"].map(x_positions).to_numpy()

                    (line,) = ax.plot(
                        x,
                        model_df["value"],
                        marker="o",
                        markersize=5,
                        linewidth=2,
                        label=model,
                    )

                    ax.fill_between(
                        x,
                        model_df["ci_lower"],
                        model_df["ci_upper"],
                        color=line.get_color(),
                        alpha=0.15,
                        linewidth=0,
                    )

                tick_sizes = [size for size in training_sizes if size % 2000 == 0]
                tick_sizes.extend([500, 1000])

                tick_positions = [x_positions[size] for size in tick_sizes]
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(
                    [f"{size:,}" for size in tick_sizes],
                    rotation=30,
                    ha="right",
                )

                ax.set_xlabel("Training size")
                ax.set_ylabel(f"{metric} on {test_set}")
                ax.legend(title="Model", frameon=False)
                ax.grid(axis="y", alpha=0.25)

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                fig.tight_layout()
                # fig.savefig(config.dir_plots / f"trainsize2_{test_set}_{metric}.png")
                plt.show()

    plot_2()
    return


@app.cell
def _(data, plt):
    def run():
        df = data.copy()

        times = df[
            (df["scope"] != "cv") & (df["metric"].isin(["fit_time", "predict_time"]))
        ].copy()

        times["component"] = times["dataset"].map(
            {
                "mimic": "Predict MIMIC",
                "tudd": "Predict TUDD",
            }
        )
        times.loc[times["metric"] == "fit_time", "component"] = "Train"

        components = ["Train", "Predict MIMIC", "Predict TUDD"]

        timings = (
            times.pivot_table(
                index=[
                    "pipeline_mlflow_run_id",
                    "model_name",
                    "training_size",
                ],
                columns="component",
                values="value",
                aggfunc="sum",
                fill_value=0,
            )
            .reindex(columns=components, fill_value=0)
            .reset_index()
        )

        timings["Total"] = timings[components].sum(axis=1)
        timings = timings.sort_values("Total")

        timings["label"] = (
            timings["model_name"] + " (n=" + timings["training_size"].astype(str) + ")"
        )

        ax = timings.set_index("label")[components].plot(
            kind="barh",
            stacked=True,
            figsize=(12, max(5, len(timings) * 0.35)),
            width=0.75,
            color=["#4C78A8", "#F2A541", "#59A14F"],
        )

        offset = timings["Total"].max() * 0.01

        for position, total in enumerate(timings["Total"]):
            ax.text(
                total + offset,
                position,
                f"{total:.2f}s",
                va="center",
                fontsize=8,
            )

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("")
        ax.set_title("Training and prediction runtime")
        ax.legend(title="Stage")
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)

        plt.tight_layout()
        # plt.savefig(config.dir_plots / "train_pred_time.png")
        plt.show()

    run()
    return


if __name__ == "__main__":
    app.run()
