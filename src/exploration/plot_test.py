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
    from matplotlib import pyplot as plt
    from src.config import config

    return (
        config,
        list_pipeline_runs,
        load_evaluation_data,
        plot_generalization_gaps,
        plot_performance_vs_runtime,
        plot_roc_auc,
        plt,
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
    list_run_id(experiment_name)
    runs = list_pipeline_runs(experiment_name)

    run_ids_to_analyze = [
        "7e34d3f5359a4e9cb5141cf4b9c1fa1f",
        "d599d0402eaa4abcb6db9a201326e27a",
        "ffa0f74a73164834b01db2bac7967ddb",
        "c87f8ffb2278400f96b24e157b81b373",
        "b580c3e2183b4a09a6301b11665cfe45",
        "147f2acdbb4046509496937b33551c35",
        "cc4a485361474a77ae9a589a83a86136",
        "1d874c9297e740e8839388ce2d2ac7ac",
        "1ef3b80164784b7c89bc692caa77c9df",
        "270eea42c596414fa91698327338dc05"
    ]


    selected_ids = runs.loc[
        runs["mlflow_run_id"].isin(run_ids_to_analyze),
        "mlflow_run_id",
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
def _(data):
    data["pipeline_run_name"].unique()
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
def _(data):
    data["metric"].unique()
    return


@app.cell
def _(config, data, plt):
    df = data.copy()
    for metric in ["roc_auc", "prc_auc", "f1"]:
        for test_set in ["tudd", "mimic"]:
            plot_df = df[
                (df["metric"] == metric)
                & (df["dataset"] == test_set)
            ].copy()
        
            for model in df["model_name"].unique():
                model_df = (
                    plot_df[plot_df["model_name"] == model]
                    .sort_values("training_size")
                )
        
                line, = plt.plot(
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
            plt.savefig(config.dir_plots / f"trainsize_{test_set}_{metric}")
            plt.show()
        
    return


@app.cell
def _(config, data, plt):
    def run():
        df = data.copy()

        times = df[
            (df["scope"] != "cv")
            & (df["metric"].isin(["fit_time", "predict_time"]))
        ].copy()

        times["component"] = times["dataset"].map({
            "mimic": "Predict MIMIC",
            "tudd": "Predict TUDD",
        })
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
            timings["model_name"]
            + " (n="
            + timings["training_size"].astype(str)
            + ")"
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
        plt.savefig(config.dir_plots / "train_pred_time.png")
        plt.show()


    run()
    return


if __name__ == "__main__":
    app.run()
