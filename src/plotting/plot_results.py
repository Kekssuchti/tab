import matplotlib.pyplot as plt

from src.mlflow.evaluation_data import list_pipeline_runs, load_evaluation_data
from src.plotting.evaluation import (
    plot_generalization_gaps,
    plot_performance_vs_runtime,
    plot_roc_auc,
)
from src.plotting.defaults import set_plot_style

if __name__ == "__main__":
    set_plot_style()
    experiment_name = "tudd_baseline_mortality"
    runs = list_pipeline_runs(experiment_name)
    print(runs[["run_name", "mlflow_run_id", "model_instances"]].to_string(index=False))

    selected_ids = runs.loc[runs["run_name"].str.startswith("training"), "mlflow_run_id"].tolist()
    results = load_evaluation_data(
        experiment_names=experiment_name,
        pipeline_runs=selected_ids,
    )
    plot_roc_auc(results)
    plot_generalization_gaps(results, loss="comparative")
    plot_performance_vs_runtime(results)
    plt.show()
