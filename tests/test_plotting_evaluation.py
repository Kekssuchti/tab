import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.plotting.evaluation import (
    plot_model_setting_performance,
    plot_model_setting_performance_vs_runtime,
    plot_over_training_size,
)


def _repeated_run_data() -> pd.DataFrame:
    rows = []
    dataset_offset = {"tudd": 0.0, "mimic": 0.1}
    for pipeline_run, run_offset in (("run-a", 0.0), ("run-b", 0.2)):
        for dataset, offset in dataset_offset.items():
            for training_size, size_offset in ((100, 0.0), (200, 0.1)):
                score = 0.6 + run_offset + offset + size_offset
                rows.append(
                    {
                        "pipeline_mlflow_run_id": pipeline_run,
                        "scope": "test",
                        "dataset": dataset,
                        "model_name": "xgboost",
                        "model_instance": "xgboost",
                        "training_size": training_size,
                        "roc_auc": score,
                        "roc_auc_ci_lower": score - 0.1,
                        "roc_auc_ci_upper": score + 0.1,
                    }
                )
    return pd.DataFrame(rows)


def test_plot_over_training_size_averages_repeated_runs_and_ci_by_dataset() -> None:
    figure = plot_over_training_size(
        _repeated_run_data(),
        datasets=("tudd", "mimic"),
        run_aggregation="average",
        log_x=False,
    )

    try:
        expected_by_dataset = {
            "tudd": np.array([0.7, 0.8]),
            "mimic": np.array([0.8, 0.9]),
        }
        for axis, dataset in zip(figure.axes, ("tudd", "mimic"), strict=True):
            assert len(axis.lines) == 1
            np.testing.assert_array_equal(axis.lines[0].get_xdata(), [100, 200])
            np.testing.assert_allclose(axis.lines[0].get_ydata(), expected_by_dataset[dataset])

            vertices = axis.collections[0].get_paths()[0].vertices
            for training_size, score in zip((100, 200), expected_by_dataset[dataset], strict=True):
                for bound in (score - 0.1, score + 0.1):
                    assert np.any(
                        np.isclose(vertices[:, 0], training_size) & np.isclose(vertices[:, 1], bound)
                    )
    finally:
        plt.close(figure)


def test_plot_over_training_size_does_not_average_by_default() -> None:
    figure = plot_over_training_size(_repeated_run_data(), datasets=("tudd",), log_x=False)

    try:
        line = figure.axes[0].lines[0]
        x_values = np.asarray(line.get_xdata())
        y_values = np.asarray(line.get_ydata())

        np.testing.assert_array_equal(x_values, [100, 100, 200, 200])
        np.testing.assert_allclose(np.sort(y_values[x_values == 100]), [0.6, 0.8])
        np.testing.assert_allclose(np.sort(y_values[x_values == 200]), [0.7, 0.9])
    finally:
        plt.close(figure)


def test_plot_over_training_size_averages_repeated_instance_ids_separately() -> None:
    data = pd.DataFrame(
        [
            {
                "pipeline_mlflow_run_id": pipeline_run,
                "scope": "test",
                "dataset": "tudd",
                "model_name": "xgboost",
                "model_instance": instance,
                "training_size": 100,
                "roc_auc": score,
            }
            for pipeline_run, instance, score in (
                ("run-a", "xgboost__0", 0.6),
                ("run-a", "xgboost__1", 0.8),
                ("run-b", "xgboost__0", 0.8),
                ("run-b", "xgboost__1", 1.0),
            )
        ]
    )

    figure = plot_over_training_size(
        data,
        datasets=("tudd",),
        run_aggregation="average",
        log_x=False,
        show_ci=False,
    )

    try:
        values_by_instance = {line.get_label(): line.get_ydata().tolist() for line in figure.axes[0].lines}
        assert values_by_instance == {"xgboost__0": [0.7], "xgboost__1": [0.9]}
    finally:
        plt.close(figure)


def _model_setting_data() -> pd.DataFrame:
    rows = []
    scores = {
        "xgboost": ((0.70, 10.0), (0.75, 8.0)),
        "logistic-regression": ((0.60, 2.0), (0.65, 1.0)),
    }
    for model, settings in scores.items():
        for dataset in ("mimic", "tudd"):
            for score, runtime in settings:
                dataset_score = score + (0.1 if dataset == "mimic" else 0.0)
                rows.append(
                    {
                        "scope": "test",
                        "statistic": "point",
                        "dataset": dataset,
                        "model_name": model,
                        "roc_auc": dataset_score,
                        "roc_auc_ci_lower": dataset_score - 0.02,
                        "roc_auc_ci_upper": dataset_score + 0.03,
                        "total_time": runtime,
                    }
                )
        rows.append(
            {
                "scope": "test",
                "statistic": "delta",
                "dataset": "tudd",
                "model_name": model,
                "roc_auc": -0.1,
                "total_time": 99.0,
            }
        )
    return pd.DataFrame(rows)


def test_model_setting_bars_filter_dataset_and_use_canonical_order_and_default_labels() -> None:
    figure = plot_model_setting_performance(_model_setting_data())

    try:
        assert isinstance(figure, Figure)
        axis = figure.axes[0]
        assert [tick.get_text() for tick in axis.get_xticklabels()] == ["LR", "XGBoost"]
        assert [text.get_text() for text in axis.get_legend().get_texts()] == ["Setting 1", "Setting 2"]
        assert axis.get_legend().get_title().get_text() == "Setting"

        # Patches are emitted setting-first in canonical model order. MIMIC and
        # delta rows have already been filtered and do not become settings.
        assert len(axis.patches) == 4
        np.testing.assert_allclose([patch.get_height() for patch in axis.patches], [0.60, 0.70, 0.65, 0.75])
        first_centers = [patch.get_x() + patch.get_width() / 2 for patch in axis.patches[:2]]
        second_centers = [patch.get_x() + patch.get_width() / 2 for patch in axis.patches[2:]]
        assert first_centers[0] < second_centers[0]
        assert first_centers[1] < second_centers[1]
        assert len(axis.collections) == 2  # one CI error-bar collection per setting
    finally:
        plt.close(figure)


def test_model_setting_bars_use_caller_labels_titles_and_model_filters() -> None:
    figure = plot_model_setting_performance(
        _model_setting_data(),
        include_models=["xgboost"],
        setting_labels=["Base", "Tuned"],
        show_ci=False,
        title="Comparison",
        legend_title="Configuration",
    )

    try:
        axis = figure.axes[0]
        assert [tick.get_text() for tick in axis.get_xticklabels()] == ["XGBoost"]
        assert [text.get_text() for text in axis.get_legend().get_texts()] == ["Base", "Tuned"]
        assert axis.get_legend().get_title().get_text() == "Configuration"
        assert axis.get_title() == "Comparison"
        assert not axis.collections
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda data: data.drop(data[(data["dataset"] == "tudd") & (data["model_name"] == "xgboost")].index[0]), "same number"),
        (lambda data: data, "Expected 2 setting labels"),
    ],
)
def test_model_setting_bars_validate_occurrence_and_label_counts(mutator, message: str) -> None:
    data = mutator(_model_setting_data())
    labels = None if "same number" in message else ["Only one"]

    with pytest.raises(ValueError, match=message):
        plot_model_setting_performance(data, setting_labels=labels)


def test_model_setting_runtime_scatter_plots_values_inverts_axis_and_annotates_models() -> None:
    figure = plot_model_setting_performance_vs_runtime(
        _model_setting_data(),
        setting_labels=["Base", "Tuned"],
        legend_title="Setup",
        title="Runtime tradeoff",
    )

    try:
        assert isinstance(figure, Figure)
        axis = figure.axes[0]
        assert axis.get_xscale() == "log"
        assert axis.get_xlim()[0] > axis.get_xlim()[1]
        offsets = np.vstack([collection.get_offsets() for collection in axis.collections if len(collection.get_offsets())])
        assert {tuple(point) for point in offsets} >= {(2.0, 0.60), (1.0, 0.65), (10.0, 0.70), (8.0, 0.75)}
        assert [text.get_text() for text in axis.texts] == ["LR", "LR", "XGBoost", "XGBoost"]
        assert [text.get_text() for text in axis.get_legend().get_texts()] == ["Base", "Tuned"]
        assert axis.get_legend().get_title().get_text() == "Setup"
        assert axis.get_title() == "Runtime tradeoff"
        assert axis.get_xlabel() == "Model total time (seconds, log scale)"
    finally:
        plt.close(figure)


def test_model_setting_runtime_scatter_supports_linear_x_and_rejects_nonpositive_log_runtime() -> None:
    data = _model_setting_data()
    data.loc[(data["dataset"] == "tudd") & (data["statistic"] == "point"), "total_time"] -= 2
    figure = plot_model_setting_performance_vs_runtime(data, log_x=False, show_ci=False)

    try:
        axis = figure.axes[0]
        assert axis.get_xscale() == "linear"
        assert axis.get_xlim()[0] > axis.get_xlim()[1]
        assert axis.get_xlabel() == "Model total time (seconds)"
    finally:
        plt.close(figure)

    with pytest.raises(ValueError, match="strictly positive"):
        plot_model_setting_performance_vs_runtime(data, log_x=True)
