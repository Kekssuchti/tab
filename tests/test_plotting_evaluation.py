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
from src.plotting.plot_utils import format_model_setting_mapping


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
        "xgboost": ((0.70, 10.0, "baseline-b"), (0.75, 8.0, "tuned-run")),
        "logistic-regression": ((0.60, 2.0, "baseline-a"), (0.65, 1.0, "tuned-run")),
    }
    for model, settings in scores.items():
        for dataset in ("mimic", "tudd"):
            for score, runtime, pipeline_run_name in settings:
                dataset_score = score + (0.1 if dataset == "mimic" else 0.0)
                rows.append(
                    {
                        "scope": "test",
                        "statistic": "point",
                        "dataset": dataset,
                        "model_name": model,
                        "pipeline_run_name": pipeline_run_name if dataset == "tudd" else "excluded-dataset-run",
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
                "pipeline_run_name": "excluded-statistic-run",
                "roc_auc": -0.1,
                "total_time": 99.0,
            }
        )
    return pd.DataFrame(rows)


def test_format_model_setting_mapping_uses_stable_runs_canonical_models_and_deduplicates() -> None:
    frame = pd.DataFrame(
        [
            {"setting_index": 0, "pipeline_run_name": "baseline-b", "model_name": "xgboost"},
            {"setting_index": 0, "pipeline_run_name": "baseline-b", "model_name": "xgboost"},
            {"setting_index": 0, "pipeline_run_name": "baseline-b", "model_name": "logistic-regression"},
            {"setting_index": 0, "pipeline_run_name": "baseline-a", "model_name": "tabpfn-3"},
            {"setting_index": 1, "pipeline_run_name": "tuned-run", "model_name": "xgboost"},
            {"setting_index": 1, "pipeline_run_name": "tuned-run", "model_name": "logistic-regression"},
        ]
    )

    assert format_model_setting_mapping(frame, ("Base", "Tuned")) == (
        "Model setting mapping:\n"
        "Base: baseline-b: LR, XGBoost; baseline-a: TabPFNv3\n"
        "Tuned: tuned-run: LR, XGBoost"
    )


def test_model_setting_bars_filter_dataset_and_use_canonical_order_and_default_labels(capsys) -> None:
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
        assert axis.get_ylim()[0] == pytest.approx(0.0)
        figure.canvas.draw()
        legend = axis.get_legend()
        legend_box = legend.get_window_extent()
        axis_box = axis.get_window_extent()
        title_box = axis.title.get_window_extent()
        assert legend_box.y0 >= axis_box.y1
        assert legend_box.y0 > title_box.y1
        assert legend_box.y1 <= figure.bbox.y1
        assert legend_box.x0 + legend_box.width / 2 == pytest.approx(axis_box.x0 + axis_box.width / 2, abs=2)
        assert legend._ncols == 2
        assert capsys.readouterr().out == (
            "Model setting mapping:\n"
            "Setting 1: baseline-a: LR; baseline-b: XGBoost\n"
            "Setting 2: tuned-run: LR, XGBoost\n"
        )
    finally:
        plt.close(figure)


def test_model_setting_bars_support_auto_and_explicit_y_limits() -> None:
    auto_figure = plot_model_setting_performance(_model_setting_data(), y_limits="auto")
    explicit_figure = plot_model_setting_performance(_model_setting_data(), y_limits=(0.55, 0.82))

    try:
        auto_limits = auto_figure.axes[0].get_ylim()
        assert 0.0 < auto_limits[0] < 0.58
        assert auto_limits[1] > 0.78
        assert explicit_figure.axes[0].get_ylim() == pytest.approx((0.55, 0.82))
    finally:
        plt.close(auto_figure)
        plt.close(explicit_figure)


@pytest.mark.parametrize(
    "y_limits",
    ["automatic", (0.5,), (0.5, 0.7, 0.9), (0.5, np.inf), (np.nan, 0.8), (0.8, 0.8), (0.9, 0.8)],
)
def test_model_setting_bars_reject_invalid_y_limits(y_limits) -> None:
    with pytest.raises(ValueError, match="y_limits"):
        plot_model_setting_performance(_model_setting_data(), y_limits=y_limits)


def test_model_setting_bars_use_caller_labels_titles_and_model_filters(capsys) -> None:
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
        assert capsys.readouterr().out == (
            "Model setting mapping:\nBase: baseline-b: XGBoost\nTuned: tuned-run: XGBoost\n"
        )
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


def test_model_setting_plots_require_valid_pipeline_run_names_on_selected_rows() -> None:
    missing_column = _model_setting_data().drop(columns="pipeline_run_name")
    with pytest.raises(ValueError, match="Missing required evaluation columns: pipeline_run_name"):
        plot_model_setting_performance(missing_column)

    invalid_selected = _model_setting_data()
    selected = (invalid_selected["dataset"] == "tudd") & (invalid_selected["statistic"] == "point")
    invalid_selected.loc[selected.idxmax(), "pipeline_run_name"] = "  "
    with pytest.raises(ValueError, match="pipeline_run_name.*fix the selected evaluation rows"):
        plot_model_setting_performance_vs_runtime(invalid_selected)


def test_model_setting_runtime_scatter_plots_values_inverts_axis_and_annotates_models(capsys) -> None:
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
        assert all(annotation.arrow_patch is None for annotation in axis.texts)
        assert capsys.readouterr().out == (
            "Model setting mapping:\n"
            "Base: baseline-a: LR; baseline-b: XGBoost\n"
            "Tuned: tuned-run: LR, XGBoost\n"
        )
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


def test_model_setting_runtime_scatter_keeps_plain_annotations_when_points_collide() -> None:
    data = _model_setting_data()
    selected = (data["dataset"] == "tudd") & (data["statistic"] == "point")
    data.loc[selected, "total_time"] = 5.0
    data.loc[selected, "roc_auc"] = 0.7

    figure = plot_model_setting_performance_vs_runtime(data, show_ci=False)

    try:
        annotations = figure.axes[0].texts
        positions = [annotation.get_position() for annotation in annotations]
        assert positions == [(5, 4)] * len(annotations)
        assert all(annotation.arrow_patch is None for annotation in annotations)
        assert [annotation.get_text() for annotation in annotations] == ["LR", "LR", "XGBoost", "XGBoost"]
    finally:
        plt.close(figure)
