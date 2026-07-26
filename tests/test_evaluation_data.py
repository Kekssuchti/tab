from collections.abc import Iterable

import mlflow
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.plot_results import (
    list_pipeline_runs,
    load_evaluation_data,
    plot_generalization_gaps,
    plot_performance_vs_runtime,
    plot_roc_auc,
)
from src.utils.evaluation_plot import calculate_comparative_generalizability


def _log_pipeline(
    tracking_uri: str,
    *,
    experiment_name: str,
    run_name: str,
    pipeline_id: str,
    models: Iterable[tuple[str, str, bool, bool]],
) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(
        run_name=run_name,
        tags={
            "run_type": "pipeline",
            "pipeline_id": pipeline_id,
            "target": "mortality",
            "task_type": "classification",
            "trained_on": "mimic",
            "train_sources": "mimic",
        },
    ) as parent:
        mlflow.log_params(
            {
                "dataset.target": "mortality",
                "dataset.train.row_count": 100,
                "dataset.test.mimic.row_count": 20,
                "dataset.test.tudd.row_count": 30,
            }
        )
        mlflow.log_metric("pipeline.total_time", 10.0)
        parent_id = parent.info.run_id

        for model_name, model_instance, succeeded, tuned in models:
            with mlflow.start_run(
                run_name=model_instance,
                nested=True,
                tags={
                    "run_type": "model",
                    "pipeline_id": pipeline_id,
                    "pipeline_mlflow_run_id": parent_id,
                    "model_name": model_name,
                    "model_instance": model_instance,
                    "model_mlflow_run_id": "set-after-start",
                    "task_type": "classification",
                    "status": "success" if succeeded else "failed",
                    "trained_on": "mimic",
                    "train_sources": "mimic",
                },
            ) as model_run:
                mlflow.set_tag("model_mlflow_run_id", model_run.info.run_id)
                mlflow.log_param("model.tuned", tuned)
                mlflow.log_metric("train.fit_time", 2.0)
                if not succeeded:
                    continue

                if tuned:
                    mlflow.log_param("model.tuning.best_params", '{"C": 1.0}')
                    mlflow.log_metrics(
                        {
                            "test.mimic.mean_accuracy": 0.9,
                            "test.mimic.mean_roc_auc": 0.8,
                            "test.mimic.ci_95_accuracy_lower": 0.85,
                            "test.mimic.ci_95_accuracy_upper": 0.95,
                            "test.tudd.mean_accuracy": 0.7,
                            "test.tudd.mean_roc_auc": 0.75,
                            "test.tudd.ci_95_accuracy_lower": 0.65,
                            "test.tudd.ci_95_accuracy_upper": 0.75,
                            "cv.total_time": 4.0,
                        }
                    )
                else:
                    mlflow.log_metrics(
                        {
                            "test.mimic.accuracy": 0.85,
                            "test.mimic.roc_auc": 0.78,
                            "test.tudd.accuracy": 0.68,
                            "test.tudd.roc_auc": 0.71,
                        }
                    )

                mlflow.log_params(
                    {
                        "test.mimic.n_classes": 2,
                        "test.tudd.n_classes": 2,
                    }
                )
                mlflow.log_metrics(
                    {
                        "test.mimic_minus_tudd.accuracy": 0.2,
                        "test.mimic_minus_tudd.roc_auc": 0.05,
                        "test.mimic.predict_time": 0.1,
                        "test.tudd.predict_time": 0.2,
                        "model.total_time": 2.3,
                    }
                )
    return parent_id


@pytest.fixture
def tracking_uri(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    _log_pipeline(
        uri,
        experiment_name="tab",
        run_name="alpha-run",
        pipeline_id="pipeline-alpha",
        models=(
            ("logistic-regression", "logistic-regression", True, True),
            ("xgboost", "xgboost", False, False),
        ),
    )
    _log_pipeline(
        uri,
        experiment_name="tab",
        run_name="beta-run",
        pipeline_id="pipeline-beta",
        models=(("xgboost", "xgboost", True, False),),
    )
    _log_pipeline(
        uri,
        experiment_name="other",
        run_name="gamma-run",
        pipeline_id="pipeline-gamma",
        models=(("ebm", "ebm", True, True),),
    )
    return uri


def test_lists_pipeline_runs_and_successful_models(tracking_uri):
    runs = list_pipeline_runs("tab", tracking_uri=tracking_uri)

    assert isinstance(runs, pd.DataFrame)
    assert list(runs["run_name"]) == ["alpha-run", "beta-run"]
    assert runs["mlflow_run_id"].notna().all()
    assert list(runs["model_instances"]) == [
        ("logistic-regression",),
        ("xgboost",),
    ]


def test_loads_multiple_experiments_into_plotting_tables(tracking_uri):
    data = load_evaluation_data(
        ("tab", "other"),
        tracking_uri=tracking_uri,
    )

    assert set(data["pipeline_run_name"]) == {
        "alpha-run",
        "beta-run",
        "gamma-run",
    }
    assert set(data["model_name"]) == {
        "logistic-regression",
        "xgboost",
        "ebm",
    }
    assert set(data["training_size"]) == {100}
    scores = data
    assert {"kind", "unit"}.isdisjoint(data.columns)
    assert set(scores["dataset"]) == {
        "mimic",
        "tudd",
        "mimic_minus_tudd",
    }
    assert {"metric", "value", "ci_lower", "ci_upper"}.isdisjoint(data.columns)
    assert {
        "accuracy",
        "accuracy_ci_lower",
        "accuracy_ci_upper",
        "roc_auc",
        "cv_time",
        "fit_time",
        "predict_time_mimic",
        "predict_time_tudd",
        "total_time",
        "generalizability_loss_roc_auc",
        "comparative_generalizability_loss_roc_auc",
    } <= set(data.columns)

    logistic_accuracy = scores.loc[
        (scores["model_name"] == "logistic-regression")
        & (scores["dataset"] == "mimic")
        & (scores["scope"] == "test")
    ].iloc[0]
    assert logistic_accuracy["accuracy"] == pytest.approx(0.9)
    assert logistic_accuracy["accuracy_ci_lower"] == pytest.approx(0.85)

    logistic_delta = scores.loc[
        (scores["model_name"] == "logistic-regression")
        & (scores["dataset"] == "mimic_minus_tudd")
    ].iloc[0]
    assert logistic_delta["accuracy"] == pytest.approx(0.2)

    assert logistic_accuracy["cv_time"] == pytest.approx(4.0)
    assert logistic_accuracy["fit_time"] == pytest.approx(2.0)
    assert logistic_accuracy["predict_time_mimic"] == pytest.approx(0.1)
    assert logistic_accuracy["predict_time_tudd"] == pytest.approx(0.2)
    assert logistic_accuracy["total_time"] == pytest.approx(6.3)

    xgboost_accuracy = scores.loc[
        (scores["model_name"] == "xgboost")
        & (scores["dataset"] == "mimic")
        & (scores["scope"] == "test")
    ].iloc[0]
    assert xgboost_accuracy["accuracy"] == pytest.approx(0.85)
    assert xgboost_accuracy["statistic"] == "point"
    assert pd.isna(xgboost_accuracy["ci_level"])


def test_filters_pipeline_runs_and_models_by_names(tracking_uri):
    data = load_evaluation_data(
        ("tab", "other"),
        pipeline_runs=("alpha-run", "gamma-run"),
        models=("logistic-regression", "ebm"),
        tracking_uri=tracking_uri,
    )

    assert set(data["pipeline_run_name"]) == {
        "alpha-run",
        "gamma-run",
    }
    assert set(data["model_name"]) == {
        "logistic-regression",
        "ebm",
    }

    logistic_only = load_evaluation_data(
        ("tab", "other"),
        pipeline_runs=("alpha-run", "gamma-run"),
        models="logistic-regression",
        tracking_uri=tracking_uri,
    )
    assert set(logistic_only["pipeline_run_name"]) == {"alpha-run"}

    untuned = load_evaluation_data(
        "tab",
        pipeline_runs="beta-run",
        tracking_uri=tracking_uri,
    )
    assert set(untuned["model_name"]) == {"xgboost"}
    untuned_accuracy = untuned.loc[
        (untuned["dataset"] == "mimic") & (untuned["scope"] == "test")
    ].iloc[0]
    assert pd.isna(untuned_accuracy["accuracy_ci_lower"])


def test_calculates_comparative_generalizability_on_external_test(tracking_uri):
    results = load_evaluation_data("tab", tracking_uri=tracking_uri)
    comparison = calculate_comparative_generalizability(results)
    by_model = comparison.set_index("model_name")

    assert set(comparison["external_dataset"]) == {"tudd"}
    assert by_model.loc["logistic-regression", "external_score"] == pytest.approx(0.75)
    assert by_model.loc[
        "logistic-regression", "comparative_generalizability_loss"
    ] == pytest.approx(0.0)
    assert by_model.loc["logistic-regression", "generalization_rank"] == 1
    assert by_model.loc[
        "xgboost", "comparative_generalizability_loss"
    ] == pytest.approx(-0.04)
    assert by_model.loc["xgboost", "generalization_rank"] == 2
    assert by_model.loc[
        "logistic-regression", "generalizability_loss"
    ] == pytest.approx(-0.05)

    logistic_external = results.loc[
        (results["model_name"] == "logistic-regression")
        & (results["scope"] == "test")
        & (results["dataset"] == "tudd")
    ].iloc[0]
    assert logistic_external["generalizability_loss_roc_auc"] == pytest.approx(-0.05)
    assert logistic_external[
        "comparative_generalizability_loss_roc_auc"
    ] == pytest.approx(0.0)


def test_plots_roc_auc_as_paired_test_centers(tracking_uri):
    results = load_evaluation_data("tab", tracking_uri=tracking_uri)

    ax = plot_roc_auc(results)

    assert len(ax.get_yticklabels()) == 2
    assert ax.get_xlabel() == "ROC AUC"
    assert {text.get_text() for text in ax.get_legend().get_texts()} == {
        "MIMIC",
        "TUDD",
    }
    plt.close(ax.figure)


def test_plots_selected_generalization_loss_with_ranks(tracking_uri):
    results = load_evaluation_data("tab", tracking_uri=tracking_uri)

    comparative_ax = plot_generalization_gaps(results)
    model_specific_ax = plot_generalization_gaps(results, loss="model_specific")

    assert [bar.get_width() for bar in comparative_ax.patches] == pytest.approx(
        [0.0, -0.04]
    )
    assert [bar.get_width() for bar in model_specific_ax.patches] == pytest.approx(
        [-0.05, -0.07]
    )
    assert {text.get_text() for text in comparative_ax.texts} == {"#1", "#2"}
    assert "Comparative generalizability loss" in comparative_ax.get_title()
    plt.close(comparative_ax.figure)
    plt.close(model_specific_ax.figure)


def test_plots_external_performance_against_model_runtime(tracking_uri):
    results = load_evaluation_data("tab", tracking_uri=tracking_uri)

    ax = plot_performance_vs_runtime(results)

    assert ax.get_xscale() == "log"
    assert ax.get_ylabel() == "External ROC AUC"
    assert "#1  logistic-regression" in ax.texts[-1].get_text()
    assert "#2  xgboost" in ax.texts[-1].get_text()
    plt.close(ax.figure)


def test_rejects_unknown_experiment_run_and_model_names(tracking_uri):
    with pytest.raises(ValueError, match="MLflow experiments not found: missing"):
        load_evaluation_data("missing", tracking_uri=tracking_uri)

    with pytest.raises(ValueError, match="No matching pipeline runs for: missing"):
        load_evaluation_data("tab", pipeline_runs="missing", tracking_uri=tracking_uri)

    with pytest.raises(ValueError, match="No matching successful models for: missing"):
        load_evaluation_data("tab", models="missing", tracking_uri=tracking_uri)
