from __future__ import annotations

from dataclasses import astuple, dataclass, fields
from typing import Literal, Sequence

import pandas as pd

from mlflow import MlflowClient
from mlflow.entities import Experiment, Run, RunStatus
from src.mlflow.serialization import training_result_to_dict
from src.mlflow.tracking_contract import (
    METRIC_CV_TOTAL_TIME,
    METRIC_MODEL_TOTAL_TIME,
    METRIC_TRAIN_FIT_TIME,
    PARAM_DATASET_TARGET,
    RUN_TYPE_MODEL,
    RUN_TYPE_PIPELINE,
    STATUS_SUCCESS,
    TAG_MODEL_INSTANCE,
    TAG_MODEL_NAME,
    TAG_PIPELINE_ID,
    TAG_PIPELINE_MLFLOW_RUN_ID,
    TAG_RUN_TYPE,
    TAG_STATUS,
    TAG_TARGET,
    TAG_TRAIN_SOURCES,
    TAG_TRAINED_ON,
    TEST_DATASETS,
    TEST_DELTA_DATASET,
    dataset_row_count_param,
    parse_test_delta_metric,
    parse_test_score_metric,
    test_mean_score_metric,
    test_n_classes_param,
    test_predict_time_metric,
    test_score_ci_metric,
    test_score_metric,
)

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT_NAME = "tab"


@dataclass(frozen=True)
class _PipelineRun:
    experiment: Experiment
    run: Run


@dataclass(frozen=True)
class _PipelineRunRow:
    mlflow_run_id: str
    pipeline_id: str
    run_name: str
    experiment_name: str
    model_instances: tuple[str, ...]
    target: str | None
    trained_on: str | None
    train_sources: tuple[str, ...]


@dataclass(frozen=True)
class _ModelContext:
    pipeline_mlflow_run_id: str
    pipeline_id: str
    pipeline_run_name: str
    experiment_name: str
    model_mlflow_run_id: str
    model_name: str
    model_instance: str
    target: str | None
    trained_on: str | None
    train_sources: tuple[str, ...]
    training_size: int | None


@dataclass(frozen=True)
class _Measurement:
    pipeline_mlflow_run_id: str
    pipeline_id: str
    pipeline_run_name: str
    experiment_name: str
    model_mlflow_run_id: str
    model_name: str
    model_instance: str
    target: str | None
    trained_on: str | None
    train_sources: tuple[str, ...]
    training_size: int | None
    kind: Literal["score", "time"]
    scope: Literal["test", "test_delta", "train", "model", "cv"]
    dataset: str | None
    metric: str
    value: float
    statistic: Literal["point", "mean", "difference"]
    unit: Literal["score", "seconds"]
    ci_level: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    n_classes: int | None = None
    test_row_count: int | None = None


def list_pipeline_runs(
    experiment_names: str | Sequence[str] = DEFAULT_EXPERIMENT_NAME,
    *,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> pd.DataFrame:
    """List parent pipeline runs and their successful model instances."""

    client = MlflowClient(tracking_uri=tracking_uri)
    experiments = _get_experiments(client, experiment_names)
    pipeline_runs, model_runs = _get_pipeline_and_model_runs(client, experiments)
    models_by_parent = _group_models_by_parent(_successful_models(model_runs))
    rows = (
        _pipeline_run_row(
            pipeline_run,
            models_by_parent.get(pipeline_run.run.info.run_id, ()),
        )
        for pipeline_run in pipeline_runs
    )
    return _frame(rows, _PipelineRunRow)


def load_evaluation_data(
    experiment_names: str | Sequence[str] = DEFAULT_EXPERIMENT_NAME,
    *,
    pipeline_runs: str | Sequence[str] | None = None,
    models: str | Sequence[str] | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> pd.DataFrame:
    """Load scores, test differences, and timings into one tidy DataFrame."""

    client = MlflowClient(tracking_uri=tracking_uri)
    experiments = _get_experiments(client, experiment_names)
    all_pipeline_runs, all_model_runs = _get_pipeline_and_model_runs(
        client, experiments
    )
    selected_pipeline_runs = _select_pipeline_runs(all_pipeline_runs, pipeline_runs)
    selected_parent_ids = {
        pipeline_run.run.info.run_id for pipeline_run in selected_pipeline_runs
    }
    selected_models = _select_models(
        [
            model_run
            for model_run in _successful_models(all_model_runs)
            if model_run.data.tags.get(TAG_PIPELINE_MLFLOW_RUN_ID)
            in selected_parent_ids
        ],
        models,
    )
    models_by_parent = _group_models_by_parent(selected_models)

    measurements = []
    for pipeline_run in selected_pipeline_runs:
        for model_run in models_by_parent.get(pipeline_run.run.info.run_id, ()):
            measurements.extend(_model_measurements(pipeline_run, model_run))
    return _frame(measurements, _Measurement)


def _get_experiments(
    client: MlflowClient,
    experiment_names: str | Sequence[str],
) -> tuple[Experiment, ...]:
    names = (
        (experiment_names,)
        if isinstance(experiment_names, str)
        else tuple(dict.fromkeys(experiment_names))
    )
    if not names:
        raise ValueError("At least one MLflow experiment name is required")

    experiments = []
    missing = []
    for name in names:
        experiment = client.get_experiment_by_name(name)
        if experiment is None:
            missing.append(name)
        else:
            experiments.append(experiment)
    if missing:
        raise ValueError(f"MLflow experiments not found: {', '.join(missing)}")
    return tuple(experiments)


def _get_pipeline_and_model_runs(
    client: MlflowClient,
    experiments: tuple[Experiment, ...],
) -> tuple[list[_PipelineRun], list[Run]]:
    pipeline_runs = []
    model_runs = []
    for experiment in experiments:
        pipeline_runs.extend(
            _PipelineRun(experiment, run)
            for run in _search_runs(
                client,
                experiment.experiment_id,
                f"tags.{TAG_RUN_TYPE} = '{RUN_TYPE_PIPELINE}'",
            )
        )
        model_runs.extend(
            _search_runs(
                client,
                experiment.experiment_id,
                f"tags.{TAG_RUN_TYPE} = '{RUN_TYPE_MODEL}'",
            )
        )
    pipeline_runs.sort(key=lambda item: item.run.info.start_time or 0)
    model_runs.sort(key=lambda run: run.info.start_time or 0)
    return pipeline_runs, model_runs


def _search_runs(
    client: MlflowClient,
    experiment_id: str,
    filter_string: str,
) -> list[Run]:
    runs = []
    page_token = None
    while True:
        page = client.search_runs(
            [experiment_id],
            filter_string=filter_string,
            max_results=1000,
            page_token=page_token,
            order_by=["attributes.start_time ASC"],
        )
        runs.extend(page)
        page_token = getattr(page, "token", None)
        if page_token is None:
            return runs


def _successful_models(model_runs: list[Run]) -> list[Run]:
    finished = RunStatus.to_string(RunStatus.FINISHED)
    return [
        run
        for run in model_runs
        if run.data.tags.get(TAG_STATUS) == STATUS_SUCCESS
        and run.info.status == finished
    ]


def _select_pipeline_runs(
    pipeline_runs: list[_PipelineRun],
    selectors: str | Sequence[str] | None,
) -> list[_PipelineRun]:
    requested = _selectors(selectors)
    if requested is None:
        return pipeline_runs
    selected = [
        pipeline_run
        for pipeline_run in pipeline_runs
        if requested & _pipeline_selectors(pipeline_run.run)
    ]
    available = set().union(*(_pipeline_selectors(item.run) for item in selected))
    _raise_for_unmatched("pipeline runs", requested, available)
    return selected


def _select_models(
    model_runs: list[Run],
    selectors: str | Sequence[str] | None,
) -> list[Run]:
    requested = _selectors(selectors)
    if requested is None:
        return model_runs
    selected = [run for run in model_runs if requested & _model_selectors(run)]
    available = set().union(*(_model_selectors(run) for run in selected))
    _raise_for_unmatched("successful models", requested, available)
    return selected


def _pipeline_selectors(run: Run) -> set[str]:
    return _present(
        run.info.run_id,
        run.data.tags.get(TAG_PIPELINE_ID),
        run.data.tags.get("mlflow.runName"),
    )


def _model_selectors(run: Run) -> set[str]:
    return _present(
        run.info.run_id,
        run.data.tags.get(TAG_MODEL_NAME),
        run.data.tags.get(TAG_MODEL_INSTANCE),
    )


def _raise_for_unmatched(
    label: str,
    requested: set[str],
    available: set[str],
) -> None:
    unmatched = sorted(requested - available)
    if unmatched:
        raise ValueError(f"No matching {label} for: {', '.join(unmatched)}")


def _pipeline_run_row(
    pipeline_run: _PipelineRun,
    model_runs: tuple[Run, ...],
) -> _PipelineRunRow:
    run = pipeline_run.run
    return _PipelineRunRow(
        mlflow_run_id=run.info.run_id,
        pipeline_id=_tag(run, TAG_PIPELINE_ID),
        run_name=_tag(run, "mlflow.runName"),
        experiment_name=pipeline_run.experiment.name,
        model_instances=tuple(_tag(model, TAG_MODEL_INSTANCE) for model in model_runs),
        target=run.data.tags.get(TAG_TARGET)
        or run.data.params.get(PARAM_DATASET_TARGET),
        trained_on=run.data.tags.get(TAG_TRAINED_ON),
        train_sources=_csv_tag(run, TAG_TRAIN_SOURCES),
    )


def _model_measurements(
    pipeline_run: _PipelineRun,
    model_run: Run,
) -> list[_Measurement]:
    parent = pipeline_run.run
    context = _ModelContext(
        pipeline_mlflow_run_id=parent.info.run_id,
        pipeline_id=_tag(parent, TAG_PIPELINE_ID),
        pipeline_run_name=_tag(parent, "mlflow.runName"),
        experiment_name=pipeline_run.experiment.name,
        model_mlflow_run_id=model_run.info.run_id,
        model_name=_tag(model_run, TAG_MODEL_NAME),
        model_instance=_tag(model_run, TAG_MODEL_INSTANCE),
        target=parent.data.tags.get(TAG_TARGET)
        or parent.data.params.get(PARAM_DATASET_TARGET),
        trained_on=model_run.data.tags.get(TAG_TRAINED_ON),
        train_sources=_csv_tag(model_run, TAG_TRAIN_SOURCES),
        training_size=_training_size(parent),
    )
    measurements = []
    for dataset in TEST_DATASETS:
        measurements.extend(_test_scores(context, parent, model_run, dataset))
    measurements.extend(_test_differences(context, model_run))
    measurements.extend(_times(context, parent, model_run))
    return measurements


def _test_scores(
    context: _ModelContext,
    parent: Run,
    run: Run,
    dataset: str,
) -> list[_Measurement]:
    metric_names = {
        metric
        for name in run.data.metrics
        if (metric := parse_test_score_metric(name, dataset)) is not None
    }
    measurements = []
    for metric in sorted(metric_names):
        mean_value = run.data.metrics.get(test_mean_score_metric(dataset, metric))
        value = mean_value
        if value is None:
            value = run.data.metrics.get(test_score_metric(dataset, metric))
        if value is None:
            continue
        lower = run.data.metrics.get(test_score_ci_metric(dataset, metric, "lower"))
        upper = run.data.metrics.get(test_score_ci_metric(dataset, metric, "upper"))
        measurements.append(
            _measurement(
                context,
                kind="score",
                scope="test",
                dataset=dataset,
                metric=metric,
                value=value,
                statistic="mean" if mean_value is not None else "point",
                unit="score",
                ci_level=0.95 if lower is not None and upper is not None else None,
                ci_lower=lower,
                ci_upper=upper,
                n_classes=_integer_param(run, test_n_classes_param(dataset)),
                test_row_count=_test_row_count(parent, dataset),
            )
        )
    return measurements


def _test_differences(
    context: _ModelContext,
    run: Run,
) -> list[_Measurement]:
    return [
        _measurement(
            context,
            kind="score",
            scope="test_delta",
            dataset=TEST_DELTA_DATASET,
            metric=metric,
            value=value,
            statistic="difference",
            unit="score",
        )
        for name, value in run.data.metrics.items()
        if (metric := parse_test_delta_metric(name)) is not None
    ]


def _times(
    context: _ModelContext,
    parent: Run,
    run: Run,
) -> list[_Measurement]:
    specs = (
        (METRIC_TRAIN_FIT_TIME, "train", None, "fit_time"),
        (test_predict_time_metric("mimic"), "test", "mimic", "predict_time"),
        (test_predict_time_metric("tudd"), "test", "tudd", "predict_time"),
        (METRIC_MODEL_TOTAL_TIME, "model", None, "total_time"),
        (METRIC_CV_TOTAL_TIME, "cv", None, "total_time"),
    )
    return [
        _measurement(
            context,
            kind="time",
            scope=scope,
            dataset=dataset,
            metric=metric,
            value=run.data.metrics[mlflow_name],
            statistic="point",
            unit="seconds",
            n_classes=_integer_param(run, test_n_classes_param(dataset))
            if dataset is not None
            else None,
            test_row_count=_test_row_count(parent, dataset)
            if dataset is not None
            else None,
        )
        for mlflow_name, scope, dataset, metric in specs
        if mlflow_name in run.data.metrics
    ]


def _measurement(
    context: _ModelContext,
    *,
    kind: Literal["score", "time"],
    scope: Literal["test", "test_delta", "train", "model", "cv"],
    dataset: str | None,
    metric: str,
    value: float,
    statistic: Literal["point", "mean", "difference"],
    unit: Literal["score", "seconds"],
    ci_level: float | None = None,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
    n_classes: int | None = None,
    test_row_count: int | None = None,
) -> _Measurement:
    return _Measurement(
        *astuple(context),
        kind=kind,
        scope=scope,
        dataset=dataset,
        metric=metric,
        value=value,
        statistic=statistic,
        unit=unit,
        ci_level=ci_level,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_classes=n_classes,
        test_row_count=test_row_count,
    )


def _group_models_by_parent(model_runs: list[Run]) -> dict[str, tuple[Run, ...]]:
    grouped = {}
    for model_run in model_runs:
        parent_id = model_run.data.tags.get(TAG_PIPELINE_MLFLOW_RUN_ID)
        if parent_id is not None:
            grouped.setdefault(parent_id, []).append(model_run)
    return {key: tuple(value) for key, value in grouped.items()}


def _frame(records, record_type: type) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        (astuple(record) for record in records),
        columns=[field.name for field in fields(record_type)],
    )


def _selectors(values: str | Sequence[str] | None) -> set[str] | None:
    if values is None:
        return None
    return {values} if isinstance(values, str) else set(values)


def _present(*values: str | None) -> set[str]:
    return {value for value in values if value is not None}


def _tag(run: Run, name: str) -> str:
    value = run.data.tags.get(name)
    if value is None:
        raise ValueError(
            f"MLflow run {run.info.run_id} is missing required tag {name!r}"
        )
    return value


def _csv_tag(run: Run, name: str) -> tuple[str, ...]:
    value = run.data.tags.get(name)
    return tuple(value.split(",")) if value else ()


def _test_row_count(run: Run, dataset: str) -> int | None:
    return _integer_param(run, dataset_row_count_param(f"test.{dataset}"))


def _training_size(run: Run) -> int | None:
    name = dataset_row_count_param("train")
    value = _integer_param(run, name)
    return value


def _integer_param(run: Run, name: str) -> int | None:
    value = run.data.params.get(name)
    return int(value) if value is not None else None
