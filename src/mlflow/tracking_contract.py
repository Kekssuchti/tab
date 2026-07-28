"""Shared names used when writing and reading the MLflow run hierarchy."""

import re

TRACKING_SCHEMA_VERSION = "1"

TAG_RUN_TYPE = "run_type"
TAG_TRACKING_SCHEMA_VERSION = "tracking_schema_version"
TAG_PIPELINE_ID = "pipeline_id"
TAG_PIPELINE_MLFLOW_RUN_ID = "pipeline_mlflow_run_id"
TAG_MODEL_MLFLOW_RUN_ID = "model_mlflow_run_id"
TAG_MODEL_NAME = "model_name"
TAG_MODEL_INSTANCE = "model_instance"
TAG_TARGET = "target"
TAG_TASK_TYPE = "task_type"
TAG_STATUS = "status"
TAG_TRAINED_ON = "trained_on"
TAG_TRAIN_SOURCES = "train_sources"

RUN_TYPE_PIPELINE = "pipeline"
RUN_TYPE_MODEL = "model"
RUN_TYPE_CV_CANDIDATE = "cv_candidate"

STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"

ARTIFACT_CONFIG = "config.json"
ARTIFACT_PIPELINE_RESULT = "pipeline_result.json"
ARTIFACT_ENVIRONMENT = "environment.json"
ARTIFACT_MANIFEST = "tracking_manifest.json"
ARTIFACT_EVALUATION_TABLE = "evaluation_metrics.json"
ARTIFACT_CV_RESULTS = "cv_results"

TEST_DATASETS = ("mimic", "tudd")
TEST_DELTA_DATASET = "mimic_minus_tudd"

PARAM_DATASET_TARGET = "dataset.target"
PARAM_MODEL_TUNED = "model.tuned"
PARAM_MODEL_BEST_PARAMS = "model.tuning.best_params"

METRIC_PIPELINE_TOTAL_TIME = "pipeline.total_time"
METRIC_TRAIN_FIT_TIME = "train.fit_time"
METRIC_MODEL_TOTAL_TIME = "model.total_time"
METRIC_CV_TOTAL_TIME = "cv.total_time"

_TEST_SCORE_CI_PATTERN = re.compile(r"^ci_95_(?P<metric>.+)_(?:lower|upper)$")
_TEST_DELTA_PREFIX = f"test.{TEST_DELTA_DATASET}."


def dataset_row_count_param(dataset_part: str) -> str:
    return f"dataset.{dataset_part}.row_count"


def test_score_metric(dataset: str, metric: str) -> str:
    return f"test.{dataset}.{metric}"


def test_mean_score_metric(dataset: str, metric: str) -> str:
    return f"test.{dataset}.mean_{metric}"


def test_score_ci_metric(dataset: str, metric: str, bound: str) -> str:
    return f"test.{dataset}.ci_95_{metric}_{bound}"


def test_delta_metric(metric: str) -> str:
    return f"{_TEST_DELTA_PREFIX}{metric}"


def test_predict_time_metric(dataset: str) -> str:
    return f"test.{dataset}.predict_time"


def test_n_classes_param(dataset: str) -> str:
    return f"test.{dataset}.n_classes"


def parse_test_score_metric(name: str, dataset: str) -> str | None:
    prefix = f"test.{dataset}."
    if not name.startswith(prefix):
        return None
    suffix = name.removeprefix(prefix)
    if suffix == "predict_time":
        return None
    if suffix.startswith("mean_"):
        return suffix.removeprefix("mean_")
    ci_match = _TEST_SCORE_CI_PATTERN.fullmatch(suffix)
    return ci_match.group("metric") if ci_match else suffix


def parse_test_delta_metric(name: str) -> str | None:
    if not name.startswith(_TEST_DELTA_PREFIX):
        return None
    return name.removeprefix(_TEST_DELTA_PREFIX)
