"""Names shared by the compact MLflow writer and readers."""

TRACKING_SCHEMA_VERSION = "1"

TAG_RUN_TYPE = "run_type"
TAG_TRACKING_SCHEMA_VERSION = "tracking_schema_version"
TAG_PIPELINE_ID = "pipeline_id"
TAG_PIPELINE_MLFLOW_RUN_ID = "pipeline_mlflow_run_id"
TAG_MODEL_NAME = "model_name"
TAG_MODEL_INSTANCE = "model_instance"
TAG_MODEL_INSTANCES = "model_instances"
TAG_TARGET = "target"
TAG_TASK_TYPE = "task_type"
TAG_STATUS = "status"
TAG_TRAINED_ON = "trained_on"
TAG_TRAIN_SOURCES = "train_sources"
TAG_TRAINING_SIZE = "training_size"

RUN_TYPE_PIPELINE = "pipeline"
RUN_TYPE_MODEL = "model"
STATUS_SUCCESS = "success"

ARTIFACT_ACTIVE_LOG = "active.log"
ARTIFACT_CONFIG = "config.json"
ARTIFACT_CONFIG_YAML = "config.yaml"
ARTIFACT_CV_RESULTS = "cv_results"
ARTIFACT_DATASET_SUMMARY = "dataset_summary.json"
ARTIFACT_ENVIRONMENT = "environment.json"
ARTIFACT_TEST_PREDICTIONS = "test_predictions"
ARTIFACT_CLASSIFICATION_METRICS = "prediction_metrics.csv"
ARTIFACT_BOOTSTRAP_METRICS = "bootstrap_metrics.csv"
ARTIFACT_PAIRWISE_WINS = "pairwise_wins"

METRIC_TRAIN_CV_TIME = "train.cv_time"
METRIC_TRAIN_FIT_TIME = "train.fit_time"


def test_score_metric(dataset: str, metric: str) -> str:
    return f"test.{dataset}.{metric}"


def test_predict_time_metric(dataset: str) -> str:
    return f"test.{dataset}.predict_time"
