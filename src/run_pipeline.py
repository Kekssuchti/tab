import argparse
from pathlib import Path

from src.classes.experiment_suite import ExperimentSuite
from src.classes.pipeline import Pipeline
from src.config import config
from src.mlflow.mlflow_logger import MLflowPipelineLogger
from src.schemas.pipeline_schemas import PipelineConfig
from src.schemas.suite_schemas import SuiteRunResult
from src.utils.config_io import (
    load_experiment_suite_config,
    load_pipeline_config,
)
from src.utils.logger import logger
from src.utils.prediction_metrics import evaluate_classification_models


def run_pipeline(config_path: str | Path):
    logger.info("Start Pipeline run")
    config_path = Path(config_path)
    pipeline_config = load_pipeline_config(config_path)

    result = run_pipeline_params(pipeline_config, config_path=config_path)
    logger.info(f"Pipeline run completed. Results: {result}")
    return result


def run_pipeline_params(
    pipeline_config: PipelineConfig,
    *,
    config_path: str | Path | None = None,
):
    mlflow_logger = MLflowPipelineLogger(config_path) if pipeline_config.mlflow.enabled else None
    pipeline = Pipeline(pipeline_config)

    def log_completed_model(partial_result, model_run):
        if mlflow_logger is None:
            return
        mlflow_logger.log_model_run(
            pipeline_config,
            partial_result,
            model_run,
            pipeline.prediction_tables,
        )

    result = pipeline.run(on_model_complete=log_completed_model)

    if mlflow_logger is not None:
        classification_evaluation = None
        if pipeline.prediction_tables:
            try:
                classification_evaluation = evaluate_classification_models(
                    pipeline.prediction_tables.frames(),
                    random_state=pipeline_config.random_states.evaluation_bootstrap_seed,
                )
            except Exception:  # noqa: BLE001 - the notebook can recover from the saved predictions
                logger.exception("Post-pipeline metric calculation failed; prediction CSVs remain available")
        try:
            mlflow_logger.log_pipeline_summary(
                pipeline_config,
                result,
                pipeline.prediction_tables,
                classification_evaluation,
            )
        except Exception:  # noqa: BLE001 - tracking must not invalidate completed model work
            logger.exception("Final MLflow logging failed; returning completed pipeline result")

    return result


def run_suite(config_path: str | Path, *, dry_run: bool = False):
    logger.info("Start experiment suite run")
    config_path = Path(config_path)
    suite_params = load_experiment_suite_config(config_path)
    suite = ExperimentSuite(suite_params, config_path)
    summary = suite.dry_run_summary()

    if dry_run:
        print(summary.format())
        return summary
    else:
        logger.info(summary)

    results = []
    for variant in summary.config_variants:
        logger.info(f"Running variant {variant.variant_id}")
        results.append(run_pipeline_params(variant.pipeline_config))

    suite_result = SuiteRunResult(
        suite_name=suite_params.name,
        results=tuple(results),
        summary=summary,
    )
    logger.info(f"Experiment suite run completed. Results: {suite_result}")
    return suite_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name",
        "-c",
        type=str,
        help="Name of the pipeline or suite config file without .yaml",
    )
    parser.add_argument("--suite", action="store_true", help="Run a suite config")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show generated suite configs without executing them",
    )
    args = parser.parse_args()

    if args.dry_run and not args.suite:
        parser.error("--dry-run is only supported with --suite")

    config_dir = config.dir_suites if args.suite else config.dir_pipelines
    path = Path(config_dir) / f"{args.config_name}.yaml"
    if args.suite:
        run_suite(path, dry_run=args.dry_run)
    else:
        run_pipeline(path)
