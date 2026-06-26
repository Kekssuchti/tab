import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from src.classes.experiment_suite import ExperimentSuite
from src.classes.pipeline import Pipeline
from src.config import config
from src.mlflow.mlflow_logger import MLflowPipelineLogger
from src.schemas.pipeline_schemas import PipelineParams
from src.schemas.suite_schemas import SuiteRunResult
from src.utils.config_io import (
    dump_pipeline_params,
    load_experiment_suite_params,
    load_pipeline_params,
)
from src.utils.logger import logger


def run_pipeline(config_path: str | Path):
    logger.info("Start Pipeline run")
    config_path = Path(config_path)
    params = load_pipeline_params(config_path)

    result = run_pipeline_params(params, config_path=config_path)
    logger.info(f"Pipeline run completed. Results: {result}")
    return result


def run_pipeline_params(
    params: PipelineParams,
    config_path: str | Path | None = None,
):
    result = Pipeline(params).run()

    if params.mlflow.enabled:
        MLflowPipelineLogger().log_pipeline_run(
            params,
            result,
            config_path=Path(config_path) if config_path is not None else None,
        )

    return result


def run_suite(config_path: str | Path, *, dry_run: bool = False):
    logger.info("Start experiment suite run")
    config_path = Path(config_path)
    suite_params = load_experiment_suite_params(config_path)
    suite = ExperimentSuite(suite_params, config_path)
    summary = suite.dry_run_summary()

    if dry_run:
        print(summary.format())
        return summary
    else:
        logger.info(summary)

    results = []
    with TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        for variant in summary.variants:
            logger.info(f"Running variant {variant.variant_id}")
            concrete_config_path = temp_dir / f"{variant.variant_id}.yaml"
            dump_pipeline_params(variant.params, concrete_config_path)
            results.append(
                run_pipeline_params(
                    variant.params,
                    config_path=concrete_config_path,
                )
            )

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
        "config_name",
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
