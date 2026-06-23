from pathlib import Path

from src.classes.pipeline import Pipeline
from src.config import config
from src.mlflow.mlflow_logger import MLflowPipelineLogger
from src.utils.config_io import load_pipeline_params
from src.utils.logger import logger


def run_pipeline(config_path: str | Path):
    logger.info("Start Pipeline run")
    config_path = Path(config_path)
    params = load_pipeline_params(config_path)
    result = Pipeline(params).run()

    if params.mlflow.enabled:
        MLflowPipelineLogger().log_pipeline_run(
            params,
            result,
            config_path=config_path,
        )

    logger.info(f"Pipeline run completed. Results: {result}")
    return result


if __name__ == "__main__":
    run_pipeline(Path(config.dir_configs) / "pipeline_tabpfn.yaml")
