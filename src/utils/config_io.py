from pathlib import Path
from typing import Any

import yaml

from src.schemas.pipeline_schemas import PipelineConfig
from src.schemas.suite_schemas import ExperimentSuiteConfig


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    config_data = _load_yaml(path)
    return PipelineConfig.model_validate(config_data)


def load_experiment_suite_config(path: str | Path) -> ExperimentSuiteConfig:
    config_data = _load_yaml(path)
    return ExperimentSuiteConfig.model_validate(config_data)


def dump_pipeline_config(pipeline_config: PipelineConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            pipeline_config.model_dump(mode="json"),
            file,
            sort_keys=False,
        )


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)

    if not isinstance(config_data, dict):
        raise ValueError(f"Expected YAML object at {path}")

    return config_data
