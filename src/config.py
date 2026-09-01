import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

# Keep os env stuff here that needs to ALWAYS be set
# MUST use for memory-efficient SDPA kernels on AMD GPUs.
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")


class Config(BaseSettings):
    """Environment-backed project settings and workspace paths."""

    seed: int = Field(default=1337, alias="SEED")

    # paths
    dir_root: Path = Path(__file__).parents[1]
    dir_cache: Path = dir_root / "cache"
    dir_plots: Path = dir_root / "plots"
    dir_log: Path = dir_root / "logs"
    dir_mlflow_artifacts: Path = dir_root / "mlartifacts"

    dir_configs: Path = dir_root / "configs"
    dir_pipelines: Path = dir_configs / "pipeline"
    dir_suites: Path = dir_configs / "suite"

    dir_data: Path = dir_root / "data"
    dir_data_toy: Path = dir_data / "toy"

    dir_external_limix: Path = dir_root / "external" / "limix"
    dir_external_tabfm: Path = dir_root / "external" / "tabfm"


@lru_cache
def get_config():
    return Config()


config = get_config()
