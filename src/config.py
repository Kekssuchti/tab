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
    dir_run_results: Path = dir_root / "run_results"
    dir_plots: Path = dir_root / "plots"
    dir_log: Path = dir_root / "logs"
    dir_mlflow_artifacts: Path = dir_root / "mlartifacts"

    dir_configs: Path = dir_root / "configs"
    dir_pipelines: Path = dir_configs / "pipeline"
    dir_suites: Path = dir_configs / "suite"

    dir_data: Path = dir_root / "data"
    dir_data_toy: Path = dir_data / "toy"

    dir_src: Path = dir_root / "src"
    dir_mlflow: Path = dir_src / "mlflow"
    dir_evaluation: Path = dir_src / "evaluation"
    dir_external_dep: Path = dir_root / "external"
    dir_external_limix: Path = dir_external_dep / "limix"

    tabpfn_token: str = Field(default="", alias="TABPFN_TOKEN")

    train_size: float = 0.8
    test_size: float = 0.2

    tfm_names: tuple[str, ...] = (
        "tabpfn-3",
        "tabicl-2",
        "limix-2m",
        "limix-16m",
        "mitra",
        "orion-bix",
        "orion-msp",
    )


@lru_cache
def get_config():
    return Config()


config = get_config()
