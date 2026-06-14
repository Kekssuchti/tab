import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    seed: int = Field(default=1337, alias="SEED")

    # paths
    root_dir: Path = Path(__file__).parents[1]
    data_dir: Path = root_dir / "data"
    cache_dir: Path = root_dir / "cache"

    src_dir: Path = root_dir / "src"
    mlflow_dir: Path = src_dir / "mlflow"
    log_dir: Path = src_dir / "logs"
    evaluation_dir: Path = src_dir / "evaluation"

    external_dep_dir: Path = root_dir / "external"
    external_limix_dir: Path = external_dep_dir / "limix"

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
