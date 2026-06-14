import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    seed: int = Field(default=1337, alias="SEED")

    # paths
    dir_root: Path = Path(__file__).parents[1]
    dir_cache: Path = dir_root / "cache"

    dir_data: Path = dir_root / "data"
    dir_data_toy: Path = dir_data / "toy"
    dir_mimic: Path = dir_data / "mimic"
    dir_europe: Path = dir_data / "europe"

    dir_src: Path = dir_root / "src"
    dir_mlflow: Path = dir_src / "mlflow"
    dir_log: Path = dir_src / "logs"
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
