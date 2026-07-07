from typing import Literal

from src.schemas.base_schemas import StrictConfig


class PlottingConfig(StrictConfig):
    enabled: bool = True
    formats: tuple[Literal["png", "pdf", "svg"], ...] = ("png",)
