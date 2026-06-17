from typing import Literal

from src.schemas.base_schemas import StrictParams


class PlottingParams(StrictParams):
    enabled: bool = True
    formats: tuple[Literal["png", "pdf", "svg"], ...] = ("png",)
