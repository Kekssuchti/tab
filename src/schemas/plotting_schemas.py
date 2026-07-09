from typing import Literal

from src.schemas.base_schemas import StrictConfig


class PlottingConfig(StrictConfig):
    """
    Configuration for optional plot generation.

    ---
    Attributes:
        enabled: bool, default=True
            Whether plotting is enabled.

        formats: tuple of {"png", "pdf", "svg"}, default=("png",)
            Output formats for generated plots.
    """

    enabled: bool = True
    formats: tuple[Literal["png", "pdf", "svg"], ...] = ("png",)
