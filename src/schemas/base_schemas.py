from typing import Literal

from pydantic import BaseModel, ConfigDict


class StrictConfig(BaseModel):
    """Pydantic base class that rejects unknown configuration fields."""

    model_config = ConfigDict(extra="forbid")


TaskType = Literal["classification", "regression"]
