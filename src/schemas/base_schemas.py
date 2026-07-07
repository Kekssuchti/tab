from typing import Literal

from pydantic import BaseModel, ConfigDict


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


TaskType = Literal["classification", "regression"]
