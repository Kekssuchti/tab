from typing import Literal

from pydantic import BaseModel, ConfigDict


class StrictParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


TaskType = Literal["classification", "regression"]
