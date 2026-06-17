from pydantic import BaseModel, ConfigDict


class StrictParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
