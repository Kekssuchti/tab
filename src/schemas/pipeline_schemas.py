from datetime import date
from pathlib import Path
from uuid import uuid4

from pydantic import Field

from src.config import config
from src.schemas.base_schemas import StrictParams
from src.schemas.dataset_schemas import DatasetParams
from src.schemas.plotting_schemas import PlottingParams
from src.schemas.training_schemas import ModelParams


def _default_run_id() -> str:
    return f"{date.today().isoformat()}_{uuid4().hex}"


class MLflowParams(StrictParams):
    enabled: bool = True
    tracking_uri: str = "sqlite:///mlflow.db"
    artifact_location: str | None = "mlartifacts"
    experiment_name: str = "tab"
    run_name: str | None = None


class PipelineParams(StrictParams):
    run_id: str = Field(default_factory=_default_run_id)
    dataset: DatasetParams = Field()
    training: tuple[ModelParams, ...] = (ModelParams(name="tabpfn-3"),)
    plotting: PlottingParams = Field()
    mlflow: MLflowParams = Field(default_factory=MLflowParams)
