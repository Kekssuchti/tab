from datetime import date
from pathlib import Path
from uuid import uuid4

from pydantic import Field

from src.config import config
from src.schemas.base_schemas import StrictParams
from src.schemas.dataset_schemas import DatasetParams
from src.schemas.plotting_schemas import PlottingParams
from src.schemas.training_schemas import ModelParams


class MLflowParams(StrictParams):
    enabled: bool = True
    tracking_uri: str = "sqlite:///mlflow.db"
    artifact_location: str | None = "mlartifacts"
    experiment_name: str = "tab"
    run_name: str | None = None
    nested_model_runs: bool = True
    log_models: bool = False


class PipelineParams(StrictParams):
    dataset: DatasetParams = Field()
    training: tuple[ModelParams, ...] = (ModelParams(name="tabpfn-3"),)
    plotting: PlottingParams = Field()
    mlflow: MLflowParams = Field(default_factory=MLflowParams)

    @property
    def run_id(self) -> str:
        # today iso format + random uuid
        return f"{date.today().isoformat()}_{uuid4().hex}"
