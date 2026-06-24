from datetime import date
from pathlib import Path

from pydantic import Field

from src.config import config
from src.schemas.base_schemas import StrictParams
from src.schemas.dataset_schemas import DatasetParams
from src.schemas.plotting_schemas import PlottingParams
from src.schemas.training_schemas import ModelParams


class MLflowParams(StrictParams):
    enabled: bool = True
    tracking_uri: str | None = "sqlite:///mlflow.db"
    artifact_location: str | None = "mlartifacts"
    experiment_name: str = "tab"
    run_name: str | None = None
    nested_model_runs: bool = True
    log_models: bool = False


class PipelineParams(StrictParams):
    run_number: int = Field(default=1, ge=1, le=9999)
    run_date: date = Field(default_factory=date.today)
    dataset: DatasetParams = Field()
    training: tuple[ModelParams, ...] = (ModelParams(name="tabpfn-3"),)
    plotting: PlottingParams = Field()
    mlflow: MLflowParams = Field(default_factory=MLflowParams)

    @property
    def run_id(self) -> str:
        return f"{self.run_number:04d}_{self.run_date.isoformat()}"

    @property
    def run_dir(self) -> Path:
        return config.dir_run_results / self.run_id
