from datetime import date
from pathlib import Path
from uuid import uuid4

from pydantic import Field

from src.schemas.base_schemas import StrictConfig
from src.schemas.dataset_schemas import DatasetConfig
from src.schemas.plotting_schemas import PlottingConfig
from src.schemas.training_schemas import ModelConfig


def _default_run_id() -> str:
    return f"{date.today().isoformat()}_{uuid4().hex}"


class MLflowConfig(StrictConfig):
    enabled: bool = True
    tracking_uri: str = "sqlite:///mlflow.db"
    artifact_location: str | None = "mlartifacts"
    experiment_name: str = "tab"
    run_name: str | None = None


class PipelineConfig(StrictConfig):
    run_id: str = Field(default_factory=_default_run_id)
    dataset: DatasetConfig = Field()
    training: tuple[ModelConfig, ...] = (ModelConfig(name="tabpfn-3"),)
    plotting: PlottingConfig = Field()
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
