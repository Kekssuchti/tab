from datetime import date
from uuid import uuid4

from pydantic import Field

from src.schemas.base_schemas import StrictConfig
from src.schemas.dataset_schemas import DatasetConfig
from src.schemas.plotting_schemas import PlottingConfig
from src.schemas.training_schemas import ModelConfig


def _default_run_id() -> str:
    return f"{date.today().isoformat()}_{uuid4().hex}"


class MLflowConfig(StrictConfig):
    """
    Configuration for MLflow tracking.

    ---
    Attributes:
        enabled: bool, default=True
            Whether MLflow logging is enabled.

        tracking_uri: str, default="sqlite:///mlflow.db"
            MLflow tracking backend URI.

        artifact_location: str or None, default="mlartifacts"
            Default artifact storage location for the experiment.

        experiment_name: str, default="tab"
            MLflow experiment name.

        run_name: str or None, default=None
            Optional name for the parent pipeline run.
    """

    enabled: bool = True
    tracking_uri: str = "sqlite:///mlflow.db"
    artifact_location: str | None = "mlartifacts"
    experiment_name: str = "tab"
    run_name: str | None = None


class PipelineConfig(StrictConfig):
    """
    Top-level pipeline configuration.

    ---
    Attributes:
        run_id: str, default=generated
            Unique identifier for this pipeline run.

        dataset: DatasetConfig
            Dataset loading, splitting, and preprocessing settings.

        training: tuple of ModelConfig, default=(ModelConfig(name="tabpfn-3"),)
            Models to train and evaluate.

        plotting: PlottingConfig
            Plotting settings.

        mlflow: MLflowConfig, default=MLflowConfig()
            MLflow logging settings.
    """

    run_id: str = Field(default_factory=_default_run_id)
    dataset: DatasetConfig = Field()
    training: tuple[ModelConfig, ...] = (ModelConfig(name="tabpfn-3"),)
    plotting: PlottingConfig = Field()
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
