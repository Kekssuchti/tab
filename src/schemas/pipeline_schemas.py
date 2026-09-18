from datetime import date
from uuid import uuid4

from pydantic import Field

from src.schemas.base_schemas import StrictConfig
from src.schemas.dataset_schemas import DatasetConfig
from src.schemas.training_schemas import ModelConfig


def _default_run_id() -> str:
    return f"{date.today().isoformat()}_{uuid4().hex}"


class RandomStates(StrictConfig):
    """
    Dedicated seeds for every pipeline step that consumes randomness.

    One field per consuming step, so a repetition can change exactly the sources
    it wants to study. Two seeds deliberately stay put:
    `DatasetConfig.random_state` keeps the train/test split identical between
    repetitions and `evaluation_bootstrap_seed` keeps the bootstrap draws of
    every repetition aligned, which is what makes confidence intervals and
    paired win matrices comparable across repetitions.

    ---
    Attributes:
        model_training_seed: int, default=1337
            Seed passed to a model adapter when the model is constructed. Covers
            the final fit as well as every cross-validation fold fit.

        model_inference_seed: int, default=1337
            Reserved. No consumer yet: adapters take a single construction-time
            seed that drives both fitting and predicting. Recorded in the run
            config so a future predict-time reseed does not change stored runs.

        cv_split_seed: int, default=1337
            Seed used to shuffle rows into the cross-validation folds.

        tuning_sampler_seed: int, default=1337
            Seed used by the Optuna sampler that proposes tuning candidates.

        training_sample_seed: int, default=1337
            Seed used to draw the configured training fraction and to order the
            combined training rows. Never touches the held-out test sets.

        evaluation_bootstrap_seed: int, default=1337
            Seed used for the bootstrap resampling behind reported intervals and
            pairwise win matrices. Keep it constant across repetitions.
    """

    model_training_seed: int = 1337
    model_inference_seed: int = 1337
    cv_split_seed: int = 1337
    tuning_sampler_seed: int = 1337
    training_sample_seed: int = 1337
    evaluation_bootstrap_seed: int = 1337

    @classmethod
    def from_seed(cls, seed: int) -> "RandomStates":
        """Return a state set where every step is seeded by the same value."""
        return cls(
            model_training_seed=seed,
            model_inference_seed=seed,
            cv_split_seed=seed,
            tuning_sampler_seed=seed,
            training_sample_seed=seed,
            evaluation_bootstrap_seed=seed,
        )


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
        random_states: RandomStates
            Dedicated seeds for every pipeline step that consumes randomness.
            Required, so every run records the seeds it was produced with.

        run_id: str, default=generated
            Unique identifier for this pipeline run.

        dataset: DatasetConfig
            Dataset loading, splitting, and preprocessing settings.

        training: tuple of ModelConfig
            Models to train and evaluate.

        mlflow: MLflowConfig, default=MLflowConfig()
            MLflow logging settings.
    """

    random_states: RandomStates
    dataset: DatasetConfig
    training: tuple[ModelConfig, ...]
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    run_id: str = Field(default_factory=_default_run_id)
