from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from src.classes.dataset import Dataset
from src.classes.evaluator import Evaluator
from src.classes.plotter import Plotter
from src.classes.trainer import Trainer
from src.evaluation.evaluate import evaluate_predictions
from src.schemas.pipeline_schemas import PipelineParams
from src.schemas.training_schemas import ModelParams
from src.utils.model_registry import MODEL_REGISTRY_CLS, MODEL_REGISTRY_REG


@dataclass(frozen=True)
class ModelRunResult:
    model_name: str
    metrics: dict[str, float]
    fit_time: float
    predict_time: float

    @property
    def total_time(self) -> float:
        return self.fit_time + self.predict_time


@dataclass(frozen=True)
class PipelineResult:
    run_id: str
    model_results: tuple[ModelRunResult, ...]


class Pipeline:
    def __init__(self, params: PipelineParams):
        self.params = params
        self.params.run_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = Dataset(params.dataset)
        self.trainer = Trainer(params.training)
        self.evaluator = Evaluator(params.evaluation)
        self.plotter = Plotter(params.plotting)

    def run(self) -> PipelineResult:
        dataset = self._create_dataset()
        model_results = []

        for model_params in self.params.training.models:
            model_results.append(self._run_model(model_params, dataset))

        return PipelineResult(
            run_id=self.params.run_id,
            model_results=tuple(model_results),
        )

    def _create_dataset(self) -> dict[str, Any]:
        params = self.params.dataset

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=params.train_size,
            random_state=params.random_state,
            stratify=y if self._can_stratify(y) else None,
        )

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    def _run_model(
        self,
        model_params: ModelParams,
        dataset: dict[str, Any],
    ) -> ModelRunResult:
        model = self._create_model(model_params)
        fit_time = model.fit(dataset["X_train"], dataset["y_train"])
        predictions, predict_time = model.predict(dataset["X_test"])
        metrics = self._evaluate(model_params, predictions, dataset["y_test"])

        return ModelRunResult(
            model_name=model_params.name,
            metrics=metrics,
            fit_time=fit_time,
            predict_time=predict_time,
        )

    def _create_model(self, model_params: ModelParams):
        registry = (
            MODEL_REGISTRY_CLS
            if model_params.task_type == "classification"
            else MODEL_REGISTRY_REG
        )

        try:
            model_cls = registry[model_params.name]
        except KeyError as exc:
            available = ", ".join(sorted(registry))
            raise ValueError(
                f"Unknown {model_params.task_type} model '{model_params.name}'. "
                f"Available models: {available}"
            ) from exc

        return model_cls(task_type=model_params.task_type, **model_params.params)

    def _evaluate(
        self,
        model_params: ModelParams,
        predictions: Any,
        y_test: Any,
    ) -> dict[str, float]:
        if model_params.task_type != "classification":
            raise NotImplementedError("Regression evaluation is not implemented yet")

        return evaluate_predictions(np.asarray(predictions), y_test)

    @staticmethod
    def _can_stratify(y: Any) -> bool:
        y_array = np.asarray(y).ravel()
        if y_array.size < 2:
            return False

        _, counts = np.unique(y_array, return_counts=True)
        return bool(counts.size > 1 and counts.min() >= 2)
