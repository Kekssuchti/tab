from dataclasses import dataclass

import numpy as np

from src.classes.dataset import Dataset
from src.classes.evaluator import Evaluator
from src.classes.plotter import Plotter
from src.classes.preprocessor import Preprocessor
from src.classes.trainer import Trainer
from src.evaluation.evaluate import evaluate_predictions
from src.schemas.pipeline_schemas import PipelineParams
from src.schemas.training_schemas import ModelTrainingResult


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
    training_results: tuple[ModelTrainingResult, ...]


class Pipeline:
    def __init__(self, params: PipelineParams):
        self.params = params
        self.params.run_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = Dataset(params.dataset)
        self.evaluator = Evaluator(params.evaluation)
        self.plotter = Plotter(params.plotting)

    def run(self) -> PipelineResult:
        data = self.dataset.get_dataset()

        preprocessor = Preprocessor(
            params_imputer=self.params.dataset.imputer,
            params_scaler=self.params.dataset.scaler_encoder,
        )
        preprocess_pipeline = preprocessor.build_pipeline()

        trainer = Trainer(
            params=self.params.training,
            preprocess_pipeline=preprocess_pipeline,
        )

        training_results = trainer.train_models(
            X_train=data.train_data.X.to_numpy(),
            y_train=data.train_data.y.to_numpy(),
        )

        model_results = []
        for tr in training_results:
            mr = self._evaluate_trained_model(tr, data)
            model_results.append(mr)

        return PipelineResult(
            run_id=self.params.run_id,
            model_results=tuple(model_results),
            training_results=tuple(training_results),
        )

    def _evaluate_trained_model(
        self,
        training_result: ModelTrainingResult,
        data,
    ) -> ModelRunResult:
        adapter = training_result.trained_model

        predictions, predict_time = adapter.predict(data.test_mimic.X.to_numpy())
        metrics = self._evaluate(
            training_result.task_type, predictions, data.test_mimic.y.to_numpy()
        )

        return ModelRunResult(
            model_name=training_result.model_name,
            metrics=metrics,
            fit_time=training_result.fit_time,
            predict_time=predict_time,
        )

    @staticmethod
    def _evaluate(
        task_type: str,
        predictions: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, float]:
        if task_type != "classification":
            raise NotImplementedError("Regression evaluation is not implemented yet")

        return evaluate_predictions(predictions, y_test)
