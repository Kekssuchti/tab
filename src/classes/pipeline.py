from dataclasses import dataclass
from time import perf_counter

from src.classes.dataset import Dataset
from src.classes.plotter import Plotter
from src.classes.trainer import Trainer
from src.evaluation.evaluation_utils import evaluate_classification_predictions
from src.schemas.dataset_schemas import DatasetBundle, DatasetSummary, XYDataset
from src.schemas.pipeline_schemas import PipelineParams
from src.schemas.training_schemas import (
    ClassificationMetrics,
    ModelTrainingResult,
)


@dataclass(frozen=True)
class TestSetEvaluationResult:
    dataset_name: str
    metrics: ClassificationMetrics
    predict_time: float


@dataclass(frozen=True)
class ModelRunResult:
    model_name: str
    test_results: tuple[TestSetEvaluationResult, ...]
    fit_time: float

    @property
    def total_time(self) -> float:
        return self.fit_time + sum(result.predict_time for result in self.test_results)

    @property
    def metrics_by_test_set(self) -> dict[str, ClassificationMetrics]:
        return {result.dataset_name: result.metrics for result in self.test_results}


@dataclass(frozen=True)
class PipelineResult:
    run_id: str
    dataset_summary: DatasetSummary
    model_results: tuple[ModelRunResult, ...]
    training_results: tuple[ModelTrainingResult, ...]
    total_time: float


class Pipeline:
    def __init__(self, params: PipelineParams):
        self.params = params
        self.params.run_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = Dataset(params.dataset)
        self.plotter = Plotter(params.plotting)

    def run(self) -> PipelineResult:
        start_time = perf_counter()
        data = self.dataset.get_dataset()
        dataset_summary = self.dataset.summarize(data)

        trainer = Trainer(
            params=self.params.training,
            default_imputer=self.params.dataset.imputer,
            default_scaler=self.params.dataset.scaler_encoder,
        )

        training_results = trainer.train_models(
            X_train=data.train_data.X,
            y_train=data.train_data.y.to_numpy(),
        )

        model_results = []
        for tr in training_results:
            mr = self._evaluate_trained_model(tr, data)
            model_results.append(mr)
            del tr, mr

        return PipelineResult(
            run_id=self.params.run_id,
            dataset_summary=dataset_summary,
            model_results=tuple(model_results),
            training_results=tuple(training_results),
            total_time=perf_counter() - start_time,
        )

    def _evaluate_trained_model(
        self,
        training_result: ModelTrainingResult,
        data: DatasetBundle,
    ) -> ModelRunResult:
        test_results = (
            self._evaluate_test_set("mimic", training_result, data.test_mimic),
            self._evaluate_test_set("tudd", training_result, data.test_tudd),
        )

        return ModelRunResult(
            model_name=training_result.model_name,
            test_results=test_results,
            fit_time=training_result.fit_time,
        )

    @staticmethod
    def _evaluate_test_set(
        dataset_name: str,
        training_result: ModelTrainingResult,
        test_set: XYDataset,
    ) -> TestSetEvaluationResult:
        if training_result.task_type != "classification":
            raise NotImplementedError("Regression evaluation is not implemented yet")

        predictions, predict_time = training_result.trained_model.predict(test_set.X)
        scoring = (
            training_result.tuning_result.scoring
            if training_result.tuning_result is not None
            else "roc_auc"
        )
        metrics = evaluate_classification_predictions(
            scoring, predictions, test_set.y.to_numpy()
        )

        return TestSetEvaluationResult(
            dataset_name=dataset_name,
            metrics=metrics,
            predict_time=predict_time,
        )
