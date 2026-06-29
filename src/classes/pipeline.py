from dataclasses import dataclass
from time import perf_counter
from typing import Callable

from src.classes.dataset import Dataset
from src.classes.plotter import Plotter
from src.classes.trainer import Trainer
from src.schemas.dataset_schemas import DatasetBundle, DatasetSummary, XYDataset
from src.schemas.pipeline_schemas import PipelineParams
from src.schemas.training_schemas import (
    ClassificationMetrics,
    ModelTrainingResult,
)
from src.utils.evaluation_utils import (
    FinalTestMetrics,
    evaluate_classification_predictions,
    final_test_metrics,
)
from src.utils.logger import logger
from src.utils.model_identity import model_instance_ids
from src.utils.model_lifecycle import release_training_result_model


@dataclass(frozen=True)
class TestSetEvaluationResult:
    dataset_name: str
    metrics: ClassificationMetrics
    predict_time: float


@dataclass(frozen=True)
class ModelRunResult:
    model_name: str
    test_results: tuple[TestSetEvaluationResult, ...]
    final_test_metrics: FinalTestMetrics
    fit_time: float

    @property
    def total_time(self) -> float:
        return self.fit_time + sum(result.predict_time for result in self.test_results)

    @property
    def metrics_by_test_set(self) -> dict[str, ClassificationMetrics]:
        return {result.dataset_name: result.metrics for result in self.test_results}


@dataclass(frozen=True)
class ModelRunRecord:
    model_instance_id: str
    training_result: ModelTrainingResult
    model_result: ModelRunResult | None

    @property
    def model_name(self) -> str:
        return self.training_result.model_name

    @property
    def succeeded(self) -> bool:
        return self.training_result.succeeded


@dataclass(frozen=True)
class PipelineResult:
    run_id: str
    dataset_summary: DatasetSummary
    model_runs: tuple[ModelRunRecord, ...]
    total_time: float

    @property
    def model_results(self) -> tuple[ModelRunResult, ...]:
        return tuple(
            run.model_result for run in self.model_runs if run.model_result is not None
        )

    @property
    def training_results(self) -> tuple[ModelTrainingResult, ...]:
        return tuple(run.training_result for run in self.model_runs)


class Pipeline:
    def __init__(self, params: PipelineParams):
        self.params = params

        self.dataset = Dataset(params.dataset)
        self.plotter = Plotter(params.plotting)

    def run(
        self,
        on_model_complete: Callable[[PipelineResult, ModelRunRecord], None]
        | None = None,
    ) -> PipelineResult:
        start_time = perf_counter()
        target = getattr(self.params.dataset, "target", "unknown")
        logger.info(
            f"Pipeline {self.params.run_id} starting: "
            f"target={target} models={len(self.params.training)}"
        )
        trainer = Trainer(
            params=self.params.training,
            default_imputer=self.params.dataset.imputer,
            default_scaler=self.params.dataset.scaler_encoder,
        )
        trainer.validate_model_configs()

        data = self.dataset.get_dataset()
        dataset_summary = self.dataset.summarize(data)

        model_runs = []
        y_train = data.train_data.y.to_numpy()
        for model_instance_id, model_params in zip(
            model_instance_ids(self.params.training),
            self.params.training,
            strict=True,
        ):
            model_start_time = perf_counter()
            tr = None
            mr = None
            failure_stage = "training"
            try:
                tr = trainer.train_model(
                    model_params=model_params,
                    X_train=data.train_data.X,
                    y_train=y_train,
                )
                failure_stage = "evaluation"
                mr = self._evaluate_trained_model(tr, data)
                logger.info(f"Model {model_instance_id} evaluated successfully")
            except Exception as exc:
                logger.exception(
                    f"Model {model_params.name} failed during {failure_stage}; continuing"
                )
                if tr is None:
                    tr = self._failed_training_result(
                        model_params=model_params,
                        fit_time=perf_counter() - model_start_time,
                        failure_stage=failure_stage,
                        exc=exc,
                    )
                else:
                    tr.error = _format_exception(exc)
                    tr.failure_stage = failure_stage
            finally:
                if tr is not None:
                    release_training_result_model(tr)
                    model_run = ModelRunRecord(
                        model_instance_id=model_instance_id,
                        training_result=tr,
                        model_result=mr,
                    )
                    model_runs.append(model_run)
                    if on_model_complete is not None:
                        on_model_complete(
                            PipelineResult(
                                run_id=self.params.run_id,
                                dataset_summary=dataset_summary,
                                model_runs=tuple(model_runs),
                                total_time=perf_counter() - start_time,
                            ),
                            model_run,
                        )

        total_time = perf_counter() - start_time
        logger.info(
            f"Pipeline {self.params.run_id} completed: "
            f"successful_models={len([run for run in model_runs if run.succeeded])}/"
            f"{len(model_runs)} total_time={total_time:.3f}s"
        )
        return PipelineResult(
            run_id=self.params.run_id,
            dataset_summary=dataset_summary,
            model_runs=tuple(model_runs),
            total_time=total_time,
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
            final_test_metrics=final_test_metrics(
                test_results[0].metrics,
                test_results[1].metrics,
            ),
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
        metrics = evaluate_classification_predictions(
            predictions, test_set.y.to_numpy()
        )

        return TestSetEvaluationResult(
            dataset_name=dataset_name,
            metrics=metrics,
            predict_time=predict_time,
        )

    @staticmethod
    def _failed_training_result(
        model_params,
        fit_time: float,
        failure_stage: str,
        exc: Exception,
    ) -> ModelTrainingResult:
        return ModelTrainingResult(
            model_name=model_params.name,
            task_type=model_params.task_type,
            trained_model=None,
            tuned=False,
            fit_time=fit_time,
            error=_format_exception(exc),
            failure_stage=failure_stage,
        )


def _format_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"
