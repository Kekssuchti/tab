from time import perf_counter
from typing import Callable

from src.classes.dataset import Dataset
from src.classes.plotter import Plotter
from src.classes.trainer import Trainer
from src.schemas.metrics import (
    ClassificationMetrics,
    ClassificationMetricsAggregate,
    FinalTestMetrics,
)
from src.schemas.pipeline_schemas import PipelineConfig
from src.schemas.run_records import (
    ModelEvaluationRecord,
    ModelRunRecord,
    ModelTrainingResult,
    PipelineRunRecord,
    TestSetEvaluationRecord,
)
from src.utils.logger import logger
from src.utils.model_identity import model_instance_ids
from src.utils.model_lifecycle import release_training_result_model


class Pipeline:
    """Orchestrate dataset loading, model training, evaluation, and cleanup."""

    def __init__(self, pipeline_config: PipelineConfig):
        self.pipeline_config = pipeline_config

        self.dataset = Dataset(pipeline_config.dataset)
        self.plotter = Plotter(pipeline_config.plotting)

    def run(
        self,
        on_model_complete: Callable[[PipelineRunRecord, ModelRunRecord], None] | None = None,
    ) -> PipelineRunRecord:
        start_time = perf_counter()
        target = getattr(self.pipeline_config.dataset, "target", "unknown")
        logger.info(
            f"Pipeline {self.pipeline_config.run_id} starting: "
            f"target={target} models={len(self.pipeline_config.training)}"
        )
        trainer = Trainer(
            self.pipeline_config.training,
            self.pipeline_config.dataset.imputer,
            self.pipeline_config.dataset.scaler_encoder,
        )

        data = self.dataset.get_dataset()
        dataset_summary = self.dataset.summarize(data)

        model_runs = []
        for model_instance_id, model_config in zip(
            model_instance_ids(self.pipeline_config.training),
            self.pipeline_config.training,
            strict=True,
        ):
            model_start_time = perf_counter()
            tr = None
            mr = None
            failure_stage = "training_evaluation"
            try:
                tr = trainer.train_evaluate_model(model_config, data)
                mr = self._model_result_from_training_result(tr)
                logger.info(f"Model {model_instance_id} trained and evaluated successfully")
            except Exception as exc:
                logger.exception(f"Model {model_config.name} failed during {failure_stage}; continuing")
                if tr is None:
                    tr = self._failed_training_result(
                        model_config=model_config,
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
                        evaluation=mr,
                    )
                    model_runs.append(model_run)
                    if on_model_complete is not None:
                        on_model_complete(
                            PipelineRunRecord(
                                run_id=self.pipeline_config.run_id,
                                dataset_summary=dataset_summary,
                                model_runs=tuple(model_runs),
                                total_time=perf_counter() - start_time,
                            ),
                            model_run,
                        )

        total_time = perf_counter() - start_time
        logger.info(
            f"Pipeline {self.pipeline_config.run_id} completed: "
            f"successful_models={len([run for run in model_runs if run.succeeded])}/"
            f"{len(model_runs)} total_time={total_time:.3f}s"
        )
        return PipelineRunRecord(
            run_id=self.pipeline_config.run_id,
            dataset_summary=dataset_summary,
            model_runs=tuple(model_runs),
            total_time=total_time,
        )

    @staticmethod
    def _failed_training_result(
        model_config,
        fit_time: float,
        failure_stage: str,
        exc: Exception,
    ) -> ModelTrainingResult:
        return ModelTrainingResult(
            model_name=model_config.name,
            task_type=getattr(model_config, "task_type", "classification"),
            trained_model=None,
            tuned=False,
            fit_time=fit_time,
            error=_format_exception(exc),
            failure_stage=failure_stage,
        )

    @staticmethod
    def _model_result_from_training_result(
        training_result: ModelTrainingResult,
    ) -> ModelEvaluationRecord | None:
        tuning_result = training_result.tuning_result
        if tuning_result is None:
            return None

        test_metrics = tuning_result.final_test_metrics
        mimic_metrics = _cv_metrics_to_classification_metrics(test_metrics.mimic_test)
        tudd_metrics = _cv_metrics_to_classification_metrics(test_metrics.tudd_test)
        test_results = (
            TestSetEvaluationRecord(
                "mimic",
                mimic_metrics,
                test_metrics.mimic_prediction_time,
            ),
            TestSetEvaluationRecord(
                "tudd",
                tudd_metrics,
                test_metrics.tudd_prediction_time,
            ),
        )
        return ModelEvaluationRecord(
            model_name=training_result.model_name,
            test_results=test_results,
            final_test_metrics=FinalTestMetrics(
                mimic_test=mimic_metrics,
                mimic_prediction_time=test_metrics.mimic_prediction_time,
                tudd_test=tudd_metrics,
                tudd_prediction_time=test_metrics.tudd_prediction_time,
            ),
            fit_time=training_result.fit_time,
        )


def _format_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _cv_metrics_to_classification_metrics(
    metrics: ClassificationMetricsAggregate,
) -> ClassificationMetrics:
    return ClassificationMetrics(
        roc_auc=metrics.mean_roc_auc,
        prc_auc=metrics.mean_prc_auc,
        f1=metrics.mean_f1,
        accuracy=metrics.mean_accuracy,
        sensitivity=metrics.mean_sensitivity,
        precision=metrics.mean_precision,
        confusion_matrix=metrics.mean_confusion_matrix,
        n_classes=metrics.n_classes,
    )
