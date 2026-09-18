from collections.abc import Callable
from time import perf_counter

from src.classes.data_registry import dataset_task_for_target
from src.classes.dataset import Dataset
from src.classes.trainer import Trainer
from src.schemas.base_schemas import TaskType
from src.schemas.metrics import (
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
from src.schemas.training_schemas import ModelConfig
from src.utils.logger import logger
from src.utils.model_identity import model_instance_ids
from src.utils.prediction_tables import PredictionTableAccumulator, cohort_fingerprints_from_summary


class Pipeline:
    """Orchestrate dataset loading, model training, evaluation, and cleanup."""

    def __init__(self, pipeline_config: PipelineConfig):
        self.pipeline_config = pipeline_config

        self.dataset = Dataset(pipeline_config.dataset)
        self.prediction_tables = PredictionTableAccumulator()

    def run(
        self,
        on_model_complete: Callable[[PipelineRunRecord, ModelRunRecord], None] | None = None,
    ) -> PipelineRunRecord:
        start_time = perf_counter()
        self.prediction_tables = PredictionTableAccumulator()
        target = self.pipeline_config.dataset.target
        task_type = dataset_task_for_target(target).task_type
        logger.info(
            f"Pipeline {self.pipeline_config.run_id} starting: "
            f"target={target} models={len(self.pipeline_config.training)}"
        )
        trainer = Trainer(
            task_type=task_type,
            default_imputer=self.pipeline_config.dataset.imputer,
            default_scaler=self.pipeline_config.dataset.scaler_encoder,
            log_transform_target=self.pipeline_config.dataset.log_transform_target,
        )

        data = self.dataset.get_dataset()
        dataset_summary = self.dataset.summarize(data)
        if task_type == "classification":
            self.prediction_tables = PredictionTableAccumulator(cohort_fingerprints_from_summary(dataset_summary))

        model_runs = []
        for model_instance_id, model_config in zip(
            model_instance_ids(self.pipeline_config.training),
            self.pipeline_config.training,
            strict=True,
        ):
            model_start_time = perf_counter()
            tr = None
            mr = None
            captured_predictions = []
            failure_stage = "training_evaluation"
            try:
                tr = trainer.train_evaluate_model(
                    model_config,
                    data,
                    on_test_predictions=captured_predictions.append,
                )
                mr = self._model_result_from_training_result(tr)
                if captured_predictions:
                    try:
                        self.prediction_tables.add(model_instance_id, captured_predictions[0])
                    except Exception:  # noqa: BLE001 - persistence must not invalidate model evaluation
                        logger.exception(f"Could not accumulate test predictions for {model_instance_id}; continuing")
                logger.info(f"Model {model_instance_id} trained and evaluated successfully")
            except Exception as exc:  # noqa: BLE001 - one model's failure must not abort the run
                logger.exception(f"Model {model_config.name} failed during {failure_stage}; continuing")
                if tr is None:
                    tr = self._failed_training_result(
                        model_config=model_config,
                        task_type=task_type,
                        fit_time=perf_counter() - model_start_time,
                        failure_stage=failure_stage,
                        exc=exc,
                    )
                else:
                    tr.error = _format_exception(exc)
                    tr.failure_stage = failure_stage
            finally:
                if tr is not None:
                    model_run = ModelRunRecord(
                        model_instance_id=model_instance_id,
                        training_result=tr,
                        evaluation=mr,
                    )
                    model_runs.append(model_run)
                    if on_model_complete is not None:
                        try:
                            on_model_complete(
                                PipelineRunRecord(
                                    run_id=self.pipeline_config.run_id,
                                    dataset_summary=dataset_summary,
                                    model_runs=tuple(model_runs),
                                    total_time=perf_counter() - start_time,
                                ),
                                model_run,
                            )
                        except Exception:  # noqa: BLE001 - tracking must not abort later models
                            logger.exception(f"Completion callback failed for {model_instance_id}; continuing")

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
        model_config: ModelConfig,
        task_type: TaskType,
        fit_time: float,
        failure_stage: str,
        exc: Exception,
    ) -> ModelTrainingResult:
        return ModelTrainingResult(
            model_name=model_config.name,
            task_type=task_type,
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
        mimic_metrics = test_metrics.mimic_test.metrics
        tudd_metrics = test_metrics.tudd_test.metrics
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
