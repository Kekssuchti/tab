from timeit import default_timer as timer
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from src.evaluation.evaluation_utils import (
    evaluate_classification_predictions,
    mean_classification_metrics,
)
from src.interfaces.model_interface import ModelAdapter, PreprocessedModelAdapter
from src.schemas.training_schemas import (
    ClassificationMetrics,
    FoldResult,
    ModelParams,
    ModelTrainingResult,
    TrainingParams,
    TuningCVResults,
    TuningResult,
)
from src.utils.logger import logger
from src.utils.model_registry import ModelSpec
from src.utils.trainer_utils import get_model_spec
from src.utils.tuning_utils import build_cv, get_candidate_params


class Trainer:
    def __init__(self, params: TrainingParams, preprocess_pipeline: Pipeline) -> None:
        self.params = params
        self.preprocess_pipeline = preprocess_pipeline

    def train_models(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> list[ModelTrainingResult]:
        logger.info("Starting with model training")
        results: list[ModelTrainingResult] = []

        for model_params in self.params.models:
            logger.info(f"Training model: {model_params.name}")
            result = self._train_single_model(model_params, X_train, y_train)
            results.append(result)

        return results

    def _train_single_model(
        self,
        model_params: ModelParams,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> ModelTrainingResult:
        spec = get_model_spec(model_params)

        if model_params.tuning is None:
            trained_model, fit_time = self._fit_model(
                model_params, spec, model_params.params, X_train, y_train
            )
            training_metrics = self._training_metrics(
                model_params, trained_model, X_train, y_train
            )

            logger.info(f"Model {model_params.name} fit in {fit_time:.3f}s")
            return ModelTrainingResult(
                model_name=model_params.name,
                task_type=model_params.task_type,
                trained_model=trained_model,
                tuned=False,
                fit_time=fit_time,
                training_metrics=training_metrics,
            )

        return self._tune_model(model_params, spec, X_train, y_train)

    def _fit_model(
        self,
        model_params: ModelParams,
        spec: ModelSpec,
        params: dict[str, Any],
        X_train,
        y_train,
    ) -> tuple[ModelAdapter, float]:
        adapter = spec.create(model_params.task_type, params)
        model = PreprocessedModelAdapter(adapter, clone(self.preprocess_pipeline))
        fit_time = model.fit(X_train, y_train)
        return model, fit_time

    def _tune_model(
        self,
        model_params: ModelParams,
        spec: ModelSpec,
        X_train,
        y_train,
    ) -> ModelTrainingResult:
        tuning = model_params.tuning
        if tuning is None:
            raise ValueError("Tuning requested without tuning parameters")

        grid = spec.tuning_grid(tuning.search_space, tuning.grid)
        candidates = get_candidate_params(grid)
        cv = build_cv(model_params, tuning)

        fold_results: list[FoldResult] = []
        fold_scores_by_candidate: list[list[float]] = []
        fold_metrics_by_candidate: list[list[ClassificationMetrics]] = []
        fold_times_by_candidate: list[list[float]] = []

        for candidate_index, candidate_params in enumerate(candidates):
            # We override model params with candidate
            params = {**model_params.params, **candidate_params}
            candidate_scores: list[float] = []
            candidate_metrics: list[ClassificationMetrics] = []
            candidate_times: list[float] = []

            for fold_index, (train_index, validation_index) in enumerate(
                cv.split(X_train, y_train)
            ):
                fold_start = timer()
                fold_model, _ = self._fit_model(
                    model_params,
                    spec,
                    params,
                    self._take_rows(X_train, train_index),
                    self._take_rows(y_train, train_index),
                )
                predictions, _ = fold_model.predict(
                    self._take_rows(X_train, validation_index)
                )
                if model_params.task_type != "classification":
                    raise NotImplementedError(
                        "Regression tuning metrics are not implemented yet"
                    )

                metrics = evaluate_classification_predictions(
                    tuning.scoring,
                    predictions,
                    self._take_rows(y_train, validation_index),
                )

                candidate_scores.append(metrics.primary_score)
                candidate_metrics.append(metrics)
                candidate_times.append(timer() - fold_start)
                fold_results.append(
                    FoldResult(
                        candidate_index=candidate_index,
                        fold_index=fold_index,
                        metrics=metrics,
                        time=candidate_times[-1],
                        params=params,
                    )
                )

            fold_scores_by_candidate.append(candidate_scores)
            fold_metrics_by_candidate.append(candidate_metrics)
            fold_times_by_candidate.append(candidate_times)

        mean_scores = [float(np.mean(scores)) for scores in fold_scores_by_candidate]
        std_scores = [float(np.std(scores)) for scores in fold_scores_by_candidate]
        mean_metrics = [
            mean_classification_metrics(tuning.scoring, metrics)
            for metrics in fold_metrics_by_candidate
        ]
        best_index = int(np.argmax(mean_scores))
        best_params = candidates[best_index]

        start = timer()
        trained_model, _ = self._fit_model(
            model_params,
            spec,
            {**model_params.params, **best_params},
            X_train,
            y_train,
        )
        fit_time = timer() - start
        training_metrics = self._training_metrics(
            model_params, trained_model, X_train, y_train
        )

        tuning_result = TuningResult(
            best_params=best_params,
            scoring=tuning.scoring,
            best_metrics=mean_metrics[best_index],
            cv_results=TuningCVResults(
                params=candidates,
                mean_scores=mean_scores,
                std_scores=std_scores,
                fold_scores=fold_scores_by_candidate,
                fold_times=fold_times_by_candidate,
                mean_metrics=mean_metrics,
            ),
            fold_results=fold_results,
        )

        logger.info(
            f"Model {model_params.name} tuning complete in {tuning_result.total_time:.3f}s. "
            f"Best {tuning.scoring}: {tuning_result.best_score:.4f}"
        )

        return ModelTrainingResult(
            model_name=model_params.name,
            task_type=model_params.task_type,
            trained_model=trained_model,
            tuned=True,
            fit_time=fit_time,
            training_metrics=training_metrics,
            tuning_result=tuning_result,
        )

    @staticmethod
    def _take_rows(data, rows: np.ndarray):
        if hasattr(data, "iloc"):
            return data.iloc[rows]
        return np.asarray(data)[rows]

    def _training_metrics(
        self,
        model_params: ModelParams,
        model: ModelAdapter,
        X_train,
        y_train,
    ) -> ClassificationMetrics | None:
        if model_params.task_type != "classification":
            return None

        predictions, _ = model.predict(X_train)
        scoring = model_params.tuning.scoring if model_params.tuning else "roc_auc"
        return evaluate_classification_predictions(scoring, predictions, y_train)
