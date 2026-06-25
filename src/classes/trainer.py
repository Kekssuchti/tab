from timeit import default_timer as timer
from typing import Any

import numpy as np

from src.classes.preprocessor import Preprocessor
from src.interfaces.model_interface import ModelAdapter, PreprocessedModelAdapter
from src.schemas.preprocessing_schemas import ImputerParams, ScalerEncoderParams
from src.schemas.training_schemas import (
    ClassificationMetrics,
    FoldResult,
    ModelParams,
    ModelTrainingResult,
    TuningCVResults,
    TuningResult,
)
from src.utils.evaluation_utils import (
    classification_score,
    evaluate_classification_predictions,
    mean_classification_metrics,
)
from src.utils.logger import logger
from src.utils.model_lifecycle import release_model
from src.utils.model_registry import ModelSpec
from src.utils.trainer_utils import get_model_spec
from src.utils.tuning_utils import build_cv, get_candidate_params


class Trainer:
    def __init__(
        self,
        params: tuple[ModelParams, ...],
        default_imputer: ImputerParams,
        default_scaler: ScalerEncoderParams,
    ) -> None:
        self.params = params
        self.default_imputer = default_imputer
        self.default_scaler = default_scaler

    def validate_model_configs(self) -> None:
        logger.info(f"Validating {len(self.params)} model config(s)")
        errors: list[str] = []
        for model_params in self.params:
            try:
                spec = get_model_spec(model_params)
                self._build_preprocess_pipeline(model_params)
                for params in self._preflight_param_sets(model_params, spec):
                    model = None
                    try:
                        model = spec.create(model_params.task_type, params)
                    finally:
                        release_model(model)
            except Exception as exc:
                errors.append(f"{model_params.name}: {exc}")

        if errors:
            joined_errors = "\n- ".join(errors)
            raise ValueError(f"Model preflight validation failed:\n- {joined_errors}")
        logger.info("Model config validation completed")

    def train_model(
        self,
        model_params: ModelParams,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> ModelTrainingResult:
        logger.info(f"Training model: {model_params.name}")

        imputer, scaler = self._resolved_preprocessing(model_params)
        logger.info(f"data imputation via: {imputer.imputation_method}")
        logger.info(f"scaling data using: {scaler.type}")

        spec = get_model_spec(model_params)

        if model_params.tuning is None:
            return self._train_without_tuning(model_params, spec, X_train, y_train)

        return self._tune_model(model_params, spec, X_train, y_train)

    def _train_without_tuning(
        self,
        model_params: ModelParams,
        spec: ModelSpec,
        X_train,
        y_train,
    ) -> ModelTrainingResult:
        trained_model = None
        try:
            trained_model, fit_time = self._fit_model(
                model_params, spec, model_params.params, X_train, y_train
            )
            logger.info(f"Model {model_params.name} fit in {fit_time:.3f}s")

            training_metrics, predict_time = self._training_metrics(
                model_params, trained_model, X_train, y_train
            )
            logger.info(f"Training predictions took {predict_time:.3f}s")
            logger.info(f"Training metrics: {training_metrics}")

            return ModelTrainingResult(
                model_name=model_params.name,
                task_type=model_params.task_type,
                trained_model=trained_model,
                tuned=False,
                fit_time=fit_time,
                training_metrics=training_metrics,
            )
        except Exception:
            release_model(trained_model)
            raise

    def _fit_model(
        self,
        model_params: ModelParams,
        spec: ModelSpec,
        params: dict[str, Any],
        X_train,
        y_train,
    ) -> tuple[ModelAdapter, float]:
        adapter = spec.create(model_params.task_type, params)
        model = PreprocessedModelAdapter(
            adapter,
            self._build_preprocess_pipeline(model_params),
        )
        fit_time = model.fit(X_train, y_train)
        return model, fit_time

    def _build_preprocess_pipeline(self, model_params: ModelParams):
        imputer, scaler = self._resolved_preprocessing(model_params)
        return Preprocessor(
            params_imputer=imputer,
            params_scaler=scaler,
        ).build_pipeline()

    def _resolved_preprocessing(
        self, model_params: ModelParams
    ) -> tuple[ImputerParams, ScalerEncoderParams]:
        preprocessing = model_params.preprocessing
        imputer = (
            preprocessing.imputer
            if preprocessing is not None and preprocessing.imputer is not None
            else self.default_imputer
        )
        scaler = (
            preprocessing.scaler_encoder
            if preprocessing is not None and preprocessing.scaler_encoder is not None
            else self.default_scaler
        )
        return imputer, scaler

    def _preflight_param_sets(
        self,
        model_params: ModelParams,
        spec: ModelSpec,
    ) -> list[dict[str, Any]]:
        if model_params.tuning is None:
            return [model_params.params]

        grid = spec.tuning_grid(
            model_params.tuning.search_space,
            model_params.tuning.grid,
        )
        return [
            {**model_params.params, **candidate_params}
            for candidate_params in get_candidate_params(grid)
        ]

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

        logger.info(f"Start tuning: {model_params.name}")
        grid = spec.tuning_grid(tuning.search_space, tuning.grid)
        candidates = get_candidate_params(grid)
        cv = build_cv(model_params, tuning)
        logger.info(
            f"Tuning {model_params.name}: candidates={len(candidates)} "
            f"folds={tuning.cv.n_splits} scoring={tuning.scoring}"
        )

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
                fold_model = None
                predictions = None
                try:
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
                        predictions,
                        self._take_rows(y_train, validation_index),
                    )

                    candidate_scores.append(
                        classification_score(metrics, tuning.scoring)
                    )
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
                finally:
                    predictions = None
                    release_model(fold_model)

            fold_scores_by_candidate.append(candidate_scores)
            fold_metrics_by_candidate.append(candidate_metrics)
            fold_times_by_candidate.append(candidate_times)

        mean_scores = [float(np.mean(scores)) for scores in fold_scores_by_candidate]
        std_scores = [float(np.std(scores)) for scores in fold_scores_by_candidate]
        mean_metrics = [
            mean_classification_metrics(metrics)
            for metrics in fold_metrics_by_candidate
        ]
        best_index = int(np.argmax(mean_scores))
        best_params = candidates[best_index]

        logger.info(f"CV Done. Best params: {best_params}")
        trained_model = None
        try:
            trained_model, fit_time = self._fit_model(
                model_params,
                spec,
                {**model_params.params, **best_params},
                X_train,
                y_train,
            )
            training_metrics, predict_time = self._training_metrics(
                model_params, trained_model, X_train, y_train
            )

            logger.info(f"Model {model_params.name} fit in {fit_time:.3f}s")
            logger.info(f"Training predictions took {predict_time:.3f}s")

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
                f"Model tuning complete in {tuning_result.total_time:.3f}s. "
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
        except Exception:
            release_model(trained_model)
            raise

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
    ) -> tuple[ClassificationMetrics | None, float]:
        if model_params.task_type != "classification":
            return None, 0.0

        predictions, predict_time = model.predict(X_train)
        return evaluate_classification_predictions(predictions, y_train), predict_time
