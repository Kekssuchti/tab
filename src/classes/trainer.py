from dataclasses import dataclass
from timeit import default_timer as timer
from typing import Any, Literal

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from src.schemas.dataset import DatasetBundle

from src.classes.preprocessor import Preprocessor
from src.interfaces.model_interface import ModelAdapter, PreprocessedModelAdapter
from src.schemas.preprocessing_schemas import ImputerConfig, ScalerEncoderConfig
from src.schemas.training_schemas import (
    ClassificationMetrics,
    FoldResult,
    ModelConfig,
    ModelTrainingResult,
    TuningResult,
)
from src.utils.databundle_utils import _databundle_to_xy_train
from src.utils.evaluation import evaluate_trained_model
from src.utils.evaluation_utils import (
    FinalTestMetrics,
    _format_metrics,
    classification_score,
    evaluate_classification_predictions,
)
from src.utils.logger import logger
from src.utils.model_lifecycle import release_model
from src.utils.model_registry import ModelSpec, get_model_spec


@dataclass
class _CandidateEvaluation:
    candidate_params: dict[str, Any]
    fold_scores: list[float]
    fold_metrics: list[ClassificationMetrics]
    fold_times: list[float]
    fold_results: list[FoldResult]


class Trainer:
    def __init__(
        self,
        configs: tuple[ModelConfig, ...],
        default_imputer: ImputerConfig,
        default_scaler: ScalerEncoderConfig,
    ) -> None:
        self.configs = configs
        self.default_imputer = default_imputer
        self.default_scaler = default_scaler

    def validate_model_configs(self) -> None:
        logger.info(f"Validating {len(self.configs)} model config(s)")
        errors: list[str] = []
        for model_config in self.configs:
            try:
                spec = get_model_spec(model_config)
                self._build_preprocess_pipeline(model_config)
                for params in self._preflight_param_sets(model_config, spec):
                    model = None
                    try:
                        model = spec.create(model_config.task_type, params)
                    finally:
                        release_model(model)
            except Exception as exc:
                errors.append(f"{model_config.name}: {exc}")

        if errors:
            joined_errors = "\n- ".join(errors)
            raise ValueError(f"Model preflight validation failed:\n- {joined_errors}")
        logger.info("Model config validation completed")

    def train_evaluate_model(
        self, model_config: ModelConfig, data: DatasetBundle
    ) -> ModelTrainingResult:
        logger.info(f"Training model: {model_config.name}")

        imputer, scaler = self._resolved_preprocessing(model_config)
        logger.info(f"data imputation via: {imputer.imputation_method}")
        logger.info(f"scaling data using: {scaler.type}")

        spec = get_model_spec(model_config)

        if model_config.tuning is None:
            return self._train_without_tuning(model_config, spec, data)

        return self._tune_model(model_config, spec, data)

    def _train_without_tuning(
        self, model_params: ModelConfig, spec: ModelSpec, data: DatasetBundle
    ) -> ModelTrainingResult:
        trained_model = None
        X_train, y_train = _databundle_to_xy_train(data)
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
            )
            # TODO: add evaluation!
        except Exception:
            release_model(trained_model)
            raise

    def _fit_model(
        self,
        model_config: ModelConfig,
        model_spec: ModelSpec,
        model_params: dict[str, Any],
        X_train,
        y_train,
    ) -> tuple[ModelAdapter, float]:
        adapter = model_spec.create(model_config.task_type, model_params)
        model = PreprocessedModelAdapter(
            adapter,
            self._build_preprocess_pipeline(model_config),
        )
        fit_time = model.fit(X_train, y_train)
        return model, fit_time

    def _build_preprocess_pipeline(self, model_params: ModelConfig):
        imputer, scaler = self._resolved_preprocessing(model_params)
        return Preprocessor(
            imputer_config=imputer,
            scaler_config=scaler,
        ).build_pipeline()

    def _resolved_preprocessing(
        self, model_config: ModelConfig
    ) -> tuple[ImputerConfig, ScalerEncoderConfig]:
        preprocessing = model_config.preprocessing
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
        model_config: ModelConfig,
        spec: ModelSpec,
    ) -> list[dict[str, Any]]:
        if model_config.tuning is None:
            return [model_config.params]

        grid = spec.tuning_grid(
            model_config.tuning.search_space,
            model_config.tuning.grid,
        )
        self._count_grid_candidates(grid)

        param_sets = [model_config.params]
        for key, values in grid.items():
            for value in values:
                candidate_params = spec.tuning_candidate_from_values({key: value})
                param_sets.append(
                    self._merge_params(model_config.params, candidate_params)
                )
        return param_sets

    def _tune_model(
        self, model_config: ModelConfig, model_spec: ModelSpec, data: DatasetBundle
    ) -> ModelTrainingResult:
        tuning = model_config.tuning
        if tuning is None:
            raise ValueError("Tuning requested without tuning parameters")

        if tuning.method == "grid":
            return self._tune_model_grid(model_config, model_spec, data)
        if tuning.method == "optuna":
            return self._tune_model_optuna(model_config, model_spec, data)

        raise ValueError(f"Unknown tuning method: {tuning.method}")

    def _tune_model_grid(
        self, model_config: ModelConfig, model_spec: ModelSpec, data: DatasetBundle
    ) -> ModelTrainingResult:
        tuning = model_config.tuning
        if tuning is None:
            raise ValueError("Tuning requested without tuning parameters")

        X_train, y_train = _databundle_to_xy_train(data)

        candidates = model_spec.tuning_candidates(tuning.search_space, tuning.grid)
        folds = list(self._build_cv(model_config, tuning).split(X_train, y_train))
        logger.info(
            f"Tuning {model_config.name}: candidates={len(candidates)} "
            f"folds={len(folds)} scoring={tuning.scoring} method=grid"
        )

        evaluations: list[_CandidateEvaluation] = []

        for candidate_index, candidate_params in enumerate(candidates):
            start_time_candidate = timer()
            evaluation = self._evaluate_cv_candidate(
                model_config,
                model_spec,
                candidate_params,
                candidate_index,
                folds,
                X_train,
                y_train,
            )
            evaluations.append(evaluation)
            logger.info(
                f"Candidate {candidate_index + 1}/{len(candidates)} completed in "
                f"{timer() - start_time_candidate:.2f}s"
            )

        return self._fit_best_tuned_model(
            model_config,
            model_spec,
            data,
            evaluations,
            folds=folds,
            method="grid",
        )

    def _tune_model_optuna(
        self, model_config: ModelConfig, model_spec: ModelSpec, data: DatasetBundle
    ) -> ModelTrainingResult:
        import optuna

        self._configure_optuna_logging(optuna)

        tuning = model_config.tuning
        if tuning is None:
            raise ValueError("Tuning requested without tuning parameters")

        grid = model_spec.tuning_grid(tuning.search_space, tuning.grid)
        grid_candidate_count = self._count_grid_candidates(grid)
        X_train, y_train = _databundle_to_xy_train(data)

        folds = list(self._build_cv(model_config, tuning).split(X_train, y_train))

        logger.info(
            f"Tuning {model_config.name}: trials={tuning.optuna.n_trials} "
            f"grid_candidates={grid_candidate_count} folds={len(folds)} "
            f"scoring={tuning.scoring} method=optuna"
        )

        evaluations: list[_CandidateEvaluation] = []

        study = optuna.create_study(
            direction="maximize",
            sampler=self._build_optuna_sampler(tuning),
        )

        def objective(trial):
            candidate_index = len(evaluations)
            sampled_params = {
                key: trial.suggest_categorical(key, values)
                for key, values in grid.items()
            }
            candidate_params = model_spec.tuning_candidate_from_values(sampled_params)
            start_time_candidate = timer()
            evaluation = self._evaluate_cv_candidate(
                model_config,
                model_spec,
                candidate_params,
                candidate_index,
                folds,
                X_train,
                y_train,
            )
            evaluations.append(evaluation)
            logger.info(
                f"Optuna trial {trial.number + 1}/{tuning.optuna.n_trials} "
                f"completed in {timer() - start_time_candidate:.2f}s"
            )
            return float(np.mean(evaluation.fold_scores))

        study.optimize(
            objective,
            n_trials=tuning.optuna.n_trials,
            timeout=tuning.optuna.timeout,
            n_jobs=1,
            gc_after_trial=True,
        )

        if not evaluations:
            raise ValueError("Optuna tuning did not complete any trials")

        return self._fit_best_tuned_model(
            model_config,
            model_spec,
            data,
            evaluations,
            folds=folds,
            method="optuna",
        )

    def _evaluate_cv_candidate(
        self,
        model_config: ModelConfig,
        model_spec: ModelSpec,
        candidate_params: dict[str, Any],
        candidate_index: int,
        folds: list[tuple[np.ndarray, np.ndarray]],
        X_train,
        y_train,
    ) -> _CandidateEvaluation:
        tuning = model_config.tuning
        if tuning is None:
            raise ValueError("Tuning requested without tuning parameters")

        model_params = self._merge_params(model_config.params, candidate_params)
        candidate_scores: list[float] = []
        candidate_metrics: list[ClassificationMetrics] = []
        candidate_times: list[float] = []
        fold_results: list[FoldResult] = []

        for fold_index, (train_index, validation_index) in enumerate(folds):
            fold_start = timer()
            fold_model = None
            predictions = None
            try:
                fold_model, _ = self._fit_model(
                    model_config,
                    model_spec,
                    model_params,
                    self._take_rows(X_train, train_index),
                    self._take_rows(y_train, train_index),
                )
                predictions, _ = fold_model.predict(
                    self._take_rows(X_train, validation_index)
                )
                if model_config.task_type != "classification":
                    raise NotImplementedError(
                        "Regression tuning metrics are not implemented yet"
                    )

                metrics = evaluate_classification_predictions(
                    predictions,
                    self._take_rows(y_train, validation_index),
                )

                candidate_scores.append(classification_score(metrics, tuning.scoring))
                candidate_metrics.append(metrics)
                candidate_times.append(timer() - fold_start)

                fold_results.append(
                    FoldResult(
                        candidate_index=candidate_index,
                        fold_index=fold_index,
                        metrics=metrics,
                        time=candidate_times[-1],
                        params=model_params,
                    )
                )
            finally:
                predictions = None
                release_model(fold_model)

        return _CandidateEvaluation(
            candidate_params=candidate_params,
            fold_scores=candidate_scores,
            fold_metrics=candidate_metrics,
            fold_times=candidate_times,
            fold_results=fold_results,
        )

    def _fit_best_tuned_model(
        self,
        model_config: ModelConfig,
        model_spec: ModelSpec,
        data: DatasetBundle,
        evaluations: list[_CandidateEvaluation],
        folds,
        *,
        method: Literal["grid", "optuna"],
    ) -> ModelTrainingResult:
        """
        Fits the best found params for a model to each fold and evaluates it against the common test sets.
        Results get consolidated into a single `ModelTrainingResult`.
        This includes mean scores for all metrics and their 95% confidence intervals.

        Args:
            model_params: The model parameters to fit.
            spec: The model spec to use for fitting.
            data: The dataset bundle to use for training and evaluation.
            evaluations: The candidate evaluations to consolidate.
            folds: The folds to use for evaluation (consistent with folds where model_params were found).
            method: The tuning method to use.

        Returns:
            A `ModelTrainingResult` with the consolidated results.

        """

        tuning_config = model_config.tuning
        if tuning_config is None:
            raise ValueError("Tuning requested without tuning parameters")

        candidates = [evaluation.candidate_params for evaluation in evaluations]
        fold_scores_by_candidate = [
            evaluation.fold_scores for evaluation in evaluations
        ]
        fold_results = [
            fold_result
            for evaluation in evaluations
            for fold_result in evaluation.fold_results
        ]

        mean_scores = [float(np.mean(scores)) for scores in fold_scores_by_candidate]
        best_index = int(np.argmax(mean_scores))
        best_params = candidates[best_index]

        logger.info(f"CV Done. Best params: {best_params}")
        base_X_train, base_y_train = _databundle_to_xy_train(data)
        sub_model_results: list[FinalTestMetrics] = []
        final_fit_time = 0.0
        # here we use bohlens method to fit the best model on each fold
        # and predict with each the test data to get more robust evaluations

        try:
            for train_indices, _ in folds:
                trained_sub_model = None
                try:
                    fold_X_train = self._take_rows(base_X_train, train_indices)
                    fold_y_train = self._take_rows(base_y_train, train_indices)

                    trained_sub_model, fit_time = self._fit_model(
                        model_config,
                        model_spec,
                        self._merge_params(model_config.params, best_params),
                        fold_X_train,
                        fold_y_train,
                    )
                    final_fit_time += fit_time

                    sub_model_result = evaluate_trained_model(
                        trained_model=trained_sub_model,
                        data=data,
                        task_type=model_config.task_type,
                    )
                    sub_model_results.append(sub_model_result)
                finally:
                    release_model(trained_sub_model)

            test_metrics = _format_metrics(sub_model_results)

            tuning_result = TuningResult(
                best_params=best_params,
                scoring=tuning_config.scoring,
                test_metrics=test_metrics,
                fold_results=fold_results,
                method=method,
            )

            logger.info(
                f"Model tuning complete in {tuning_result.total_time:.3f}s. "
                f"Best AUROC MIMIC: {tuning_result.test_metrics.mimic_test.mean_roc_auc:.4f}, "
                f"Best AUROC TUDD: {tuning_result.test_metrics.tudd_test.mean_roc_auc:.4f}"
            )

            return ModelTrainingResult(
                model_name=model_config.name,
                task_type=model_config.task_type,
                tuned=True,
                fit_time=final_fit_time,
                tuning_result=tuning_result,
            )

        except Exception:
            raise

    @staticmethod
    def _build_optuna_sampler(tuning):
        import optuna

        seed = tuning.cv.random_state
        if tuning.optuna.sampler == "tpe":
            return optuna.samplers.TPESampler(
                seed=seed,
                n_startup_trials=tuning.optuna.n_startup_trials,
            )
        if tuning.optuna.sampler == "random":
            return optuna.samplers.RandomSampler(seed=seed)

        raise ValueError(f"Unknown Optuna sampler: {tuning.optuna.sampler}")

    @staticmethod
    def _configure_optuna_logging(optuna_module) -> None:
        optuna_module.logging.disable_default_handler()
        optuna_module.logging.enable_propagation()
        optuna_module.logging.set_verbosity(optuna_module.logging.WARNING)

    @staticmethod
    def _count_grid_candidates(grid: dict[str, list[Any]]) -> int:
        if not grid:
            raise ValueError("Tuning requires a non-empty grid")

        candidate_count = 1
        for values in grid.values():
            if not values:
                raise ValueError("Tuning grid values must be non-empty")
            candidate_count *= len(values)
        return candidate_count

    @classmethod
    def _merge_params(
        cls,
        base: dict[str, Any],
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(base)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = cls._merge_params(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _take_rows(data, rows: np.ndarray):
        if hasattr(data, "iloc"):
            return data.iloc[rows]
        return np.asarray(data)[rows]

    @staticmethod
    def _build_cv(model_config: ModelConfig, tuning):
        cv_cls = (
            StratifiedKFold if model_config.task_type == "classification" else KFold
        )
        return cv_cls(
            n_splits=tuning.cv.n_splits,
            shuffle=tuning.cv.shuffle,
            random_state=tuning.cv.random_state,
        )

    def _training_metrics(
        self,
        model_config: ModelConfig,
        model: ModelAdapter,
        X_train,
        y_train,
    ) -> tuple[ClassificationMetrics | None, float]:
        if model_config.task_type != "classification":
            return None, 0.0

        predictions, predict_time = model.predict(X_train)
        return evaluate_classification_predictions(predictions, y_train), predict_time
