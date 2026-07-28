from dataclasses import dataclass
from functools import partial
from timeit import default_timer as timer
from typing import Any

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from src.classes.preprocessor import Preprocessor
from src.config import config
from src.interfaces.model_interface import ModelAdapter, PreprocessedModelAdapter
from src.schemas.base_schemas import TaskType
from src.schemas.dataset_schemas import DatasetBundle
from src.schemas.metrics import FinalTestMetrics
from src.schemas.preprocessing_schemas import ImputerConfig, ScalerEncoderConfig
from src.schemas.run_records import FoldRecord, ModelTrainingResult, TuningRecord
from src.schemas.training_schemas import ModelConfig, TuningMethod
from src.utils.evaluation import evaluate_trained_model, evaluate_trained_model_bootstrap
from src.utils.evaluation_utils import (
    aggregate_final_test_metrics,
    classification_score,
    evaluate_classification_predictions,
)
from src.utils.logger import logger
from src.utils.model_lifecycle import release_model
from src.utils.model_registry import ModelSpec, get_model_spec
from src.utils.optuna_callbacks import (
    stop_stale_study,
)


@dataclass
class _CandidateEvaluation:
    """In-memory tuning results for one hyperparameter candidate."""

    candidate_params: dict[str, Any]
    fold_scores: list[float]
    fold_results: list[FoldRecord]


class Trainer:
    """Train, tune, evaluate, and release configured model adapters."""

    def __init__(
        self,
        task_type: TaskType,
        default_imputer: ImputerConfig,
        default_scaler: ScalerEncoderConfig,
    ) -> None:
        self.task_type = task_type
        self.default_imputer = default_imputer
        self.default_scaler = default_scaler

    def train_evaluate_model(self, model_config: ModelConfig, data: DatasetBundle) -> ModelTrainingResult:
        logger.info(f"Training model: {model_config.name}")

        imputer, scaler = self._resolved_preprocessing(model_config)
        logger.info(f"data imputation via: {imputer.imputation_method}")
        logger.info(f"scaling data using: {scaler.type}")

        spec = get_model_spec(model_config, self.task_type)
        return self._tune_model(model_config, spec, data)

    def _fit_model(
        self,
        model_config: ModelConfig,
        model_spec: ModelSpec,
        model_params: dict[str, Any],
        X_train,
        y_train,
    ) -> tuple[ModelAdapter, float]:
        adapter = model_spec.create(self.task_type, model_params)
        model = PreprocessedModelAdapter(
            adapter,
            self._build_preprocess_pipeline(model_config),
        )
        try:
            fit_time = model.fit(X_train, y_train)
            return model, fit_time
        except Exception:
            release_model(model)
            raise

    def _build_preprocess_pipeline(self, model_params: ModelConfig):
        imputer, scaler = self._resolved_preprocessing(model_params)
        return Preprocessor(
            imputer_config=imputer,
            scaler_config=scaler,
        ).build_pipeline()

    def _resolved_preprocessing(self, model_config: ModelConfig) -> tuple[ImputerConfig, ScalerEncoderConfig]:
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

    def _tune_model(self, model_config: ModelConfig, model_spec: ModelSpec, data: DatasetBundle) -> ModelTrainingResult:
        tuning = model_config.tuning

        if tuning.method == "grid":
            return self._tune_model_grid(model_config, model_spec, data)
        if tuning.method == "optuna":
            return self._tune_model_optuna(model_config, model_spec, data)

        raise ValueError(f"Unknown tuning method: {tuning.method}")

    def _tune_model_grid(
        self, model_config: ModelConfig, model_spec: ModelSpec, data: DatasetBundle
    ) -> ModelTrainingResult:
        tuning = model_config.tuning
        X_train, y_train = _training_data(data)

        candidates = model_spec.tuning_candidates(tuning.search_space, tuning.grid)
        folds = list(self._build_cv(tuning).split(X_train, y_train))
        logger.info(
            f"Tuning {model_config.name}: candidates={len(candidates)} "
            f"folds={len(folds)} scoring={tuning.scoring} method=grid"
        )

        evaluations: list[_CandidateEvaluation] = []

        if len(candidates) == 1:
            # skip initial cv, just pass params as best params
            best_params = candidates[0]
            logger.info(f"Only 1 set of params given, cv tuning skipped: {best_params}")

            evaluation = _CandidateEvaluation(
                candidate_params=best_params,
                fold_scores=[],
                fold_results=[],
            )
            evaluations.append(evaluation)
        else:
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
        search_space = model_spec.tuning_search_space(tuning.search_space, tuning.grid)
        if not search_space:
            raise ValueError("Tuning requires a non-empty search space")
        X_train, y_train = _training_data(data)

        folds = list(self._build_cv(tuning).split(X_train, y_train))

        logger.info(
            f"Tuning {model_config.name}: trials={tuning.optuna.n_trials} "
            f"dimensions={len(search_space)} folds={len(folds)} "
            f"scoring={tuning.scoring} method=optuna"
        )

        evaluations: list[_CandidateEvaluation] = []

        study = optuna.create_study(
            direction="maximize",
            sampler=self._build_optuna_sampler(tuning),
        )

        def objective(trial):
            candidate_index = len(evaluations)
            candidate_params = model_spec.sample_tuning_candidate(trial, search_space)
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
            callbacks=[
                partial(
                    stop_stale_study,
                    patience=tuning.optuna.patience,
                    minimum_trials=(tuning.optuna.n_startup_trials + tuning.optuna.patience),
                )
            ],
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
        candidate_scores: list[float] = []
        fold_results: list[FoldRecord] = []

        for fold_index, (train_index, validation_index) in enumerate(folds):
            fold_start = timer()
            fold_model = None
            prediction = None
            try:
                fold_model, _ = self._fit_model(
                    model_config,
                    model_spec,
                    candidate_params,
                    self._take_rows(X_train, train_index),
                    self._take_rows(y_train, train_index),
                )
                prediction = fold_model.predict(self._take_rows(X_train, validation_index))
                if self.task_type != "classification":
                    raise NotImplementedError("Regression tuning metrics are not implemented yet")

                metrics = evaluate_classification_predictions(
                    prediction.values,
                    self._take_rows(y_train, validation_index),
                )

                candidate_scores.append(classification_score(metrics, tuning.scoring))

                fold_results.append(
                    FoldRecord(
                        candidate_index=candidate_index,
                        fold_index=fold_index,
                        metrics=metrics,
                        time=timer() - fold_start,
                        model_params=candidate_params,
                    )
                )
            finally:
                prediction = None
                release_model(fold_model)

        return _CandidateEvaluation(
            candidate_params=candidate_params,
            fold_scores=candidate_scores,
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
        method: TuningMethod,
    ) -> ModelTrainingResult:
        """
        Fits the best found params for a model to each fold and evaluates it against the common test sets.
        Results get consolidated into a single `ModelTrainingResult`.
        This includes mean scores for all metrics and their 95% confidence intervals.

        Args:
            model_config: The model configuration to fit.
            model_spec: The model spec to use for fitting.
            data: The dataset bundle to use for training and evaluation.
            evaluations: The candidate evaluations to consolidate.
            folds: The folds used to find the best parameters.
            method: The tuning method to use.

        Returns:
            A `ModelTrainingResult` with the consolidated results.

        """

        tuning_config = model_config.tuning

        fold_results = [fold_result for evaluation in evaluations for fold_result in evaluation.fold_results]

        candidates = [evaluation.candidate_params for evaluation in evaluations]
        if len(candidates) == 1:
            best_params = candidates[0]
        else:
            fold_scores_by_candidate = [evaluation.fold_scores for evaluation in evaluations]
            mean_scores = [float(np.mean(scores)) for scores in fold_scores_by_candidate]
            best_index = int(np.argmax(mean_scores))
            best_params = candidates[best_index]

        logger.info(f"CV Done. Best params: {best_params}")
        base_X_train, base_y_train = _training_data(data)
        sub_model_results: list[FinalTestMetrics] = []
        final_fit_time = 0.0
        # here we use bohlens method to fit the best model on each fold
        # and predict with each the test data to get more robust evaluations

        if config.eval_bootstrap:
            trained_model = None
            try:
                trained_model, fit_time = self._fit_model(
                    model_config,
                    model_spec,
                    best_params,
                    base_X_train,
                    base_y_train,
                )
                final_fit_time = fit_time
                test_metrics = evaluate_trained_model_bootstrap(
                    trained_model=trained_model,
                    data=data,
                    task_type=self.task_type,
                )
            finally:
                release_model(trained_model)
        else:
            for train_indices, _ in folds:
                trained_sub_model = None
                try:
                    fold_X_train = self._take_rows(base_X_train, train_indices)
                    fold_y_train = self._take_rows(base_y_train, train_indices)

                    trained_sub_model, fit_time = self._fit_model(
                        model_config,
                        model_spec,
                        best_params,
                        fold_X_train,
                        fold_y_train,
                    )
                    final_fit_time += fit_time

                    sub_model_result = evaluate_trained_model(
                        trained_model=trained_sub_model,
                        data=data,
                        task_type=self.task_type,
                    )
                    sub_model_results.append(sub_model_result)
                finally:
                    release_model(trained_sub_model)

            test_metrics = aggregate_final_test_metrics(sub_model_results)

        tuning_result = TuningRecord(
            best_params=best_params,
            scoring=tuning_config.scoring,
            final_test_metrics=test_metrics,
            fold_results=fold_results,
            method=method,
        )

        mimic_metrics = tuning_result.final_test_metrics.mimic_test.metrics
        tudd_metrics = tuning_result.final_test_metrics.tudd_test.metrics
        if self.task_type == "classification":
            logger.info(
                f"Model tuning complete in {tuning_result.total_time:.3f}s. "
                f"Best AUROC MIMIC: {mimic_metrics.roc_auc:.4f}, "
                f"Best AUROC TUDD: {tudd_metrics.roc_auc:.4f}"
            )
        else:
            logger.info(
                f"Model tuning complete in {tuning_result.total_time:.3f}s. "
                f"Best R2 MIMIC: {mimic_metrics.r2:.4f}, Best R2 TUDD: {tudd_metrics.r2:.4f}"
            )

        return ModelTrainingResult(
            model_name=model_config.name,
            task_type=self.task_type,
            tuned=True,
            fit_time=final_fit_time,
            tuning_result=tuning_result,
        )

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
    def _take_rows(data, rows: np.ndarray):
        if hasattr(data, "iloc"):
            return data.iloc[rows]
        return np.asarray(data)[rows]

    def _build_cv(self, tuning):
        cv_cls = StratifiedKFold if self.task_type == "classification" else KFold
        return cv_cls(
            n_splits=tuning.cv.n_splits,
            shuffle=tuning.cv.shuffle,
            random_state=tuning.cv.random_state,
        )


def _training_data(data: DatasetBundle):
    return data.train_data.X, data.train_data.y.to_numpy()
