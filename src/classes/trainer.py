from timeit import default_timer as timer

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.interfaces.model_interface import ModelAdapter, PreprocessedModelAdapter
from src.schemas.training_schemas import (
    FoldResult,
    HPOParams,
    HPOResult,
    ModelParams,
    ModelTrainingResult,
    TrainingParams,
)
from src.utils.logger import logger
from src.utils.model_registry import MODEL_REGISTRY_CLS, MODEL_REGISTRY_REG, ModelSpec

SCORING_ALIASES = {
    "prc_auc": "average_precision",
    "sensitivity": "recall",
}


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
        registry = (
            MODEL_REGISTRY_CLS
            if model_params.task_type == "classification"
            else MODEL_REGISTRY_REG
        )

        spec = self._get_model_spec(model_params, registry)
        adapter = spec.create(model_params.task_type, model_params.params)

        if model_params.hpo is not None:
            return self._train_with_hpo(
                model_params,
                spec,
                adapter,
                X_train,
                y_train,
            )
        else:
            return self._train_without_hpo(
                model_params,
                spec,
                adapter,
                X_train,
                y_train,
            )

    @staticmethod
    def _get_model_spec(
        model_params: ModelParams, registry: dict[str, ModelSpec]
    ) -> ModelSpec:
        try:
            return registry[model_params.name]
        except KeyError as exc:
            available = ", ".join(sorted(registry))
            raise ValueError(
                f"Unknown {model_params.task_type} model '{model_params.name}'. "
                f"Available models: {available}"
            ) from exc

    def _train_without_hpo(
        self,
        model_params: ModelParams,
        spec: ModelSpec,
        adapter: ModelAdapter,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> ModelTrainingResult:
        if not spec.supports_sklearn_pipeline:
            wrapped_adapter = PreprocessedModelAdapter(
                adapter, clone(self.preprocess_pipeline)
            )
            fit_time = wrapped_adapter.fit(X_train, y_train)

            logger.info(
                f"Model {model_params.name} trained without HPO in {fit_time:.3f}s"
            )

            return ModelTrainingResult(
                model_name=model_params.name,
                task_type=model_params.task_type,
                trained_model=wrapped_adapter,
                optimized_hyperparameters=False,
                fit_time=fit_time,
            )

        pipeline = Pipeline(
            [
                *clone(self.preprocess_pipeline).steps,
                ("model", adapter.estimator_for_training()),
            ]
        )

        start = timer()
        pipeline.fit(X_train, y_train)
        fit_time = timer() - start

        adapter.set_trained_estimator(pipeline)

        logger.info(f"Model {model_params.name} trained without HPO in {fit_time:.3f}s")

        return ModelTrainingResult(
            model_name=model_params.name,
            task_type=model_params.task_type,
            trained_model=adapter,
            optimized_hyperparameters=False,
            fit_time=fit_time,
        )

    def _train_with_hpo(
        self,
        model_params: ModelParams,
        spec: ModelSpec,
        adapter: ModelAdapter,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> ModelTrainingResult:
        hpo_params = model_params.hpo
        if hpo_params is None:
            raise ValueError("HPO requested without HPO parameters")
        if not spec.supports_sklearn_pipeline:
            raise ValueError(
                f"Model '{model_params.name}' does not support sklearn HPO"
            )

        pipeline = Pipeline(
            [
                *clone(self.preprocess_pipeline).steps,
                ("model", adapter.estimator_for_training()),
            ]
        )

        param_grid = {
            f"model__{key}": value
            for key, value in spec.search_grid(
                hpo_params.search_space, hpo_params.search_grid
            ).items()
        }

        cv = self._build_cv(model_params, hpo_params)

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=SCORING_ALIASES.get(hpo_params.scoring, hpo_params.scoring),
            cv=cv,
            return_train_score=True,
            n_jobs=-1,
        )

        start = timer()
        search.fit(X_train, y_train)
        fit_time = timer() - start

        adapter.set_trained_estimator(search.best_estimator_)

        fold_results = self._extract_fold_results(search, hpo_params)

        hpo_result = HPOResult(
            best_params=self._strip_model_prefix(search.best_params_),
            best_score=search.best_score_,
            scoring=hpo_params.scoring,
            cv_results=search.cv_results_,
            fold_results=fold_results,
        )

        logger.info(
            f"Model {model_params.name} HPO complete in {fit_time:.3f}s. "
            f"Best {hpo_params.scoring}: {search.best_score_:.4f}"
        )

        return ModelTrainingResult(
            model_name=model_params.name,
            task_type=model_params.task_type,
            trained_model=adapter,
            optimized_hyperparameters=True,
            fit_time=fit_time,
            hpo_result=hpo_result,
        )

    @staticmethod
    def _build_cv(model_params: ModelParams, hpo_params: HPOParams):
        cv_cls = (
            StratifiedKFold if model_params.task_type == "classification" else KFold
        )
        return cv_cls(
            n_splits=hpo_params.cv.n_splits,
            shuffle=hpo_params.cv.shuffle,
            random_state=hpo_params.cv.random_state,
        )

    @staticmethod
    def _strip_model_prefix(params: dict[str, object]) -> dict[str, object]:
        return {key.removeprefix("model__"): value for key, value in params.items()}

    @staticmethod
    def _extract_fold_results(
        search: GridSearchCV, hpo_params: HPOParams
    ) -> list[FoldResult]:
        best_index = search.best_index_
        n_splits = hpo_params.cv.n_splits
        fold_results: list[FoldResult] = []

        for fold_idx in range(n_splits):
            train_key = f"split{fold_idx}_train_score"
            test_key = f"split{fold_idx}_test_score"

            fold_results.append(
                FoldResult(
                    fold_index=fold_idx,
                    train_score=float(search.cv_results_[train_key][best_index]),
                    test_score=float(search.cv_results_[test_key][best_index]),
                )
            )

        return fold_results
