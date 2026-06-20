from timeit import default_timer as timer

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.schemas.training_schemas import (
    FoldResult,
    HPOParams,
    HPOResult,
    ModelParams,
    ModelTrainingResult,
    TrainingParams,
)
from src.utils.logger import logger
from src.utils.model_registry import MODEL_REGISTRY_CLS, MODEL_REGISTRY_REG


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

        try:
            model_cls = registry[model_params.name]
        except KeyError as exc:
            available = ", ".join(sorted(registry))
            raise ValueError(
                f"Unknown {model_params.task_type} model '{model_params.name}'. "
                f"Available models: {available}"
            ) from exc

        adapter = model_cls(task_type=model_params.task_type, **model_params.params)

        if model_params.optimize_hyperparameters:
            return self._train_with_hpo(
                model_params,
                adapter,
                X_train,
                y_train,
            )
        else:
            return self._train_without_hpo(
                model_params,
                adapter,
                X_train,
                y_train,
            )

    def _train_without_hpo(
        self,
        model_params: ModelParams,
        adapter,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> ModelTrainingResult:
        pipeline = Pipeline([*self.preprocess_pipeline.steps, ("model", adapter.model)])

        start = timer()
        pipeline.fit(X_train, y_train)
        fit_time = timer() - start

        adapter.model = pipeline.named_steps["model"]

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
        adapter,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> ModelTrainingResult:
        hpo_params = model_params.hyperparameter_optimization_params
        assert hpo_params is not None

        pipeline = Pipeline([*self.preprocess_pipeline.steps, ("model", adapter.model)])

        param_grid = {
            f"model__{key}": value for key, value in hpo_params.search_grid.items()
        }

        cv = StratifiedKFold(
            n_splits=hpo_params.cv.n_splits,
            shuffle=hpo_params.cv.shuffle,
            random_state=hpo_params.cv.random_state,
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=hpo_params.scoring,
            cv=cv,
            return_train_score=True,
            n_jobs=-1,
        )

        start = timer()
        search.fit(X_train, y_train)
        fit_time = timer() - start

        adapter.model = search.best_estimator_.named_steps["model"]

        fold_results = self._extract_fold_results(search, hpo_params)

        hpo_result = HPOResult(
            best_params=search.best_params_,
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
