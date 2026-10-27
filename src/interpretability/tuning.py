"""Training-only baseline HPO; final refitting and explanations belong to the caller."""

from __future__ import annotations

from dataclasses import asdict
from functools import partial
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import optuna

from src.classes.data_registry import dataset_task_for_target
from src.classes.trainer import Trainer
from src.interpretability import TARGET
from src.schemas.pipeline_schemas import RandomStates
from src.schemas.preprocessing_schemas import ImputerConfig, ScalerEncoderConfig
from src.schemas.training_schemas import (
    ClassificationScoring,
    CrossValidationConfig,
    ModelConfig,
    OptunaConfig,
    TuningConfig,
)
from src.utils.model_registry import get_model_spec
from src.utils.optuna_callbacks import stop_stale_study

if TYPE_CHECKING:
    from src.schemas.dataset_schemas import XYDataset

SAMPLER_SEED = 1337
# A searched XGBoost parameter must not also arrive under its alias.
_XGBOOST_ALIASES = (("eta", "learning_rate"), ("alpha", "reg_alpha"), ("lambda", "reg_lambda"))


def select_parameters(
    model_name: str,
    base_params: dict,
    train: XYDataset,
    seed: int,
    *,
    hpo: dict,
    inference_seed: int,
) -> tuple[dict, dict]:
    """Select EBM/XGBoost kwargs using only the provided source-training subset.

    ``hpo`` supplies a registry ``search_space`` and the Optuna trial counts.
    TPE, five shuffled stratified folds, ROC AUC, and stale-study stopping match
    the baseline experiment. Returned records contain values, never live models.
    """
    started = perf_counter()
    tuning = TuningConfig(
        method="optuna",
        search_space=hpo["search_space"],
        scoring=ClassificationScoring.ROC_AUC,
        cv=CrossValidationConfig(n_splits=5, shuffle=True),
        optuna=OptunaConfig(
            sampler="tpe",
            n_trials=hpo["n_trials"],
            n_startup_trials=hpo["n_startup_trials"],
        ),
    )
    states = RandomStates(
        model_training_seed=seed,
        model_inference_seed=inference_seed,
        cv_split_seed=seed,
        tuning_sampler_seed=SAMPLER_SEED,
        training_sample_seed=seed,
    )
    task_type = dataset_task_for_target(TARGET).task_type
    trainer = Trainer(
        task_type=task_type,
        default_imputer=ImputerConfig(imputation_method="none", flag_missing=False),
        default_scaler=ScalerEncoderConfig(type="none"),
        random_states=states,
    )
    model_config = ModelConfig(name=model_name, tuning=tuning)
    spec = get_model_spec(model_config, task_type)
    search_space = spec.tuning_search_space(tuning.search_space, tuning.grid)
    if model_name == "xgboost":
        base_params = _without_xgboost_aliases(base_params, search_space)
    # Match final-fit numeric conversion without imputing, scaling or adding columns.
    X, y = train.X.astype(float), train.y.to_numpy()
    folds = trainer._build_cv_folds(tuning, X, y)
    trainer._configure_optuna_logging(optuna)
    study = optuna.create_study(
        direction=tuning.scoring.optimization_direction,
        sampler=trainer._build_optuna_sampler(tuning),
    )
    trials: dict[int, dict] = {}

    def objective(trial):
        trial_started = perf_counter()
        candidate = {**base_params, **spec.sample_tuning_candidate(trial, search_space)}
        # The core evaluates and releases each fold model in its own finally block.
        evaluation = trainer._evaluate_cv_candidate(model_config, spec, candidate, trial.number, folds, X, y)
        score = float(np.mean(evaluation.fold_scores))
        trials[trial.number] = {
            "number": trial.number,
            "candidate_params": candidate,
            "fold_scores": evaluation.fold_scores,
            "fold_results": [asdict(fold) for fold in evaluation.fold_results],
            "score": score,
            "seconds": perf_counter() - trial_started,
        }
        return score

    study.optimize(
        objective,
        n_trials=tuning.optuna.n_trials,
        timeout=tuning.optuna.timeout,
        n_jobs=1,
        callbacks=[
            partial(
                stop_stale_study,
                patience=tuning.optuna.patience,
                minimum_trials=tuning.optuna.n_startup_trials + tuning.optuna.patience,
            )
        ],
        gc_after_trial=True,
    )
    best_params = dict(trials[study.best_trial.number]["candidate_params"])
    return best_params, {
        "method": "optuna",
        "search_space": tuning.search_space,
        "config": tuning.model_dump(mode="json"),
        "seeds": states.model_dump(mode="json"),
        "scoring": tuning.scoring.value,
        "best_score": float(study.best_value),
        "best_params": best_params,
        "trials": list(trials.values()),
        "total_seconds": perf_counter() - started,
    }


def _without_xgboost_aliases(base_params: dict, search_space) -> dict:
    """Drop the caller's spelling of a searched XGBoost parameter, and its alias."""
    params = dict(base_params)
    for name, alias in _XGBOOST_ALIASES:
        if name in search_space or alias in search_space:
            params.pop(name, None)
            params.pop(alias, None)
    return params
