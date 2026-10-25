"""Training-only baseline HPO; final refitting and explanations belong to the caller."""

from __future__ import annotations

from dataclasses import asdict
from functools import partial
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.schemas.dataset_schemas import XYDataset


def select_parameters(model_name: str, base_params: dict, train: XYDataset, seed: int, hpo: dict) -> tuple[dict, dict]:
    """Select EBM/XGBoost kwargs using only the provided source-training subset.

    ``hpo`` supplies ``search_space``, ``n_trials`` and ``n_startup_trials``;
    optional ``patience``, ``timeout`` and ``inference_seed`` default to 10,
    None and 3337. TPE, five shuffled stratified folds and ROC AUC match the
    baseline experiment. Returned records contain values, never live models.
    """
    import numpy as np
    import optuna

    from src.classes.data_registry import dataset_task_for_target
    from src.classes.trainer import Trainer
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

    started = perf_counter()
    if model_name not in ("ebm", "xgboost"):
        raise ValueError("Only EBM and XGBoost are tuned; TabPFN remains fixed")
    if model_name == "ebm":
        base_params = {"interactions": 0, **base_params}
        if base_params["interactions"] != 0:
            raise ValueError("This experiment requires EBM interactions=0")

    tuning = TuningConfig(
        method="optuna",
        search_space=hpo["search_space"],
        scoring=ClassificationScoring.ROC_AUC,
        cv=CrossValidationConfig(n_splits=5, shuffle=True),
        optuna=OptunaConfig(
            sampler="tpe",
            n_trials=hpo["n_trials"],
            n_startup_trials=hpo["n_startup_trials"],
            patience=hpo.get("patience", 10),
            timeout=hpo.get("timeout"),
        ),
    )
    model_config = ModelConfig(name=model_name, tuning=tuning)
    states = RandomStates(
        model_training_seed=seed,
        model_inference_seed=hpo.get("inference_seed", 3337),
        cv_split_seed=seed,
        tuning_sampler_seed=1337,
        training_sample_seed=seed,
    )
    task_type = dataset_task_for_target("mortality").task_type
    trainer = Trainer(
        task_type=task_type,
        default_imputer=ImputerConfig(imputation_method="none", flag_missing=False),
        default_scaler=ScalerEncoderConfig(type="none"),
        random_states=states,
    )
    spec = get_model_spec(model_config, task_type)
    search_space = spec.tuning_search_space(tuning.search_space, tuning.grid)
    if not search_space:
        raise ValueError("Tuning requires a non-empty search space")
    if model_name == "xgboost":
        # A searched eta must also override an extra learning_rate, and vice versa.
        base_params = dict(base_params)
        for name, alias in (("eta", "learning_rate"), ("alpha", "reg_alpha"), ("lambda", "reg_lambda")):
            if name in search_space:
                base_params.pop(alias, None)
            elif alias in search_space:
                base_params.pop(name, None)
    # Match final-fit numeric conversion without imputing, scaling or adding columns.
    X, y = train.X.astype(float), train.y.to_numpy()
    folds = trainer._build_cv_folds(tuning, X, y)
    trainer._configure_optuna_logging(optuna)
    study = optuna.create_study(
        direction=tuning.scoring.optimization_direction,
        sampler=trainer._build_optuna_sampler(tuning),
    )
    trials = []

    def objective(trial):
        trial_started = perf_counter()
        candidate = {**base_params, **spec.sample_tuning_candidate(trial, search_space)}
        # The core evaluates and releases each fold model in its own finally block.
        evaluation = trainer._evaluate_cv_candidate(model_config, spec, candidate, trial.number, folds, X, y)
        score = float(np.mean(evaluation.fold_scores))
        trials.append(
            {
                "number": trial.number,
                "candidate_params": candidate,
                "fold_scores": evaluation.fold_scores,
                "fold_results": [asdict(fold) for fold in evaluation.fold_results],
                "score": score,
                "seconds": perf_counter() - trial_started,
            }
        )
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
    if not trials:
        raise ValueError("Optuna tuning did not complete any trials")
    best_params = dict(trials[study.best_trial.number]["candidate_params"])
    return best_params, {
        "method": "optuna",
        "search_space": tuning.search_space,
        "config": tuning.model_dump(mode="json"),
        "seeds": states.model_dump(mode="json"),
        "scoring": tuning.scoring.value,
        "best_score": float(study.best_value),
        "best_params": best_params,
        "trials": trials,
        "total_seconds": perf_counter() - started,
    }
