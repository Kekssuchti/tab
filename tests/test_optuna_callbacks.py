from functools import partial

import optuna
import pytest

from src.utils.optuna_callbacks import stop_stale_study


def test_stop_stale_study_uses_default_patience():
    study = optuna.create_study(direction="maximize")

    study.optimize(
        lambda trial: 1.0,
        n_trials=100,
        callbacks=[partial(stop_stale_study, patience=10)],
    )

    assert len(study.trials) == 11


def test_stop_stale_study_resets_patience_after_improvement():
    scores = [1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]
    study = optuna.create_study(direction="maximize")

    study.optimize(
        lambda trial: scores[trial.number],
        n_trials=len(scores),
        callbacks=[partial(stop_stale_study, patience=3)],
    )

    assert len(study.trials) == 7
    assert study.best_trial.number == 3


def test_stop_stale_study_waits_for_minimum_trials():
    study = optuna.create_study(direction="maximize")

    study.optimize(
        lambda trial: 1.0,
        n_trials=100,
        callbacks=[partial(stop_stale_study, patience=3, minimum_trials=8)],
    )

    assert len(study.trials) == 8
