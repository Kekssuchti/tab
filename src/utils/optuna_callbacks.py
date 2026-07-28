from typing import Any


def stop_stale_study(
    study: Any,
    trial: Any,
    patience: int,
    minimum_trials: int = 0,
) -> None:
    """Stop after ``patience`` completed trials without a new best score."""
    if minimum_trials < 0:
        raise ValueError("Early stopping minimum trials cannot be negative")
    if trial.value is None:
        return

    completed_trials = [
        completed_trial for completed_trial in study.get_trials(deepcopy=False) if completed_trial.value is not None
    ]
    if len(completed_trials) < minimum_trials:
        return

    best_trial_number = study.best_trial.number
    completed_since_best = sum(completed_trial.number > best_trial_number for completed_trial in completed_trials)
    if completed_since_best >= patience:
        study.stop()
