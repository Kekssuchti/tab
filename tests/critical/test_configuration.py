"""Configuration risks not proved by E2Es with reduced model/search budgets."""

from functools import partial

import optuna
import pytest

from src.schemas.training_schemas import ModelConfig
from src.utils.model_registry import ModelSpec, get_model_spec
from src.utils.optuna_callbacks import stop_stale_study
from src.utils.tuning_distributions import DiscreteUniform, IntUniform, LogUniform, UniformChoice


def test_explicit_nested_grid_replaces_named_space_without_merging_or_aliasing_candidates():
    """Dotted TFM settings must reach one nested config, without stale named-space dimensions."""
    spec = get_model_spec(ModelConfig(name="tabpfn-3"), "classification")
    grid = {
        "inference_config.SUBSAMPLE_SAMPLES": [32, 48],
        "inference_config.PREPROCESS_TRANSFORMS.name": ["power"],
        "inference_config.PREPROCESS_TRANSFORMS.categorical_name": ["none"],
    }
    candidates = spec.tuning_candidates("good", grid)
    assert candidates == [
        {
            "inference_config": {
                "SUBSAMPLE_SAMPLES": size,
                "PREPROCESS_TRANSFORMS": {"name": "power", "categorical_name": "none"},
            }
        }
        for size in (32, 48)
    ]
    candidates[0]["inference_config"]["PREPROCESS_TRANSFORMS"]["name"] = "changed"
    assert candidates[1]["inference_config"]["PREPROCESS_TRANSFORMS"]["name"] == "power"
    assert grid["inference_config.PREPROCESS_TRANSFORMS.name"] == ["power"]


def test_optuna_conditional_domains_convert_to_nested_adapter_parameters():
    """A constant-or-distribution choice must not sample the inactive branch or leak selector keys."""
    spec = ModelSpec(
        adapter_path="unused:Adapter",
        search_spaces={
            "mixed": {
                "inference_config.SUBSAMPLE_SAMPLES": UniformChoice(None, DiscreteUniform(0.1, 0.9, 0.2)),
                "C": LogUniform(0.01, 10),
                "n_estimators": IntUniform(1, 3, 1),
                "solver": ["lbfgs", "saga"],
            }
        },
    )
    space = spec.tuning_search_space("mixed", None)
    selector = "inference_config.SUBSAMPLE_SAMPLES.__uniform_choice"
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=7))
    study.enqueue_trial({selector: 0, "C": 0.1, "n_estimators": 1, "solver": "lbfgs"})
    study.enqueue_trial({selector: 1, f"{selector}_1": 0.5, "C": 2.0, "n_estimators": 3, "solver": "saga"})
    candidates = []

    def objective(trial):
        candidates.append(spec.sample_tuning_candidate(trial, space))
        return 0.0

    study.optimize(objective, n_trials=2)
    assert candidates == [
        {"inference_config": {"SUBSAMPLE_SAMPLES": None}, "C": 0.1, "n_estimators": 1, "solver": "lbfgs"},
        {"inference_config": {"SUBSAMPLE_SAMPLES": 0.5}, "C": 2.0, "n_estimators": 3, "solver": "saga"},
    ]
    assert f"{selector}_1" not in study.trials[0].params
    assert study.trials[1].distributions[f"{selector}_1"] == optuna.distributions.FloatDistribution(0.1, 0.9, step=0.2)
    assert study.trials[1].distributions["C"] == optuna.distributions.FloatDistribution(0.01, 10, log=True)
    assert study.trials[1].distributions["n_estimators"] == optuna.distributions.IntDistribution(1, 3)


def test_grid_search_rejects_continuous_domains_instead_of_silently_discretizing():
    spec = get_model_spec(ModelConfig(name="logistic-regression"), "classification")
    with pytest.raises(ValueError, match="Grid tuning cannot expand distribution domains"):
        spec.tuning_candidates("good", None)


def test_nested_grid_rejects_parent_child_collision_instead_of_overwriting_settings():
    spec = get_model_spec(ModelConfig(name="tabpfn-3"), "classification")
    with pytest.raises(ValueError, match="conflicts with non-nested parameter"):
        spec.tuning_candidates(None, {"inference_config": ["auto"], "inference_config.SUBSAMPLE_SAMPLES": [32]})


@pytest.mark.parametrize("direction,sign", [("maximize", 1), ("minimize", -1)])
def test_early_stop_resets_on_improvement_and_counts_completed_stale_trials_only(direction, sign):
    """Failed/pruned trials do not spend patience, whereas equal scores do."""
    study = optuna.create_study(direction=direction)

    def objective(trial):
        if trial.number in (1, 6):
            raise optuna.TrialPruned()
        if trial.number == 2:
            raise RuntimeError("synthetic failed candidate")
        score = 2.0 if trial.number in (4, 5) else 1.0
        return sign * score

    study.optimize(
        objective,
        n_trials=20,
        catch=(RuntimeError,),
        callbacks=[partial(stop_stale_study, patience=3)],
    )
    assert [trial.number for trial in study.trials] == list(range(9))
    assert study.best_trial.number == 4
    assert study.best_value == sign * 2.0


def test_early_stop_preserves_startup_budget_even_when_patience_is_already_exhausted():
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: 1.0,
        n_trials=20,
        callbacks=[partial(stop_stale_study, patience=3, minimum_trials=8)],
    )
    assert len(study.trials) == 8
