from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isclose, isfinite
from typing import Any


class OptunaDistribution(ABC):
    """A registry-defined domain sampled through an Optuna trial."""

    @abstractmethod
    def suggest(self, trial, name: str) -> Any:
        """Sample this domain for one trial."""


@dataclass(frozen=True)
class Uniform(OptunaDistribution):
    low: float
    high: float

    def __post_init__(self) -> None:
        _validate_bounds(self.low, self.high)

    def suggest(self, trial, name: str) -> float:
        return trial.suggest_float(name, self.low, self.high)


@dataclass(frozen=True)
class DiscreteUniform(OptunaDistribution):
    low: float
    high: float
    step: float

    def __post_init__(self) -> None:
        _validate_bounds(self.low, self.high)
        if not isfinite(self.step) or self.step <= 0:
            raise ValueError(
                "DiscreteUniform step must be finite and greater than zero"
            )

        number_of_steps = (self.high - self.low) / self.step
        if not isclose(number_of_steps, round(number_of_steps)):
            raise ValueError(
                "DiscreteUniform range must be evenly divisible by its step"
            )

    def suggest(self, trial, name: str) -> float:
        return trial.suggest_float(name, self.low, self.high, step=self.step)


@dataclass(frozen=True)
class IntUniform(OptunaDistribution):
    low: int
    high: int
    step: int

    def __post_init__(self) -> None:
        _validate_bounds(self.low, self.high)
        if not isfinite(self.step) or self.step <= 0:
            raise ValueError(
                "DiscreteUniform step must be finite and greater than zero"
            )

        number_of_steps = (self.high - self.low) / self.step
        if not isclose(number_of_steps, round(number_of_steps)):
            raise ValueError(
                "DiscreteUniform range must be evenly divisible by its step"
            )

    def suggest(self, trial, name: str) -> int:
        return trial.suggest_int(name, self.low, self.high, step=self.step)


@dataclass(frozen=True)
class LogUniform(OptunaDistribution):
    low: float
    high: float

    def __post_init__(self) -> None:
        _validate_bounds(self.low, self.high)
        if self.low <= 0:
            raise ValueError("LogUniform bounds must be greater than zero")

    def suggest(self, trial, name: str) -> float:
        return trial.suggest_float(name, self.low, self.high, log=True)


@dataclass(frozen=True, init=False)
class UniformChoice(OptunaDistribution):
    choices: tuple[Any, ...]

    def __init__(self, *choices: Any) -> None:
        if not choices:
            raise ValueError("UniformChoice requires at least one choice")
        object.__setattr__(self, "choices", choices)

    def suggest(self, trial, name: str) -> Any:
        choice_index = trial.suggest_categorical(
            f"{name}.__uniform_choice",
            range(len(self.choices)),
        )
        choice = self.choices[choice_index]
        if isinstance(choice, OptunaDistribution):
            return choice.suggest(trial, f"{name}.__uniform_choice_{choice_index}")
        return choice


def _validate_bounds(low: Any, high: Any) -> None:
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        raise TypeError("Distribution bounds must be numeric")
    if not isfinite(low) or not isfinite(high):
        raise ValueError("Distribution bounds must be finite")
    if low >= high:
        raise ValueError("Distribution low must be less than high")
