from dataclasses import dataclass
from typing import Any

from pydantic import Field, model_validator

from src.schemas.base_schemas import StrictParams
from src.schemas.pipeline_schemas import PipelineParams


class OverrideRange(StrictParams):
    start: int | float
    stop: int | float
    step: int | float = Field(gt=0)

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.stop < self.start:
            raise ValueError("stop must be greater than or equal to start")
        return self

    def values(self) -> tuple[int | float, ...]:
        values = []
        current = self.start
        while current <= self.stop:
            values.append(current)
            current += self.step
        return tuple(values)


class SuiteOverride(StrictParams):
    path: str
    values: tuple[Any, ...] | None = None
    range: OverrideRange | None = None

    @model_validator(mode="after")
    def validate_source(self):
        if bool(self.values) == bool(self.range):
            raise ValueError("Exactly one of values or range must be set")
        if self.values is not None and len(self.values) == 0:
            raise ValueError("values must not be empty")
        return self

    def expanded_values(self) -> tuple[Any, ...]:
        if self.values is not None:
            return self.values
        if self.range is None:
            raise ValueError("Override is missing values or range")
        return self.range.values()


class ExperimentSuiteParams(StrictParams):
    name: str
    base_config: str
    matrix: tuple[SuiteOverride, ...] = Field(min_length=1)


@dataclass(frozen=True)
class ExpandedPipelineConfig:
    variant_id: str
    params: PipelineParams
    overrides: dict[str, Any]


@dataclass(frozen=True)
class SuiteDryRunSummary:
    suite_name: str
    config_count: int
    models_per_config: int
    total_model_runs: int
    changed_parameters: tuple[str, ...]
    variants: tuple[ExpandedPipelineConfig, ...]

    def format(self) -> str:
        changed = ", ".join(self.changed_parameters)
        lines = [
            f"Suite: {self.suite_name}",
            f"Configs: {self.config_count}",
            f"Models per config: {self.models_per_config}",
            f"Total model runs: {self.total_model_runs}",
            f"Changed parameters: {changed}",
        ]
        for variant in self.variants:
            overrides = ", ".join(
                f"{path}={value}" for path, value in variant.overrides.items()
            )
            lines.append(f"- {variant.variant_id}: {overrides}")
        return "\n".join(lines)


@dataclass(frozen=True)
class SuiteRunResult:
    suite_name: str
    results: tuple[Any, ...]
    summary: SuiteDryRunSummary
