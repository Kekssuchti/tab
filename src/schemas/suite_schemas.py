from dataclasses import dataclass
from typing import Any

from pydantic import Field, model_validator

from src.schemas.base_schemas import StrictConfig
from src.schemas.pipeline_schemas import PipelineConfig
from src.schemas.run_records import PipelineRunRecord


class OverrideRangeConfig(StrictConfig):
    """
    Numeric range used to expand suite overrides.

    ---
    Attributes:
        start: int or float
            First value in the range.

        stop: int or float
            Last allowed value in the range.

        step: int or float
            Increment between generated values.
    """

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


class SuiteOverrideConfig(StrictConfig):
    """
    One parameter override in an experiment suite matrix.

    ---
    Attributes:
        path: str
            Dot path to the pipeline config field being changed.

        values: tuple or None, default=None
            Explicit values to test.

        range: OverrideRangeConfig or None, default=None
            Numeric range of values to test.
    """

    path: str
    values: tuple[Any, ...] | None = None
    range: OverrideRangeConfig | None = None

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


class ExperimentSuiteConfig(StrictConfig):
    """
    Configuration for expanding a base pipeline into multiple variants.

    ---
    Attributes:
        name: str
            Suite name.

        base_config: str
            Path to the base pipeline configuration.

        matrix: tuple of SuiteOverrideConfig
            Overrides whose Cartesian product defines the suite variants.
    """

    name: str
    base_config: str
    matrix: tuple[SuiteOverrideConfig, ...] = Field(min_length=1)


@dataclass(frozen=True)
class ExpandedPipelineConfig:
    """
    One expanded pipeline configuration from a suite.

    ---
    Attributes:
        variant_id: str
            Stable identifier for the expanded variant.

        pipeline_config: PipelineConfig
            Materialized pipeline configuration.

        overrides: dict
            Override values applied to the base config.
    """

    variant_id: str
    pipeline_config: PipelineConfig
    overrides: dict[str, Any]


@dataclass(frozen=True)
class SuiteDryRunSummary:
    """
    Dry-run summary for an expanded experiment suite.

    ---
    Attributes:
        suite_name: str
            Suite name.

        config_count: int
            Number of expanded pipeline configurations.

        models_per_config: int
            Number of model runs in each configuration.

        total_model_runs: int
            Total number of model runs across the suite.

        changed_parameters: tuple of str
            Config paths changed by the suite matrix.

        config_variants: tuple of ExpandedPipelineConfig
            Expanded pipeline variants.
    """

    suite_name: str
    config_count: int
    models_per_config: int
    total_model_runs: int
    changed_parameters: tuple[str, ...]
    config_variants: tuple[ExpandedPipelineConfig, ...]

    def format(self) -> str:
        changed = ", ".join(self.changed_parameters)
        lines = [
            f"Suite: {self.suite_name}",
            f"Configs: {self.config_count}",
            f"Models per config: {self.models_per_config}",
            f"Total model runs: {self.total_model_runs}",
            f"Changed parameters: {changed}",
        ]
        for variant in self.config_variants:
            overrides = ", ".join(f"{path}={value}" for path, value in variant.overrides.items())
            lines.append(f"- {variant.variant_id}: {overrides}")
        return "\n".join(lines)


@dataclass(frozen=True)
class SuiteRunResult:
    """
    Result of executing an experiment suite.

    ---
    Attributes:
        suite_name: str
            Suite name.

        results: tuple of PipelineRunRecord
            Pipeline results produced by suite execution.

        summary: SuiteDryRunSummary
            Expansion summary for the executed suite.
    """

    suite_name: str
    results: tuple[PipelineRunRecord, ...]
    summary: SuiteDryRunSummary
