from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

from src.schemas.pipeline_schemas import PipelineConfig
from src.schemas.suite_schemas import (
    ExpandedPipelineConfig,
    ExperimentSuiteConfig,
    SuiteDryRunSummary,
)
from src.utils.config_io import load_pipeline_config


class ExperimentSuite:
    def __init__(
        self, experiment_suite_config: ExperimentSuiteConfig, suite_path: str | Path
    ):
        self.experiment_suite_config = experiment_suite_config
        self.suite_path = Path(suite_path)

    def expand(self) -> tuple[ExpandedPipelineConfig, ...]:
        pipeline_config = load_pipeline_config(self._base_config_path())
        variants = []
        override_paths = tuple(
            override.path for override in self.experiment_suite_config.matrix
        )
        value_sets = tuple(
            override.expanded_values()
            for override in self.experiment_suite_config.matrix
        )

        for index, values in enumerate(product(*value_sets)):
            overrides = dict(zip(override_paths, values, strict=True))
            variant_id = self._variant_id(index, overrides)
            params = self._apply_overrides(pipeline_config, overrides)
            params.run_id = f"{pipeline_config.run_id}_{self.experiment_suite_config.name}_{variant_id}"
            params.mlflow.run_name = self._run_name(pipeline_config, variant_id)
            variants.append(
                ExpandedPipelineConfig(
                    variant_id=variant_id,
                    pipeline_config=params,
                    overrides=overrides,
                )
            )

        return tuple(variants)

    def dry_run_summary(self) -> SuiteDryRunSummary:
        variants = self.expand()
        models_per_config = len(variants[0].pipeline_config.training) if variants else 0
        return SuiteDryRunSummary(
            suite_name=self.experiment_suite_config.name,
            config_count=len(variants),
            models_per_config=models_per_config,
            total_model_runs=len(variants) * models_per_config,
            changed_parameters=tuple(
                override.path for override in self.experiment_suite_config.matrix
            ),
            config_variants=variants,
        )

    def _base_config_path(self) -> Path:
        base_config = Path(self.experiment_suite_config.base_config)
        if base_config.is_absolute():
            return base_config
        return self.suite_path.parent / base_config

    def _apply_overrides(
        self,
        pipeline_config: PipelineConfig,
        overrides: dict[str, Any],
    ) -> PipelineConfig:
        data = deepcopy(pipeline_config.model_dump(mode="json"))
        for path, value in overrides.items():
            _set_path_value(data, path, value)
        return PipelineConfig.model_validate(data)

    def _run_name(self, base_params: PipelineConfig, variant_id: str) -> str:
        base_name = base_params.mlflow.run_name or self.experiment_suite_config.name
        return f"{base_name}/{variant_id}"

    @staticmethod
    def _variant_id(index: int, overrides: dict[str, Any]) -> str:
        if len(overrides) == 1:
            path, value = next(iter(overrides.items()))
            return _slug(f"{path.split('.')[-1]}-{value}")
        return f"v{index:03d}"


def _set_path_value(data: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    if any(part == "" for part in parts):
        raise ValueError(f"Invalid override path '{path}'")

    current = data
    for part in parts[:-1]:
        current = _get_child(current, part, path)

    _set_child(current, parts[-1], value, path)


def _get_child(current: Any, part: str, path: str) -> Any:
    if isinstance(current, dict):
        if part not in current:
            raise ValueError(f"Unknown override path '{path}'")
        return current[part]
    if isinstance(current, list):
        index = _list_index(part, path)
        try:
            return current[index]
        except IndexError as exc:
            raise ValueError(f"Override path index out of range '{path}'") from exc
    raise ValueError(f"Cannot traverse override path '{path}'")


def _set_child(current: Any, part: str, value: Any, path: str) -> None:
    if isinstance(current, dict):
        if part not in current:
            raise ValueError(f"Unknown override path '{path}'")
        current[part] = value
        return
    if isinstance(current, list):
        index = _list_index(part, path)
        try:
            current[index] = value
            return
        except IndexError as exc:
            raise ValueError(f"Override path index out of range '{path}'") from exc
    raise ValueError(f"Cannot set override path '{path}'")


def _list_index(part: str, path: str) -> int:
    try:
        return int(part)
    except ValueError as exc:
        raise ValueError(f"Expected list index in override path '{path}'") from exc


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value).strip("-")
