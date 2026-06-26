from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import import_module
from itertools import product
from typing import TYPE_CHECKING, Any

from src.schemas.base_schemas import TaskType
from src.schemas.training_schemas import ModelParams
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.interfaces.model_interface import ModelAdapter
from tabpfn.classifier import ClassifierEvalMetrics

SKLEARN_ADAPTER = "src.adapter.sklearn_adapter"
TABPFN_ADAPTER = "src.adapter.tabpfn_adapter:TabPFNAdapter"
TABICL_ADAPTER = "src.adapter.tabicl_adapter:TabICLAdapter"
LIMIX_ADAPTER = "src.adapter.limix_adapter:LimixAdapter"
MITRA_ADAPTER = "src.adapter.mitra_adapter:MitraAdapter"
ORION_MSP_ADAPTER = "src.adapter.orion_msp_adapter:OrionMSPAdapter"
ORION_BIX_ADAPTER = "src.adapter.orion_bix_adapter:OrionBixAdapter"


@dataclass(frozen=True)
class ModelSpec:
    adapter_path: str
    default_params: dict[str, Any] = field(default_factory=dict)
    search_spaces: Mapping[str, Mapping[str, Sequence[Any]]] = field(
        default_factory=dict
    )

    def create(self, task_type: TaskType, params: dict[str, Any]) -> "ModelAdapter":
        adapter_cls = _load_adapter_cls(self.adapter_path)
        return adapter_cls(task_type=task_type, **{**self.default_params, **params})

    def tuning_grid(
        self, search_space: str | None, overrides: dict[str, list[Any]] | None
    ) -> dict[str, list[Any]]:
        if overrides is not None:
            return _copy_grid(overrides)

        if search_space is None:
            search_space = "default"
        try:
            return _copy_grid(self.search_spaces[search_space])
        except KeyError as exc:
            available = ", ".join(sorted(self.search_spaces)) or "none"
            raise ValueError(
                f"Unknown tuning search space '{search_space}'. Available: {available}"
            ) from exc

    def tuning_candidates(
        self, search_space: str | None, overrides: dict[str, list[Any]] | None
    ) -> list[dict[str, Any]]:
        return _expand_grid(self.tuning_grid(search_space, overrides))


@dataclass(frozen=True)
class ModelCatalog:
    registries: Mapping[TaskType, Mapping[str, ModelSpec]]

    def spec_for(self, model_params: ModelParams) -> ModelSpec:
        registry = self.registries[model_params.task_type]

        try:
            return registry[model_params.name]
        except KeyError as exc:
            available = ", ".join(sorted(registry))
            raise ValueError(
                f"Unknown {model_params.task_type} model '{model_params.name}'. "
                f"Available models: {available}"
            ) from exc

    def available_models(self, task_type: TaskType) -> tuple[str, ...]:
        return tuple(sorted(self.registries[task_type]))


def get_model_spec(model_params: ModelParams) -> ModelSpec:
    return MODEL_CATALOG.spec_for(model_params)


def _copy_grid(grid: Mapping[str, Sequence[Any]]) -> dict[str, list[Any]]:
    return {key: list(values) for key, values in grid.items()}


def _expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        raise ValueError("Tuning requires a non-empty grid")

    keys = list(grid)
    values = [grid[key] for key in keys]
    if any(not value for value in values):
        raise ValueError("Tuning grid values must be non-empty")

    return [dict(zip(keys, combination)) for combination in product(*values)]


@lru_cache
def _load_adapter_cls(adapter_path: str):
    module_name, class_name = adapter_path.split(":", maxsplit=1)
    logger.info(f"Loading model adapter: {adapter_path}")
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Could not import model adapter module '{module_name}'"
        ) from exc
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Model adapter '{class_name}' not found in '{module_name}'"
        ) from exc


CLASSIFICATION_SEARCH_SPACES = {
    "logistic-regression": {
        "default": {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "penalty": ["l2", "l1", "elasticnet"],
            "solver": ["lbfgs", "liblinear", "saga"],
            "l1_ratio": [0.25, 0.5, 0.75],
            "max_iter": [100, 300],
        }
    },
    "xgboost": {
        "default": {
            "max_depth": [3, 6, 9, 12, None],
            "learning_rate": [0.01, 0.1, 0.3],
            "n_estimators": [50, 100, 200, 500, 1000, 2000],
        }
    },
    "ebm": {
        "default": {
            "learning_rate": [0.005, 0.015, 0.1],
            "max_bins": [256, 512, 1024],
            "outer_bags": [8, 14],
            "inner_bags": [0, 4],
            "min_sample_leaf": [4, 10],
        }
    },
    "tabpfn": {
        "default": {
            "n_estimators": [4, 8, 16],
            "softmax_temperature": [0.5, 0.75, 0.9, 1.2],
            "balance_probabilities": [True, False],
            "eval_metric": [ClassifierEvalMetrics.ROC_AUC, ClassifierEvalMetrics.F1],
        },
    },
    "tabicl-2": {
        "default": {
            "n_estimators": [4, 8, 16],
            "norm_methods": ["power", "quantile", "quantile_rtdl", "robust"],
            "softmax_temperature": [0.5, 0.75, 0.9, 1.2],
            "average_logits": [True, False],
        }
    },
    "limix": {
        "default": {
            "softmax_temperature": [0.5, 0.75, 0.9, 1.2],
        }
    },
    "orion": {
        "default": {
            "n_estimators": [4, 8, 16],
            "norm_methods": ["power", "quantile", "quantile_rtdl", "robust"],
            "softmax_temperature": [0.5, 0.75, 0.9, 1.2],
        }
    },
    "mitra": {
        "default": {
            "n_estimators": [4, 8, 16],
            "shuffle_classes": [True, False],
            "shuffle_features": [True, False],
            "use_random_transforms": [True, False],
        }
    },
}

REGRESSION_SEARCH_SPACES = {
    "xgboost": {
        "default": {
            "n_estimators": [100, 300],
            "max_depth": [3, 5],
            "learning_rate": [0.03, 0.1],
            "subsample": [0.8, 1.0],
        }
    },
    "ebm": {
        "default": {
            "max_bins": [128, 256],
            "learning_rate": [0.01, 0.05],
            "interactions": [0, 5],
        }
    },
}

# Note that search spaces with parameters that have only 1 value are still worth it
# since we use the 1 value as the "default" for the hyperparameter
# While adjusting the other hyperparameters with multiple values
MODEL_REGISTRY_CLS = {
    "logistic-regression": ModelSpec(
        f"{SKLEARN_ADAPTER}:LinearModelAdapter",
        search_spaces=CLASSIFICATION_SEARCH_SPACES["logistic-regression"],
    ),
    "xgboost": ModelSpec(
        f"{SKLEARN_ADAPTER}:XGBoostAdapter",
        search_spaces=CLASSIFICATION_SEARCH_SPACES["xgboost"],
    ),
    "ebm": ModelSpec(
        f"{SKLEARN_ADAPTER}:EBMAdapter",
        search_spaces=CLASSIFICATION_SEARCH_SPACES["ebm"],
    ),
    "tabpfn-3": ModelSpec(
        TABPFN_ADAPTER,
        search_spaces=CLASSIFICATION_SEARCH_SPACES["tabpfn"],
    ),
    "tabpfn-2.5": ModelSpec(
        TABPFN_ADAPTER,
        default_params={"version": "v2.5"},
        search_spaces=CLASSIFICATION_SEARCH_SPACES["tabpfn"],
    ),
    "tabicl-2": ModelSpec(
        TABICL_ADAPTER,
        search_spaces=CLASSIFICATION_SEARCH_SPACES["tabicl-2"],
    ),
    "limix-2m": ModelSpec(
        LIMIX_ADAPTER,
        default_params={"size": "2M"},
        search_spaces=CLASSIFICATION_SEARCH_SPACES["limix"],
    ),
    "limix-16m": ModelSpec(
        LIMIX_ADAPTER,
        default_params={"size": "16M"},
        search_spaces=CLASSIFICATION_SEARCH_SPACES["limix"],
    ),
    "mitra": ModelSpec(
        MITRA_ADAPTER, search_spaces=CLASSIFICATION_SEARCH_SPACES["mitra"]
    ),
    "orion-msp": ModelSpec(
        ORION_MSP_ADAPTER, search_spaces=CLASSIFICATION_SEARCH_SPACES["orion"]
    ),
    "orion-bix": ModelSpec(
        ORION_BIX_ADAPTER, search_spaces=CLASSIFICATION_SEARCH_SPACES["orion"]
    ),
}


MODEL_REGISTRY_REG = {
    "linear-regression": ModelSpec(f"{SKLEARN_ADAPTER}:LinearModelAdapter"),
    "xgboost": ModelSpec(
        f"{SKLEARN_ADAPTER}:XGBoostAdapter",
        search_spaces=REGRESSION_SEARCH_SPACES["xgboost"],
    ),
    "ebm": ModelSpec(
        f"{SKLEARN_ADAPTER}:EBMAdapter",
        search_spaces=REGRESSION_SEARCH_SPACES["ebm"],
    ),
    "tabpfn-3": ModelSpec(TABPFN_ADAPTER),
    "tabicl-2": ModelSpec(TABICL_ADAPTER),
    "limix-2m": ModelSpec(LIMIX_ADAPTER, default_params={"size": "2M"}),
    "limix-16m": ModelSpec(LIMIX_ADAPTER, default_params={"size": "16M"}),
    "mitra": ModelSpec(MITRA_ADAPTER),
}

MODEL_CATALOG = ModelCatalog(
    registries={
        "classification": MODEL_REGISTRY_CLS,
        "regression": MODEL_REGISTRY_REG,
    }
)
