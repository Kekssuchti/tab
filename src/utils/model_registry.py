from dataclasses import dataclass, field
from functools import lru_cache
from importlib import import_module
from typing import TYPE_CHECKING, Any

from src.schemas.base_schemas import TaskType
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.interfaces.model_interface import ModelAdapter

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
    search_spaces: dict[str, dict[str, list[Any]]] = field(default_factory=dict)

    def create(self, task_type: TaskType, params: dict[str, Any]) -> "ModelAdapter":
        adapter_cls = _load_adapter_cls(self.adapter_path)
        return adapter_cls(
            task_type=task_type, **{**self.default_params, **params}
        )

    def tuning_grid(
        self, search_space: str | None, overrides: dict[str, list[Any]] | None
    ) -> dict[str, list[Any]]:
        # if we get any search space override / new grid we use it
        if overrides:
            return overrides

        # if we dont get override we use the provided search space.
        # This will most likely be default.
        # But can later also be specified with "best baseline" or whatever
        if search_space is None:
            search_space = "default"
        try:
            return self.search_spaces[search_space]
        except KeyError as exc:
            available = ", ".join(sorted(self.search_spaces)) or "none"
            raise ValueError(
                f"Unknown tuning search space '{search_space}'. Available: {available}"
            ) from exc


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


tabpfn_search_spaces = {
    "default": {
        "average_before_softmax": [True, False],
    }
}

# Note that search spaces with parameters that have only 1 value are still worth it
# since we use the 1 value as the "default" for the hyperparameter
# While adjusting the other hyperparameters with multiple values
MODEL_REGISTRY_CLS = {
    "logistic-regression": ModelSpec(
        f"{SKLEARN_ADAPTER}:LinearModelAdapter",
        search_spaces={
            "default": {
                "C": [0.1, 1.0, 10.0],
                "penalty": ["l2"],
                "solver": ["lbfgs"],
            }
        },
    ),
    "xgboost": ModelSpec(
        f"{SKLEARN_ADAPTER}:XGBoostAdapter",
        search_spaces={
            "default": {
                "n_estimators": [100, 300],
                "max_depth": [3, 5],
                "learning_rate": [0.03, 0.1],
                "gamma": [0.0, 0.1],
                "subsample": [0.8, 1.0],
            }
        },
    ),
    "ebm": ModelSpec(
        f"{SKLEARN_ADAPTER}:EBMAdapter",
        search_spaces={
            "default": {
                "max_bins": [128, 256, 1024],
                "learning_rate": [0.015, 0.03, 0.05],
                "interactions": [0, 1, 3, 5],
            }
        },
    ),
    "tabpfn-3": ModelSpec(
        TABPFN_ADAPTER,
        search_spaces=tabpfn_search_spaces,
    ),
    "tabpfn-2.5": ModelSpec(
        TABPFN_ADAPTER,
        default_params={"version": "v2.5"},
        search_spaces=tabpfn_search_spaces,
    ),
    "tabicl-2": ModelSpec(
        TABICL_ADAPTER,
        search_spaces={
            "default": {
                "n_estimators": [4, 8, 16],
                "class_shuffle_method": ["shift"],
                "softmax_temperature": [0.75, 0.9, 1.0, 1.1],
                "average_logits": [True, False],
            }
        },
    ),
    "limix-2m": ModelSpec(LIMIX_ADAPTER, default_params={"size": "2M"}),
    "limix-16m": ModelSpec(LIMIX_ADAPTER, default_params={"size": "16M"}),
    "mitra": ModelSpec(MITRA_ADAPTER),
    "orion-msp": ModelSpec(ORION_MSP_ADAPTER),
    "orion-bix": ModelSpec(ORION_BIX_ADAPTER),
}


MODEL_REGISTRY_REG = {
    "linear-regression": ModelSpec(f"{SKLEARN_ADAPTER}:LinearModelAdapter"),
    "xgboost": ModelSpec(
        f"{SKLEARN_ADAPTER}:XGBoostAdapter",
        search_spaces={
            "default": {
                "n_estimators": [100, 300],
                "max_depth": [3, 5],
                "learning_rate": [0.03, 0.1],
                "subsample": [0.8, 1.0],
            }
        },
    ),
    "ebm": ModelSpec(
        f"{SKLEARN_ADAPTER}:EBMAdapter",
        search_spaces={
            "default": {
                "max_bins": [128, 256],
                "learning_rate": [0.01, 0.05],
                "interactions": [0, 5],
            }
        },
    ),
    "tabpfn-3": ModelSpec(TABPFN_ADAPTER),
    "tabicl-2": ModelSpec(TABICL_ADAPTER),
    "limix-2m": ModelSpec(LIMIX_ADAPTER, default_params={"size": "2M"}),
    "limix-16m": ModelSpec(LIMIX_ADAPTER, default_params={"size": "16M"}),
    "mitra": ModelSpec(MITRA_ADAPTER),
}
