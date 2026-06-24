from dataclasses import dataclass, field
from typing import Any

from src.adapter.limix_adapter import LimixAdapter
from src.adapter.mitra_adapter import MitraAdapter
from src.adapter.orion_bix_adapter import OrionBixAdapter
from src.adapter.orion_msp_adapter import OrionMSPAdapter
from src.adapter.sklearn_adapter import EBMAdapter, LinearModelAdapter, XGBoostAdapter
from src.adapter.tabicl_adapter import TabICLAdapter
from src.adapter.tabpfn_adapter import TabPFNAdapter
from src.interfaces.model_interface import ModelAdapter, TaskType


@dataclass(frozen=True)
class ModelSpec:
    adapter_cls: type[ModelAdapter]
    default_params: dict[str, Any] = field(default_factory=dict)
    search_spaces: dict[str, dict[str, list[Any]]] = field(default_factory=dict)

    def create(self, task_type: TaskType, params: dict[str, Any]) -> ModelAdapter:
        return self.adapter_cls(
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


tabpfn_search_spaces = {
    "default": {
        "average_before_softmax": [True, False],
    }
}

# Note that search spaaces with parametes that have only 1 value are still worth it
# since we use the 1 value as the "default" for the hyperparameter
# While adjusting the other hyperparameters with multiple values
MODEL_REGISTRY_CLS = {
    "logistic-regression": ModelSpec(
        LinearModelAdapter,
        search_spaces={
            "default": {
                "C": [0.1, 1.0, 10.0],
                "penalty": ["l2"],
                "solver": ["lbfgs"],
            }
        },
    ),
    "xgboost": ModelSpec(
        XGBoostAdapter,
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
        EBMAdapter,
        search_spaces={
            "default": {
                "max_bins": [128, 256],
                "learning_rate": [0.01, 0.05],
                "interactions": [0, 5],
            }
        },
    ),
    "tabpfn-3": ModelSpec(
        TabPFNAdapter,
        search_spaces=tabpfn_search_spaces,
    ),
    "tabpfn-2.5": ModelSpec(
        TabPFNAdapter,
        default_params={"version": "v2.5"},
        search_spaces=tabpfn_search_spaces,
    ),
    "tabicl-2": ModelSpec(
        TabICLAdapter,
        search_spaces={
            "default": {
                "n_estimators": [4, 8, 16],
                "class_shuffle_method": ["shift"],
                "softmax_temperature": [0.75, 0.9, 1.0, 1.1],
                "average_logits": [True, False],
            }
        },
    ),
    "limix-2m": ModelSpec(LimixAdapter, default_params={"size": "2M"}),
    "limix-16m": ModelSpec(LimixAdapter, default_params={"size": "16M"}),
    "mitra": ModelSpec(MitraAdapter),
    "orion-msp": ModelSpec(OrionMSPAdapter),
    "orion-bix": ModelSpec(OrionBixAdapter),
}


MODEL_REGISTRY_REG = {
    "linear-regression": ModelSpec(LinearModelAdapter),
    "xgboost": ModelSpec(
        XGBoostAdapter,
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
        EBMAdapter,
        search_spaces={
            "default": {
                "max_bins": [128, 256],
                "learning_rate": [0.01, 0.05],
                "interactions": [0, 5],
            }
        },
    ),
    "tabpfn-3": ModelSpec(TabPFNAdapter),
    "tabicl-2": ModelSpec(TabICLAdapter),
    "limix-2m": ModelSpec(LimixAdapter, default_params={"size": "2M"}),
    "limix-16m": ModelSpec(LimixAdapter, default_params={"size": "16M"}),
    "mitra": ModelSpec(MitraAdapter),
}
