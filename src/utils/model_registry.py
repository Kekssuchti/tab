from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from src.schemas.base_schemas import TaskType
from src.schemas.training_schemas import ModelConfig
from src.utils.model_registry_utils import (
    SearchDomain,
    _copy_search_space,
    _expand_candidate,
    _expand_grid,
    _load_adapter_cls,
)
from src.utils.tuning_distributions import (
    DiscreteUniform,
    IntUniform,
    LogUniform,
    OptunaDistribution,
    Uniform,
    UniformChoice,
)

if TYPE_CHECKING:
    from src.interfaces.model_interface import ModelAdapter


@dataclass(frozen=True)
class ModelSpec:
    """
    Registry entry for one model adapter.

    ---
    Attributes:
        adapter_path: str
            Import path of the adapter class.

        default_params: dict
            Parameters always applied before user parameters. Defaults to empty.

        search_spaces: mapping
            Named categorical and distribution-based spaces available for tuning. Defaults to empty.
    """

    adapter_path: str
    default_params: dict[str, Any] = field(default_factory=dict)
    search_spaces: Mapping[str, Mapping[str, SearchDomain]] = field(default_factory=dict)

    def create(self, task_type: TaskType, params: dict[str, Any]) -> "ModelAdapter":
        adapter_cls = _load_adapter_cls(self.adapter_path)
        return adapter_cls(task_type=task_type, **{**self.default_params, **params})

    def tuning_grid(self, search_space: str | None, overrides: dict[str, list[Any]] | None) -> dict[str, list[Any]]:
        space = self.tuning_search_space(search_space, overrides)
        distributions = [key for key, domain in space.items() if isinstance(domain, OptunaDistribution)]
        if distributions:
            joined = ", ".join(distributions)
            raise ValueError("Grid tuning cannot expand distribution domains: " + joined)
        return {key: list(domain) for key, domain in space.items()}

    def tuning_search_space(
        self, search_space: str | None, overrides: dict[str, list[Any]] | None
    ) -> dict[str, SearchDomain]:
        if overrides is not None:
            return _copy_search_space(overrides)

        if search_space is None:
            search_space = "default"
        try:
            return _copy_search_space(self.search_spaces[search_space])
        except KeyError as exc:
            available = ", ".join(sorted(self.search_spaces)) or "none"
            raise ValueError(f"Unknown tuning search space '{search_space}'. Available: {available}") from exc

    def tuning_candidates(
        self, search_space: str | None, overrides: dict[str, list[Any]] | None
    ) -> list[dict[str, Any]]:
        return _expand_grid(self.tuning_grid(search_space, overrides))

    def tuning_candidate_from_values(self, values: Mapping[str, Any]) -> dict[str, Any]:
        return _expand_candidate(values)

    def sample_tuning_candidate(
        self,
        trial,
        search_space: Mapping[str, SearchDomain],
    ) -> dict[str, Any]:
        sampled_values = {}
        for key, domain in search_space.items():
            if isinstance(domain, OptunaDistribution):
                sampled_values[key] = domain.suggest(trial, key)
            else:
                sampled_values[key] = trial.suggest_categorical(key, domain)
        return self.tuning_candidate_from_values(sampled_values)


@dataclass(frozen=True)
class ModelCatalog:
    """
    Registry of available models by task type.

    ---
    Attributes:
        registries: mapping
            Mapping from task type to model-name registry.
    """

    registries: Mapping[TaskType, Mapping[str, ModelSpec]]

    def spec_for(self, model_config: ModelConfig, task_type: TaskType) -> ModelSpec:
        registry = self.registries[task_type]

        try:
            return registry[model_config.name]
        except KeyError as exc:
            available = ", ".join(sorted(registry))
            raise ValueError(f"Unknown {task_type} model '{model_config.name}'. Available models: {available}") from exc

    def available_models(self, task_type: TaskType) -> tuple[str, ...]:
        return tuple(sorted(self.registries[task_type]))


def get_model_spec(model_config: ModelConfig, task_type: TaskType) -> ModelSpec:
    return MODEL_CATALOG.spec_for(model_config, task_type)


SKLEARN_ADAPTER = "src.adapter.sklearn_adapter"
TABPFN_ADAPTER = "src.adapter.tabpfn_adapter:TabPFNAdapter"
TABICL_ADAPTER = "src.adapter.tabicl_adapter:TabICLAdapter"
LIMIX_ADAPTER = "src.adapter.limix_adapter:LimixAdapter"
MITRA_ADAPTER = "src.adapter.mitra_adapter:MitraAdapter"
ORION_MSP_ADAPTER = "src.adapter.orion_msp_adapter:OrionMSPAdapter"
ORION_BIX_ADAPTER = "src.adapter.orion_bix_adapter:OrionBixAdapter"
TABFM_ADAPTER = "src.adapter.tabfm_adapter:TabfmAdapter"
TABSWIFT_ADAPTER = "src.adapter.tabswift_adapter:TabSwiftAdapter"


SEARCH_SPACES = {
    "linear-regression": {
        "default": {
            "fit_intercept": [True],
        },
        "good": {
            "fit_intercept": [True, False],
            "positive": [True, False],
        },
    },
    "logistic-regression": {
        "default": {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "solver": ["lbfgs", "newton-cg", "saga"],
            "max_iter": [300, 500, 1000],
        },
        "good": {
            "C": LogUniform(0.001, 1000.0),
            "solver": ["lbfgs", "newton-cg", "saga", "sag"],
            "max_iter": IntUniform(100, 1000, 10),
            "fit_intercept": [True, False],
            "class_weight": [None, "balanced"],
            "warm_start": [False, True],
        },
    },
    "xgboost": {
        "default": {
            "n_estimators": [50, 100, 200, 500, 1000, 2000],
            "max_depth": [3, 6, 9, 12, None],
            "learning_rate": [0.01, 0.05, 0.1, 0.3],
        },
        "good": {
            # after Shwartz-Ziv et. al
            "n_estimators": IntUniform(100, 4000, 10),
            "eta": LogUniform(np.e**-7, 1),  # np.e**-7=0.00091
            "max_depth": IntUniform(1, 10, 1),
            "subsample": Uniform(0.2, 1),
            "colsample_bytree": Uniform(0.2, 1),
            "colsample_bylevel": Uniform(0.2, 1),
            "min_child_weight": LogUniform(np.e**-16, np.e**5),  # 1.125e-07 - 148.413
            "reg_alpha": UniformChoice(0, LogUniform(np.e**-16, np.e**2)),  # l1 regularization
            "reg_lambda": UniformChoice(0, LogUniform(np.e**-16, np.e**2)),  # l2 regularization
        },
    },
    "ebm": {
        "default": {
            # taken from bohlen et al.
            "learning_rate": [0.005, 0.015, 0.03],
            "max_bins": [256, 512, 1024],
            "outer_bags": [8, 14],
            "inner_bags": [0, 4],
            "min_samples_leaf": [4, 10],
        },
        "good": {
            "max_leaves": IntUniform(2, 3, 1),
            "smoothing_rounds": IntUniform(0, 1000, 25),
            "learning_rate": LogUniform(0.0025, 0.2),
            "interaction_smoothing_rounds": IntUniform(0, 200, 25),
            "min_hessian": LogUniform(1 * 10**-10, 1 * 10**-2),
            "validation_size": Uniform(0.05, 0.25),
            "min_samples_leaf": IntUniform(2, 20, 1),
            "early_stopping_tolerance": LogUniform(1 * 10**-10, 1 * 10**-5),
            "gain_scale": LogUniform(0.5, 5),
        },
    },
    "tabpfn": {
        "default": {
            "n_estimators": [4, 8, 16],
            "softmax_temperature": [0.75, 0.8, 0.9, 0.95, 1],
            "inference_config.POLYNOMIAL_FEATURES": ["no", 5, 10, 15],
            "balance_probabilities": [True, False],
        },
        "good": {
            # based on tabpfn2 paper
            # later paper dont really discuss this.
            "n_estimators": [1, 4, 8, 16, 32],
            "softmax_temperature": Uniform(0.7, 1.1),
            "balance_probabilities": [True, False],
            "inference_config.SUBSAMPLE_SAMPLES": UniformChoice(None, DiscreteUniform(0.1, 0.8, 0.1)),
            "inference_config.POLYNOMIAL_FEATURES": UniformChoice("no", DiscreteUniform(1, 20, 1)),
            "inference_config.ENABLE_GPU_PREPROCESSING": [True],
        },
        "best": {
            "n_estimators": [32],
        },
        "best_2_6": {
            "n_estimators": [4],
        },
        "best_2_5": {
            "n_estimators": [2],
        },
    },
    "tabicl": {
        "default": {
            "n_estimators": [4, 8, 16],
            "softmax_temperature": [0.75, 0.8, 0.9, 0.95, 1],
            "norm_methods": ["power", "quantile", "quantile_rtdl", "robust"],
            "average_logits": [True, False],
        },
        "good": {
            "n_estimators": [1, 4, 8, 16, 32],
            "softmax_temperature": Uniform(0.7, 1.1),
            "average_logits": [True, False],
            "norm_methods": ["power", "quantile", "quantile_rtdl", "robust"],
        },
        "best": {
            "n_estimators": [32],
        },
    },
    "limix": {
        "default": {
            "softmax_temperature": Uniform(0.7, 1.1),
        },
        "best": {
            "softmax_temperature": [0.9],
        },
    },
    "orion": {
        "default": {
            "n_estimators": [1, 2, 4, 8, 16, 32],
            "softmax_temperature": Uniform(0.7, 1.1),
            "norm_methods": ["power", "quantile", "quantile_rtdl", "robust"],
            "average_logits": [True, False],
        },
        "bix": {
            "n_estimators": [1, 2],
            "softmax_temperature": Uniform(0.7, 1.1),
        },
        "best": {
            "n_estimators": [32],
        },
    },
    "mitra": {
        "default": {
            "n_estimators": [1, 2, 4],
            "shuffle_classes": [True, False],
            "shuffle_features": [True, False],
            "use_random_transforms": [True, False],
        },
        "best": {
            "n_estimators": [8],
        },
    },
    "tabfm": {
        "default": {
            "n_estimators": [1, 2],
            "softmax_temperature": Uniform(0.7, 1.1),
        },
        "best": {
            "n_estimators": [4],
        },
    },
    "tabswift": {
        "default": {
            # could be extended!
            "n_estimators": [4, 8, 16, 32],
        },
        "best": {
            "n_estimators": [32],
        },
    },
}

# Note that search spaces with parameters that have only 1 value are still worth it
# since we use the 1 value as the "default" for the hyperparameter
# While adjusting the other hyperparameters with multiple values
_COMMON_REGISTRY = {
    "xgboost": ModelSpec(
        f"{SKLEARN_ADAPTER}:XGBoostAdapter",
        search_spaces=SEARCH_SPACES["xgboost"],
    ),
    "ebm": ModelSpec(
        f"{SKLEARN_ADAPTER}:EBMAdapter",
        search_spaces=SEARCH_SPACES["ebm"],
    ),
    "tabpfn-3": ModelSpec(
        TABPFN_ADAPTER,
        search_spaces=SEARCH_SPACES["tabpfn"],
    ),
    "tabpfn-2.5": ModelSpec(
        TABPFN_ADAPTER,
        default_params={"version": "v2.5"},
        search_spaces=SEARCH_SPACES["tabpfn"],
    ),
    "tabpfn-2.6": ModelSpec(
        TABPFN_ADAPTER,
        default_params={
            "version": "v2.6",
            "predict_batch_size": 8192,
            "fit_mode": "fit_preprocessors",
        },
        search_spaces=SEARCH_SPACES["tabpfn"],
    ),
    "tabicl-2": ModelSpec(
        TABICL_ADAPTER,
        search_spaces=SEARCH_SPACES["tabicl"],
    ),
    "limix-2m": ModelSpec(
        LIMIX_ADAPTER,
        default_params={"size": "2M"},
        search_spaces=SEARCH_SPACES["limix"],
    ),
    "limix-16m": ModelSpec(
        LIMIX_ADAPTER,
        default_params={"size": "16M"},
        search_spaces=SEARCH_SPACES["limix"],
    ),
    "mitra": ModelSpec(MITRA_ADAPTER, search_spaces=SEARCH_SPACES["mitra"]),
    "tabswift": ModelSpec(TABSWIFT_ADAPTER, search_spaces=SEARCH_SPACES["tabswift"]),
}

MODEL_REGISTRY_CLS = {
    **_COMMON_REGISTRY,
    "logistic-regression": ModelSpec(
        f"{SKLEARN_ADAPTER}:LinearModelAdapter",
        search_spaces=SEARCH_SPACES["logistic-regression"],
    ),
    "orion-msp": ModelSpec(ORION_MSP_ADAPTER, search_spaces=SEARCH_SPACES["orion"]),
    "orion-bix": ModelSpec(ORION_BIX_ADAPTER, search_spaces=SEARCH_SPACES["orion"]),
    "tabfm": ModelSpec(TABFM_ADAPTER, search_spaces=SEARCH_SPACES["tabfm"]),
}

MODEL_REGISTRY_REG = {
    **_COMMON_REGISTRY,
    "linear-regression": ModelSpec(
        f"{SKLEARN_ADAPTER}:LinearModelAdapter", search_spaces=SEARCH_SPACES["linear-regression"]
    ),
}

MODEL_CATALOG = ModelCatalog(
    registries={
        "classification": MODEL_REGISTRY_CLS,
        "regression": MODEL_REGISTRY_REG,
    }
)
