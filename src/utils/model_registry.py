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

        default_params: dict, default={}
            Parameters always applied before user parameters.

        search_spaces: mapping, default={}
            Named categorical and distribution-based spaces available for tuning.
    """

    adapter_path: str
    default_params: dict[str, Any] = field(default_factory=dict)
    search_spaces: Mapping[str, Mapping[str, SearchDomain]] = field(
        default_factory=dict
    )

    def create(self, task_type: TaskType, params: dict[str, Any]) -> "ModelAdapter":
        adapter_cls = _load_adapter_cls(self.adapter_path)
        return adapter_cls(task_type=task_type, **{**self.default_params, **params})

    def tuning_grid(
        self, search_space: str | None, overrides: dict[str, list[Any]] | None
    ) -> dict[str, list[Any]]:
        space = self.tuning_search_space(search_space, overrides)
        distributions = [
            key
            for key, domain in space.items()
            if isinstance(domain, OptunaDistribution)
        ]
        if distributions:
            joined = ", ".join(distributions)
            raise ValueError(
                "Grid tuning cannot expand distribution domains: " + joined
            )
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
            raise ValueError(
                f"Unknown tuning search space '{search_space}'. Available: {available}"
            ) from exc

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

    def spec_for(self, model_params: ModelConfig) -> ModelSpec:
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


def get_model_spec(model_params: ModelConfig) -> ModelSpec:
    return MODEL_CATALOG.spec_for(model_params)


SKLEARN_ADAPTER = "src.adapter.sklearn_adapter"
TABPFN_ADAPTER = "src.adapter.tabpfn_adapter:TabPFNAdapter"
TABICL_ADAPTER = "src.adapter.tabicl_adapter:TabICLAdapter"
LIMIX_ADAPTER = "src.adapter.limix_adapter:LimixAdapter"
MITRA_ADAPTER = "src.adapter.mitra_adapter:MitraAdapter"
ORION_MSP_ADAPTER = "src.adapter.orion_msp_adapter:OrionMSPAdapter"
ORION_BIX_ADAPTER = "src.adapter.orion_bix_adapter:OrionBixAdapter"
TABFM_ADAPTER = "src.adapter.tabfm_adapter:TabfmAdapter"
TABSWIFT_ADAPTER = "src.adapter.tabswift_adapter:TabSwiftAdapter"


CLASSIFICATION_SEARCH_SPACES = {
    "logistic-regression": {
        "default": {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "solver": ["lbfgs", "newton-cg", "saga"],
            "max_iter": [300, 500, 1000],
            # "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
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
            "reg_alpha": UniformChoice(
                0, LogUniform(np.e**-16, np.e**2)
            ),  # l1 regularization
            "reg_lambda": UniformChoice(
                0, LogUniform(np.e**-16, np.e**2)
            ),  # l2 regularization
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
        }
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
            "inference_config.SUBSAMPLE_SAMPLES": UniformChoice(
                None, DiscreteUniform(0.1, 0.8, 0.1)
            ),
            "inference_config.POLYNOMIAL_FEATURES": UniformChoice(
                "no", DiscreteUniform(1, 20, 1)
            ),
            "inference_config.ENABLE_GPU_PREPROCESSING": [True],
        },
        "best": {
            "n_estimators": [32],
        },
        "tabpfn2.5": {
            "n_estimators": [1, 2],  # oom else
            "softmax_temperature": Uniform(0.7, 1.1),
            "balance_probabilities": [True, False],
            "inference_config.SUBSAMPLE_SAMPLES": UniformChoice(
                None, DiscreteUniform(0.1, 0.8, 0.1)
            ),
            "inference_config.POLYNOMIAL_FEATURES": UniformChoice(
                "no", DiscreteUniform(1, 20, 1)
            ),
            "inference_config.ENABLE_GPU_PREPROCESSING": [True],
        },
    },
    "tabicl-2": {
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
            "n_estimators": [1, 2],  # 4, 8
            "softmax_temperature": Uniform(0.7, 1.1),
        },
        "best": {
            "n_estimators": [4],
        },
    },
    "tabswift": {
        "default": {
            "n_estimators": [1, 2],
        },
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
    "tabfm": ModelSpec(
        TABFM_ADAPTER, search_spaces=CLASSIFICATION_SEARCH_SPACES["tabfm"]
    ),
    "tabswift": ModelSpec(
        TABSWIFT_ADAPTER, search_spaces=CLASSIFICATION_SEARCH_SPACES["tabswift"]
    ),
}


MODEL_REGISTRY_REG = {
    # "linear-regression": ModelSpec(f"{SKLEARN_ADAPTER}:LinearModelAdapter"),
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
    # "limix-2m": ModelSpec(LIMIX_ADAPTER, default_params={"size": "2M"}),
    # "limix-16m": ModelSpec(LIMIX_ADAPTER, default_params={"size": "16M"}),
    # "mitra": ModelSpec(MITRA_ADAPTER),
    "tabswift": ModelSpec(TABSWIFT_ADAPTER, search_spaces=CLASSIFICATION_SEARCH_SPACES["tabswift"]),
}

MODEL_CATALOG = ModelCatalog(
    registries={
        "classification": MODEL_REGISTRY_CLS,
        "regression": MODEL_REGISTRY_REG,
    }
)
