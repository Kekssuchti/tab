from src.schemas.training_schemas import ModelParams
from src.utils.model_registry import MODEL_REGISTRY_CLS, MODEL_REGISTRY_REG, ModelSpec


def get_model_spec(model_params: ModelParams) -> ModelSpec:
    registry = (
        MODEL_REGISTRY_CLS
        if model_params.task_type == "classification"
        else MODEL_REGISTRY_REG
    )

    try:
        return registry[model_params.name]
    except KeyError as exc:
        available = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown {model_params.task_type} model '{model_params.name}'. "
            f"Available models: {available}"
        ) from exc
