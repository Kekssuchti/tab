from collections import Counter, defaultdict

from src.schemas.training_schemas import ModelParams


def model_instance_ids(models: tuple[ModelParams, ...]) -> list[str]:
    counts = Counter(model.name for model in models)
    seen: defaultdict[str, int] = defaultdict(int)
    model_ids = []
    for model in models:
        index = seen[model.name]
        seen[model.name] += 1
        model_ids.append(
            model.name if counts[model.name] == 1 else f"{model.name}__{index}"
        )
    return model_ids
