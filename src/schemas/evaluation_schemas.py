from src.schemas.base_schemas import StrictParams


class EvaluationParams(StrictParams):
    metrics: tuple[str, ...] = ("all",)
