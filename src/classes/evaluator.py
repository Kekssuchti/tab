from src.schemas.evaluation_schemas import EvaluationParams


class Evaluator:
    def __init__(self, params: EvaluationParams) -> None:
        self.params = params
