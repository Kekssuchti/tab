from src.schemas.training_schemas import TrainingParams


class Trainer:
    def __init__(self, params: TrainingParams) -> None:
        self.params = params
