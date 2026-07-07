from src.schemas.plotting_schemas import PlottingConfig


class Plotter:
    def __init__(self, params: PlottingConfig) -> None:
        self.params = params
