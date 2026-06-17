from src.schemas.plotting_schemas import PlottingParams


class Plotter:
    def __init__(self, params: PlottingParams) -> None:
        self.params = params
