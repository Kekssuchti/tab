import numpy as np
import pandas as pd
from src.schemas.dataset import DatasetBundle


def _databundle_to_xy_train(data: DatasetBundle) -> tuple[pd.DataFrame, np.ndarray]:
    X_train = data.train_data.X
    y_train = data.train_data.y.to_numpy()
    return X_train, y_train
