import numpy as np
import pandas as pd

from src.config import config


def load_toy_classification_data():
    data = pd.read_csv(config.dir_data_toy / "dna_methylation.csv")
    target = "methylation_status"
    return data.drop(columns=target), data[target].to_numpy()


def load_toy_regression_data():
    data = pd.read_csv(config.dir_data_toy / "employee_salary_regression.csv")
    target = "annual_salary_usd"
    features = pd.get_dummies(
        data.drop(columns=[target, "employee_id"]),
        columns=["education_level", "job_role"],
    ).astype(np.float32)
    return features, data[target].to_numpy()
