import numpy as np
import pandas as pd


def model_smoke_data():
    """Small balanced, nonclinical context and distinct held-out prediction rows."""
    rng = np.random.default_rng(2026)
    features = rng.normal(size=(60, 6)).astype(np.float32)
    labels = np.tile([0, 1], 30)
    features[:, 0] += labels
    frame = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(6)])
    return frame.iloc[:48].copy(), labels[:48].copy(), frame.iloc[48:].copy()
