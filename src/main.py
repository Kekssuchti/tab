from sklearn.model_selection import train_test_split

from src.adapter.limix_adapter import LimixAdapter
from src.adapter.mitra_adapter import MitraAdapter
from src.adapter.tabicl_adapter import TabICLAdapter
from src.adapter.tabpfn_adapter import TabPFNAdapter
from src.config import config
from src.evaluation.evaluate import evaluate_predictions
from src.interfaces.model_interface import TFModelInterface
from src.utils import load_data
from src.utils.logger import logger
from src.utils.model_registry import MODEL_REGISTRY


def train_model(model: TFModelInterface, X_train, X_test, y_train):
    time_train = model.fit(X_train=X_train, y_train=y_train)

    predictions, time_pred = model.predict(X_test=X_test)

    return predictions, time_train + time_pred


def main():
    X, y = load_data.load_toy_data_cls()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=config.train_size
    )
    logger.info("Dataset prepared")

    model_tabpfn = TabPFNAdapter()
    model_icl = TabICLAdapter()
    model_limix_2m = LimixAdapter(size="16M")
    model_mitra = MitraAdapter()

    predictions, time_total = train_model(model_mitra, X_train, X_test, y_train)

    metrics = evaluate_predictions(predictions, y_test)
    logger.info(f"Metrics: {metrics}")
    logger.info(f"Total time (train+pred): {time_total:.3f}s")


if __name__ == "__main__":
    main()
