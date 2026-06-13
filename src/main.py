from sklearn.model_selection import train_test_split

from src.adapter.tabpfn_adapter import TabPFN
from src.config import config
from src.evaluation.evaluate import evaluate_predictions
from src.interfaces.model_interface import TFModelInterface
from src.utils import load_data
from src.utils.logger import logger


def train_model(model: TFModelInterface, X_train, X_test, y_train):
    time_train = model.fit(X_train=X_train, y_train=y_train)

    predictions, time_pred = model.predict(X_test=X_test)

    return predictions, time_train + time_pred


def main():

    X, y = load_data.load_toy_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=config.train_size
    )
    logger.info("Dataset prepared")

    model_tab = TabPFN()

    predictions, time_total = train_model(model_tab, X_train, X_test, y_train)

    metrics = evaluate_predictions(predictions, y_test)
    logger.info(f"Metrics: {metrics}")
    logger.info(f"Total time (train+pred): {time_total:.3f}s")


if __name__ == "__main__":
    main()
