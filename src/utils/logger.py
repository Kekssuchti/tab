import logging
from functools import lru_cache
from logging.handlers import RotatingFileHandler

from pythonjsonlogger.json import JsonFormatter

from src.config import config

ACTIVE_LOG_FILE = "active.log"
APPEND_LOG_FILE = "app.log"
MAX_APPEND_LOG_BYTES = 10 * 1024 * 1024
APPEND_LOG_BACKUP_COUNT = 10


@lru_cache
def configure_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    config.dir_log.mkdir(parents=True, exist_ok=True)

    formatter = JsonFormatter(["asctime", "levelname", "filename", "lineno", "message"])

    active_handler = logging.FileHandler(
        config.dir_log / ACTIVE_LOG_FILE,
        mode="w",
    )
    active_handler.setFormatter(formatter)
    logger.addHandler(active_handler)

    append_handler = RotatingFileHandler(
        config.dir_log / APPEND_LOG_FILE,
        mode="a",
        maxBytes=MAX_APPEND_LOG_BYTES,
        backupCount=APPEND_LOG_BACKUP_COUNT,
    )
    append_handler.setFormatter(formatter)
    logger.addHandler(append_handler)

    logger.info("logger is configured")
    return logger


logger = configure_logger()
