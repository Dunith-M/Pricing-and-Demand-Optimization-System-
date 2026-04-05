import logging
import os


def setup_logger(name: str = "dynamic_pricing") -> logging.Logger:
    """
    Centralized logger (file + console)
    """

    os.makedirs("logs", exist_ok=True)

    log_file = "logs/project.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # format
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger