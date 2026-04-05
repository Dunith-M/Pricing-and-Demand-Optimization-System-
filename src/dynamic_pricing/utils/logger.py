import logging
import os
from datetime import datetime


def setup_logger(name: str = "dynamic_pricing") -> logging.Logger:
    """
    Centralized logger setup
    """

    if not os.path.exists("logs"):
        os.makedirs("logs")

    log_file = f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Format
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger