from src.dynamic_pricing.utils.logger import setup_logger
from src.dynamic_pricing.utils.logging_helpers import (
    log_stage_start,
    log_stage_end,
)

logger = setup_logger()


def main():
    try:
        log_stage_start(logger, "MAIN")

        logger.info("System initialized")

        log_stage_end(logger, "MAIN")

    except Exception as e:
        logger.error("Pipeline failed", exc_info=True)
        raise e


if __name__ == "__main__":
    main()