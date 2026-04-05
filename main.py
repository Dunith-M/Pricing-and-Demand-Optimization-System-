from src.dynamic_pricing.config.configuration import ConfigurationManager
from src.dynamic_pricing.utils.logger import setup_logger


def main():
    # 🔹 Initialize logger
    logger = setup_logger()
    logger.info("===== PROJECT STARTED =====")

    try:
        # 🔹 Load configuration
        config = ConfigurationManager()

        logger.info("Loading configuration files...")

        data_config = config.get_data_ingestion_config()
        transform_config = config.get_data_transformation_config()
        model_config = config.get_model_trainer_config()
        paths_config = config.get_paths_config()

        # 🔹 Log key values (sanity check)
        logger.info(f"Raw Data Path: {data_config.raw_data_path}")
        logger.info(f"Target Column: {transform_config.target_column}")
        logger.info(f"Numerical Columns: {transform_config.numerical_columns}")
        logger.info(f"Categorical Columns: {transform_config.categorical_columns}")
        logger.info(f"Model: {model_config.model_name}")
        logger.info(f"Model Params: {model_config.model_params}")
        logger.info(f"Model Save Path: {paths_config.model_path}")

        logger.info("===== CONFIG LOADED SUCCESSFULLY =====")

    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        raise e

    logger.info("===== PROJECT FINISHED =====")


if __name__ == "__main__":
    main()