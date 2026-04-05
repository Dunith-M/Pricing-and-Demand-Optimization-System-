def log_stage_start(logger, stage_name: str):
    logger.info(f"===== {stage_name} START =====")


def log_stage_end(logger, stage_name: str):
    logger.info(f"===== {stage_name} END =====")


def log_dataset_info(logger, df):
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")


def log_missing_values(logger, df):
    missing = df.isnull().sum()
    logger.info(f"Missing values:\n{missing}")


def log_model_metrics(logger, rmse, mae, r2):
    logger.info(f"RMSE: {rmse}")
    logger.info(f"MAE: {mae}")
    logger.info(f"R2: {r2}")


def log_optimization_results(logger, results: dict):
    logger.info(f"Optimization Results: {results}")


def log_error(logger, error: Exception):
    logger.error(f"Error: {str(error)}", exc_info=True)