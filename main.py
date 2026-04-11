from pathlib import Path

from src.dynamic_pricing.pipeline.inference_input_pipeline import (
    InferenceInputPipeline,
)
from src.dynamic_pricing.pipeline.prediction_pipeline import PredictionPipeline
from src.dynamic_pricing.pipeline.preprocessing_pipeline import (
    FeatureEngineering,
    prepare_categorical_data,
    replace_inf_with_nan,
)
from src.dynamic_pricing.utils.logger import setup_logger
from src.dynamic_pricing.utils.logging_helpers import (
    log_stage_end,
    log_stage_start,
)


logger = setup_logger()


def resolve_input_path() -> str:
    """
    Resolve input CSV path dynamically.
    """
    candidate_paths = [
        "data/processed/feature_engineered_data.csv",
        "data/processed/airbnb.csv",
        "data/raw/airbnb.csv",
    ]

    for candidate in candidate_paths:
        if Path(candidate).exists():
            return candidate

    raise FileNotFoundError(
        "No input CSV found. Checked: " + ", ".join(candidate_paths)
    )


def main():
    try:
        log_stage_start(logger, "INFERENCE INPUT PIPELINE")
        logger.info("System initialized")

        input_pipeline = InferenceInputPipeline(schema_path="configs/schema.yaml")

        input_path = resolve_input_path()
        logger.info(f"Using input path: {input_path}")

        df_clean = input_pipeline.run(input_path)
        logger.info(f"Processed data shape: {df_clean.shape}")

        log_stage_end(logger, "INFERENCE INPUT PIPELINE")

        log_stage_start(logger, "PREDICTION PIPELINE")

        prediction_pipeline = PredictionPipeline()
        prediction_df = prediction_pipeline.run(df_clean)

        logger.info(f"Prediction output shape: {prediction_df.shape}")

        print("\nPrediction Results:")
        print(prediction_df.head())

        log_stage_end(logger, "PREDICTION PIPELINE")

    except Exception as error:
        logger.error("Pipeline failed", exc_info=True)
        raise error


if __name__ == "__main__":
    main()
