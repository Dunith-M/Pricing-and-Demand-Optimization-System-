from pathlib import Path

from src.dynamic_pricing.pipeline.inference_pipeline import (
    EndToEndInferencePipeline,
)

from src.dynamic_pricing.utils.logger import setup_logger
from src.dynamic_pricing.utils.logging_helpers import (
    log_stage_start,
    log_stage_end,
)

logger = setup_logger()


def resolve_input_path() -> str:
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
        log_stage_start(logger, "END-TO-END INFERENCE")

        pipeline = EndToEndInferencePipeline()

        input_path = resolve_input_path()
        logger.info(f"Using input path: {input_path}")

        report_df = pipeline.run(input_path)

        print("\n✅ FINAL OUTPUT:")
        print(report_df.head())

        log_stage_end(logger, "END-TO-END INFERENCE")

    except Exception as error:
        logger.error("Pipeline failed", exc_info=True)
        raise error


if __name__ == "__main__":
    main()