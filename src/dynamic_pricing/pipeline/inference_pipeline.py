from pathlib import Path

from src.dynamic_pricing.pipeline.inference_input_pipeline import (
    InferenceInputPipeline,
)
from src.dynamic_pricing.pipeline.prediction_pipeline import PredictionPipeline
from src.dynamic_pricing.pipeline.optimization_inference_pipeline import (
    OptimizationInferencePipeline,
)
from src.dynamic_pricing.pipeline.inference_reporting_pipeline import (
    InferenceReportingPipeline,
)

from src.dynamic_pricing.utils.logger import get_logger

logger = get_logger(__name__)


class EndToEndInferencePipeline:
    def __init__(self, schema_path="configs/schema.yaml"):
        self.schema_path = schema_path

        self.input_pipeline = InferenceInputPipeline(schema_path=schema_path)
        self.prediction_pipeline = PredictionPipeline()
        self.optimization_pipeline = OptimizationInferencePipeline()
        self.reporting_pipeline = InferenceReportingPipeline()

    def run(self, input_path: str):
        try:
            logger.info("Starting End-to-End Inference Pipeline")

            # STEP 1 — Input
            df_clean = self.input_pipeline.run(input_path)

            # STEP 2 — Prediction
            prediction_df = self.prediction_pipeline.run(df_clean)

            # STEP 3 — Optimization
            optimized_df = self.optimization_pipeline.run(prediction_df)

            # STEP 4 — Reporting
            report_df = self.reporting_pipeline.run(optimized_df)

            logger.info("End-to-End Pipeline completed successfully")

            return report_df

        except Exception as e:
            logger.error("End-to-End Pipeline failed", exc_info=True)
            raise e