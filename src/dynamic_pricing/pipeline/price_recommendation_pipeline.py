from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pandas as pd

from src.dynamic_pricing.components.demand_simulator import (
    DemandSimulator,
    DemandSimulationConfig,
)
from src.dynamic_pricing.components.optimizer import (
    PriceOptimizer,
    OptimizerConfig,
)
from src.dynamic_pricing.pipeline.preprocessing_pipeline import (
    FeatureEngineering,
    prepare_categorical_data,
    replace_inf_with_nan,
)


# ============================================================
# OPTIONAL LOGGER IMPORT
# ============================================================
try:
    from src.dynamic_pricing.utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)


# ============================================================
# CONFIG
# ============================================================

@dataclass
class PriceRecommendationPipelineConfig:
    """
    Configuration for the full recommendation pipeline.
    """
    reports_dir: str = "artifacts/reports"
    recommendations_output_path: str = "artifacts/reports/price_recommendations.csv"
    summary_output_path: str = "artifacts/reports/optimization_summary.json"


# ============================================================
# PIPELINE
# ============================================================

class PriceRecommendationPipeline:
    """
    End-to-end pipeline:
    1. Simulate demand for multiple candidate prices
    2. Optimize price selection using OR-Tools
    3. Convert selected prices into business-ready recommendations
    4. Save outputs
    """

    def __init__(
        self,
        simulation_config: DemandSimulationConfig | None = None,
        optimizer_config: OptimizerConfig | None = None,
        pipeline_config: PriceRecommendationPipelineConfig | None = None,
    ):
        self.simulation_config = simulation_config or DemandSimulationConfig()
        self.optimizer_config = optimizer_config or OptimizerConfig()
        self.pipeline_config = pipeline_config or PriceRecommendationPipelineConfig()

    # --------------------------------------------------------
    # RULE-BASED EXPLANATION LOGIC
    # --------------------------------------------------------
    @staticmethod
    def generate_reason(row: pd.Series) -> str:
        """
        Generate simple business-friendly explanation.
        """
        current_price = float(row["current_price"])
        recommended_price = float(row["recommended_price"])
        demand_before = float(row["predicted_demand_before"])
        demand_after = float(row["predicted_demand_after"])
        revenue_change = float(row["expected_revenue_change"])

        if recommended_price < current_price and demand_after > demand_before:
            return "Lower price is expected to increase demand and improve occupancy."

        if recommended_price > current_price and revenue_change > 0:
            return "Higher price is expected to improve revenue with acceptable demand loss."

        if abs(recommended_price - current_price) < 1e-6:
            return "Current price appears close to the best feasible price."

        if revenue_change <= 0:
            return "Recommendation is constrained by business rules or limited feasible options."

        return "Selected as the best feasible price under the optimization model."

    # --------------------------------------------------------
    # BUSINESS OUTPUT FORMATTER
    # --------------------------------------------------------
    def build_recommendation_output(self, optimized_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert raw optimization output into business-ready recommendation table.
        """
        logger.info("Building business-ready recommendation output.")

        df = optimized_df.copy()

        # Current demand estimate:
        # In prior optimizer version, current_revenue_estimate was computed using
        # current_price * predicted_demand (same selected-row predicted_demand).
        # That is not ideal as a true before-state, but acceptable for version 1.
        # Here we explicitly define before/after columns for business readability.
        df["predicted_demand_before"] = df["predicted_demand"]
        df["predicted_demand_after"] = df["predicted_demand"]

        # Revenue before/after
        df["expected_revenue_before"] = df["current_price"] * df["predicted_demand_before"]
        df["expected_revenue_after"] = df["recommended_price"] * df["predicted_demand_after"]
        df["expected_revenue_change"] = (
            df["expected_revenue_after"] - df["expected_revenue_before"]
        )

        # Human-readable action label
        df["price_action"] = df.apply(self._derive_price_action, axis=1)

        # Explanation
        df["reason"] = df.apply(self.generate_reason, axis=1)

        # Select business-facing columns
        final_columns = [
            "listing_id",
            "current_price",
            "recommended_price",
            "price_action",
            "predicted_demand_before",
            "predicted_demand_after",
            "expected_revenue_before",
            "expected_revenue_after",
            "expected_revenue_change",
            "price_change",
            "price_change_pct",
            "reason",
        ]

        # include optional business columns if they exist
        optional_columns = [
            "room_type",
            "neighbourhood_group",
        ]
        final_columns_with_optional = []

        for col in ["listing_id", "current_price", "recommended_price"]:
            final_columns_with_optional.append(col)

        for col in optional_columns:
            if col in df.columns:
                final_columns_with_optional.append(col)

        final_columns_with_optional.extend([
            "price_action",
            "predicted_demand_before",
            "predicted_demand_after",
            "expected_revenue_before",
            "expected_revenue_after",
            "expected_revenue_change",
            "price_change",
            "price_change_pct",
            "reason",
        ])

        result_df = df[final_columns_with_optional].copy()

        logger.info("Recommendation output built successfully. Shape: %s", result_df.shape)
        return result_df

    # --------------------------------------------------------
    # PRICE ACTION LABEL
    # --------------------------------------------------------
    @staticmethod
    def _derive_price_action(row: pd.Series) -> str:
        """
        Translate price change into simple action label.
        """
        price_change = float(row["price_change"])

        if price_change > 0:
            return "Increase Price"
        if price_change < 0:
            return "Decrease Price"
        return "Keep Price"

    # --------------------------------------------------------
    # SUMMARY REPORT
    # --------------------------------------------------------
    def build_summary(self, recommendations_df: pd.DataFrame) -> dict:
        """
        Build summary metrics for reporting.
        """
        logger.info("Building optimization summary.")

        summary = {
            "num_listings": int(recommendations_df["listing_id"].nunique()),
            "num_recommendations": int(len(recommendations_df)),
            "increase_price_count": int((recommendations_df["price_action"] == "Increase Price").sum()),
            "decrease_price_count": int((recommendations_df["price_action"] == "Decrease Price").sum()),
            "keep_price_count": int((recommendations_df["price_action"] == "Keep Price").sum()),
            "average_price_change_pct": float(recommendations_df["price_change_pct"].mean()),
            "total_expected_revenue_before": float(recommendations_df["expected_revenue_before"].sum()),
            "total_expected_revenue_after": float(recommendations_df["expected_revenue_after"].sum()),
            "total_expected_revenue_change": float(recommendations_df["expected_revenue_change"].sum()),
        }

        logger.info("Summary built successfully.")
        return summary

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------
    def save_outputs(self, recommendations_df: pd.DataFrame, summary: dict) -> None:
        """
        Save final recommendation CSV and summary JSON.
        """
        os.makedirs(self.pipeline_config.reports_dir, exist_ok=True)

        logger.info(
            "Saving price recommendations to: %s",
            self.pipeline_config.recommendations_output_path
        )
        recommendations_df.to_csv(
            self.pipeline_config.recommendations_output_path,
            index=False
        )

        logger.info(
            "Saving optimization summary to: %s",
            self.pipeline_config.summary_output_path
        )
        with open(self.pipeline_config.summary_output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        logger.info("Final outputs saved successfully.")

    # --------------------------------------------------------
    # FULL RUN
    # --------------------------------------------------------
    def run(self) -> pd.DataFrame:
        logger.info("=" * 80)
        logger.info("PRICE RECOMMENDATION PIPELINE STARTED")
        logger.info("=" * 80)

        # Step 1: Demand simulation
        simulator = DemandSimulator(self.simulation_config)
        simulation_df = simulator.run()
        logger.info("Demand simulation completed. Shape: %s", simulation_df.shape)

        # Step 2: Optimization
        optimizer = PriceOptimizer(self.optimizer_config)
        optimized_df = optimizer.run()
        logger.info("Optimization completed. Shape: %s", optimized_df.shape)

        # Step 3: Business-ready recommendations
        recommendations_df = self.build_recommendation_output(optimized_df)

        # Step 4: Summary
        summary = self.build_summary(recommendations_df)

        # Step 5: Save
        self.save_outputs(recommendations_df, summary)

        logger.info("=" * 80)
        logger.info("PRICE RECOMMENDATION PIPELINE FINISHED")
        logger.info("=" * 80)

        return recommendations_df


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    pipeline = PriceRecommendationPipeline()
    pipeline.run()
