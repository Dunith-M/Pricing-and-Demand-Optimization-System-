import json
from pathlib import Path

import pandas as pd

from src.dynamic_pricing.utils.logger import get_logger

logger = get_logger(__name__)


class InferenceReportingPipeline:
    def __init__(
        self,
        output_dir: str = "artifacts/reports/inference",
        keep_threshold: float = 0.02,
    ):
        """
        Reporting layer for optimized inference results

        Args:
            output_dir (str): directory to save reporting artifacts
            keep_threshold (float): relative improvement threshold to classify as KEEP
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.csv_output_path = self.output_dir / "price_recommendations.csv"
        self.json_output_path = self.output_dir / "inference_summary.json"

        self.keep_threshold = keep_threshold

    def generate_recommendation(self, row: pd.Series) -> str:
        """
        Create business-facing recommendation label
        """
        current_price = row["current_price"]
        optimized_price = row["optimized_price"]

        if current_price == 0:
            return "keep"

        price_change_ratio = (optimized_price - current_price) / current_price

        if price_change_ratio > self.keep_threshold:
            return "increase"
        elif price_change_ratio < -self.keep_threshold:
            return "decrease"
        return "keep"

    def generate_reason(self, row: pd.Series) -> str:
        """
        Create human-readable explanation
        """
        recommendation = row["recommendation"]
        improvement = row["expected_improvement"]
        demand_before = row["predicted_demand_before"]
        demand_after = row["predicted_demand_after"]
        revenue_before = row["predicted_revenue_before"]
        revenue_after = row["predicted_revenue_after"]

        if recommendation == "increase":
            if revenue_after > revenue_before:
                return (
                    f"Increase price because expected revenue improves by "
                    f"{improvement:.2f} while demand remains acceptable."
                )
            return "Increase price due to optimization output."

        elif recommendation == "decrease":
            if demand_after > demand_before:
                return (
                    f"Decrease price because lower price is expected to raise demand "
                    f"and improve revenue by {improvement:.2f}."
                )
            return "Decrease price due to optimization output."

        return (
            f"Keep current price because expected improvement is small "
            f"and current price is already near optimal."
        )

    def build_report_dataframe(self, optimized_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build final reporting dataframe
        """
        try:
            logger.info("Building reporting dataframe")

            report_df = optimized_df.copy()

            report_df["recommendation"] = report_df.apply(
                self.generate_recommendation, axis=1
            )

            report_df["reason"] = report_df.apply(self.generate_reason, axis=1)

            required_columns = [
                "listing_id",
                "current_price",
                "optimized_price",
                "predicted_demand_before",
                "predicted_revenue_before",
                "predicted_demand_after",
                "predicted_revenue_after",
                "expected_improvement",
                "recommendation",
                "reason",
            ]

            report_df = report_df[required_columns]

            logger.info("Reporting dataframe created successfully")

            return report_df

        except Exception as error:
            logger.error("Failed to build reporting dataframe", exc_info=True)
            raise error

    def export_csv(self, report_df: pd.DataFrame):
        """
        Export recommendations CSV
        """
        try:
            logger.info(f"Saving CSV report to: {self.csv_output_path}")
            report_df.to_csv(self.csv_output_path, index=False)
            logger.info("CSV export completed")
        except Exception as error:
            logger.error("CSV export failed", exc_info=True)
            raise error

    def export_summary_json(self, report_df: pd.DataFrame):
        """
        Export summary JSON
        """
        try:
            logger.info(f"Saving summary JSON to: {self.json_output_path}")

            summary = {
                "total_listings": int(len(report_df)),
                "increase_count": int(
                    (report_df["recommendation"] == "increase").sum()
                ),
                "decrease_count": int(
                    (report_df["recommendation"] == "decrease").sum()
                ),
                "keep_count": int((report_df["recommendation"] == "keep").sum()),
                "total_expected_improvement": float(
                    report_df["expected_improvement"].sum()
                ),
                "average_expected_improvement": float(
                    report_df["expected_improvement"].mean()
                ),
                "max_expected_improvement": float(
                    report_df["expected_improvement"].max()
                ),
                "min_expected_improvement": float(
                    report_df["expected_improvement"].min()
                ),
            }

            with open(self.json_output_path, "w", encoding="utf-8") as file:
                json.dump(summary, file, indent=4)

            logger.info("Summary JSON export completed")

        except Exception as error:
            logger.error("JSON export failed", exc_info=True)
            raise error

    def run(self, optimized_df: pd.DataFrame) -> pd.DataFrame:
        """
        Full reporting pipeline
        """
        try:
            logger.info("Starting Inference Reporting Pipeline")

            report_df = self.build_report_dataframe(optimized_df)

            self.export_csv(report_df)
            self.export_summary_json(report_df)

            logger.info("Inference Reporting Pipeline completed successfully")

            return report_df

        except Exception as error:
            logger.error("Inference reporting pipeline failed", exc_info=True)
            raise error