from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd

from src.dynamic_pricing.pipeline.preprocessing_pipeline import (
    FeatureEngineering,
    prepare_categorical_data,
    replace_inf_with_nan,
)


# ============================================================
# OPTIONAL LOGGER IMPORT
# ============================================================
try:
    from airbnb_price.logger import logger
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
class DemandSimulationConfig:
    """
    Configuration for demand-price simulation.

    Adjust these paths/columns if your project naming is slightly different.
    """
    model_path: str = "artifacts/models/final_model.pkl"
    preprocessor_path: str = "artifacts/preprocessing/preprocessor.pkl"
    input_data_path: str = "artifacts/data/processed/listings_for_optimization.csv"

    output_dir: str = "artifacts/optimization"
    output_npy_path: str = "artifacts/optimization/demand_price_curve.npy"
    output_csv_path: str = "artifacts/optimization/demand_price_curve.csv"

    listing_id_column: str = "id"
    price_column: str = "price"

    # If your project predicts "demand_score", keep this name for clarity
    prediction_column_name: str = "predicted_demand"

    # candidate price generation
    n_price_points: int = 15
    min_price_factor: float = 0.80
    max_price_factor: float = 1.20

    # safety guard
    clip_negative_predictions_to_zero: bool = True


# ============================================================
# DEMAND SIMULATOR
# ============================================================

class DemandSimulator:
    """
    Simulates demand over multiple candidate prices for each listing.

    Core logic:
    - Take each listing row
    - Generate multiple candidate prices
    - Replace the price feature
    - Transform input using preprocessor
    - Predict demand using trained model
    - Save results for OR optimization
    """

    def __init__(self, config: DemandSimulationConfig):
        self.config = config
        self.project_root = Path(__file__).resolve().parents[3]
        self.model = None
        self.preprocessor = None

    def _resolve_existing_path(self, *candidate_paths: str) -> Path:
        for candidate in candidate_paths:
            candidate_path = Path(candidate)
            if not candidate_path.is_absolute():
                candidate_path = self.project_root / candidate_path

            if candidate_path.exists():
                return candidate_path

        raise FileNotFoundError(
            "None of the candidate paths exist: " + ", ".join(candidate_paths)
        )

    # --------------------------------------------------------
    # LOAD ARTIFACTS
    # --------------------------------------------------------
    def load_artifacts(self) -> None:
        """Load trained model and fitted preprocessor."""
        model_path = self._resolve_existing_path(self.config.model_path)
        logger.info("Loading model from: %s", model_path)
        self.model = joblib.load(model_path)

        preprocessor_path = self._resolve_existing_path(self.config.preprocessor_path)
        logger.info("Loading preprocessor from: %s", preprocessor_path)
        self.preprocessor = joblib.load(preprocessor_path)

        logger.info("Artifacts loaded successfully.")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------
    def load_input_data(self) -> pd.DataFrame:
        """Load listing data used for optimization simulation."""
        input_data_path = self._resolve_existing_path(
            self.config.input_data_path,
            "data/processed/feature_engineered_data.csv",
            "artifacts/data/splits/test.csv",
            "artifacts/data/splits/train.csv",
        )
        logger.info("Loading input listing data from: %s", input_data_path)
        df = pd.read_csv(input_data_path)

        if self.config.price_column not in df.columns:
            raise ValueError(
                f"Price column '{self.config.price_column}' not found in input data."
            )

        if self.config.listing_id_column not in df.columns:
            raise ValueError(
                f"Listing ID column '{self.config.listing_id_column}' not found in input data."
            )

        logger.info("Input data shape: %s", df.shape)
        return df

    # --------------------------------------------------------
    # PRICE RANGE CREATION
    # --------------------------------------------------------
    def generate_price_candidates(self, current_price: float) -> np.ndarray:
        """
        Generate candidate prices around the current price.

        Example:
        current = 100
        min_factor = 0.8
        max_factor = 1.2
        -> prices from 80 to 120
        """
        min_price = current_price * self.config.min_price_factor
        max_price = current_price * self.config.max_price_factor

        # protect against bad values
        min_price = max(min_price, 1.0)
        max_price = max(max_price, min_price + 1.0)

        candidate_prices = np.linspace(
            min_price,
            max_price,
            self.config.n_price_points
        )

        # round for cleaner business output
        candidate_prices = np.round(candidate_prices, 2)

        return candidate_prices

    # --------------------------------------------------------
    # PREDICT DEMAND FOR ONE PRICE
    # --------------------------------------------------------
    def predict_demand_for_row(self, row_df: pd.DataFrame) -> float:
        """
        Predict demand for a single-row DataFrame after replacing price.
        """
        transformed_row = self.preprocessor.transform(row_df)
        prediction = self.model.predict(transformed_row)[0]

        if self.config.clip_negative_predictions_to_zero:
            prediction = max(float(prediction), 0.0)

        return float(prediction)

    # --------------------------------------------------------
    # MAIN SIMULATION
    # --------------------------------------------------------
    def simulate_demand_curves(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-demand simulation for every listing.

        Output columns:
        - listing_id
        - current_price
        - candidate_price
        - predicted_demand
        - expected_revenue
        - price_change
        - price_change_pct
        """
        logger.info("Starting demand-price simulation.")
        simulation_results: List[dict] = []

        total_rows = len(df)

        for idx, (_, row) in enumerate(df.iterrows(), start=1):
            listing_id = row[self.config.listing_id_column]
            current_price = float(row[self.config.price_column])

            candidate_prices = self.generate_price_candidates(current_price)

            for candidate_price in candidate_prices:
                simulated_row = row.copy()
                simulated_row[self.config.price_column] = candidate_price

                simulated_row_df = pd.DataFrame([simulated_row])

                predicted_demand = self.predict_demand_for_row(simulated_row_df)
                expected_revenue = candidate_price * predicted_demand

                simulation_results.append(
                    {
                        "listing_id": listing_id,
                        "current_price": round(current_price, 2),
                        "candidate_price": round(float(candidate_price), 2),
                        self.config.prediction_column_name: round(predicted_demand, 6),
                        "expected_revenue": round(expected_revenue, 6),
                        "price_change": round(float(candidate_price - current_price), 2),
                        "price_change_pct": round(
                            ((candidate_price - current_price) / current_price) * 100, 4
                        ) if current_price != 0 else 0.0,
                    }
                )

            if idx % 100 == 0 or idx == total_rows:
                logger.info(
                    "Simulated %s/%s listings completed.",
                    idx,
                    total_rows
                )

        result_df = pd.DataFrame(simulation_results)
        logger.info("Demand-price simulation completed. Output shape: %s", result_df.shape)

        return result_df

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------
    def save_outputs(self, result_df: pd.DataFrame) -> None:
        """Save outputs as CSV and NPY."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        logger.info("Saving CSV output to: %s", self.config.output_csv_path)
        result_df.to_csv(self.config.output_csv_path, index=False)

        logger.info("Saving NPY output to: %s", self.config.output_npy_path)
        np.save(self.config.output_npy_path, result_df.to_records(index=False))

        logger.info("Outputs saved successfully.")

    # --------------------------------------------------------
    # FULL RUN
    # --------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """
        Full pipeline:
        1. load model + preprocessor
        2. load input data
        3. simulate candidate-price demand curves
        4. save output artifacts
        """
        logger.info("=" * 80)
        logger.info("DEMAND SIMULATION PIPELINE STARTED")
        logger.info("=" * 80)

        self.load_artifacts()
        df = self.load_input_data()
        result_df = self.simulate_demand_curves(df)
        self.save_outputs(result_df)

        logger.info("=" * 80)
        logger.info("DEMAND SIMULATION PIPELINE FINISHED")
        logger.info("=" * 80)

        return result_df


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    config = DemandSimulationConfig()
    simulator = DemandSimulator(config)
    simulator.run()
