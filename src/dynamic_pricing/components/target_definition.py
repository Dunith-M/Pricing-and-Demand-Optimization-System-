# src/airbnb_price/components/target_definition.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# If you already created logger utility, import it
from src.dynamic_pricing.utils.logger import get_logger

logger = get_logger(__name__)


class TargetDefinition:
    def __init__(self, config):
        self.test_size = config["data_transformation"]["test_size"]
        self.random_state = config["data_transformation"]["random_state"]
        self.target_dir = "artifacts/data/target"

        os.makedirs(self.target_dir, exist_ok=True)

    @staticmethod
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min() + 1e-8)

    def create_demand_score(self, df: pd.DataFrame) -> pd.Series:
        logger.info("Creating demand_score...")

        reviews_per_month = df["reviews per month"].fillna(0)
        number_of_reviews = df["number of reviews"].fillna(0)
        availability = df["availability 365"].fillna(365)

        norm_reviews_pm = self.normalize(reviews_per_month)
        norm_total_reviews = self.normalize(number_of_reviews)
        norm_availability = 1 - (availability / 365)

        demand_score = (
            0.5 * norm_reviews_pm +
            0.3 * norm_total_reviews +
            0.2 * norm_availability
        )

        return demand_score

    def run(self, df: pd.DataFrame):
        logger.info("Starting target definition pipeline...")

        # Create target
        df["demand_score"] = self.create_demand_score(df)

        # Separate features and target
        X = df.drop(columns=["demand_score"])
        y = df["demand_score"]

        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Target mean: {y.mean():.4f}")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        # Save target arrays
        np.save(os.path.join(self.target_dir, "y_train.npy"), y_train.values)
        np.save(os.path.join(self.target_dir, "y_test.npy"), y_test.values)

        logger.info("Saved target arrays successfully")

        return X_train, X_test, y_train, y_test