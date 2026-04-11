import numpy as np
import pandas as pd

from src.dynamic_pricing.utils.logger import get_logger

logger = get_logger(__name__)


class OptimizationInferencePipeline:
    def __init__(
        self,
        price_min_factor: float = 0.7,
        price_max_factor: float = 1.3,
        num_price_points: int = 10,
    ):
        """
        Optimization layer

        Args:
            price_min_factor: lower bound multiplier
            price_max_factor: upper bound multiplier
            num_price_points: number of price candidates
        """
        self.price_min_factor = price_min_factor
        self.price_max_factor = price_max_factor
        self.num_price_points = num_price_points

    def generate_price_candidates(self, current_price: float):
        """
        Generate price search space
        """
        price_min = current_price * self.price_min_factor
        price_max = current_price * self.price_max_factor

        return np.linspace(price_min, price_max, self.num_price_points)

    def simulate_demand(self, base_demand, current_price, new_price):
        """
        Simple demand-price relationship

        Assumption:
        price ↑ → demand ↓
        """
        elasticity = -1.2  # you can tune this later

        price_ratio = new_price / current_price

        adjusted_demand = base_demand * (price_ratio ** elasticity)

        return max(adjusted_demand, 0)

    def optimize_single_listing(self, row):
        """
        Optimize price for a single listing
        """
        current_price = row["current_price"]
        base_demand = row["predicted_demand_before"]
        base_revenue = row["predicted_revenue_before"]

        price_candidates = self.generate_price_candidates(current_price)

        best_price = current_price
        best_revenue = base_revenue
        best_demand = base_demand

        for price in price_candidates:
            demand = self.simulate_demand(base_demand, current_price, price)
            revenue = price * demand

            if revenue > best_revenue:
                best_price = price
                best_revenue = revenue
                best_demand = demand

        return pd.Series(
            {
                "optimized_price": round(best_price, 2),
                "predicted_demand_after": best_demand,
                "predicted_revenue_after": best_revenue,
                "expected_improvement": best_revenue - base_revenue,
            }
        )

    def run(self, prediction_df: pd.DataFrame):
        """
        Run optimization for all listings
        """
        try:
            logger.info("Starting Optimization Inference Pipeline")

            optimized_df = prediction_df.copy()

            optimization_results = optimized_df.apply(
                self.optimize_single_listing, axis=1
            )

            final_df = pd.concat([optimized_df, optimization_results], axis=1)

            logger.info("Optimization completed")

            return final_df

        except Exception as e:
            logger.error(f"Optimization pipeline failed: {e}")
            raise e