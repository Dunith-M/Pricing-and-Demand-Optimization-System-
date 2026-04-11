from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
from ortools.linear_solver import pywraplp


# ============================================================
# OPTIONAL LOGGER IMPORT
# ============================================================
try:
    from dynamic_pricing.logger import logger
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
class OptimizerConfig:
    """
    Configuration for OR optimization layer.
    """
    input_curve_path: str = "artifacts/optimization/demand_price_curve.csv"
    output_dir: str = "artifacts/optimization"
    output_prices_path: str = "artifacts/optimization/optimized_prices.csv"
    output_summary_path: str = "artifacts/optimization/optimization_summary.json"

    solver_name: str = "SCIP"   # SCIP or CBC
    listing_id_column: str = "listing_id"

    current_price_column: str = "current_price"
    candidate_price_column: str = "candidate_price"
    predicted_demand_column: str = "predicted_demand"
    expected_revenue_column: str = "expected_revenue"

    # Optional business constraints
    enable_avg_price_change_constraint: bool = False
    max_average_price_change_pct: float = 10.0

    enable_neighborhood_consistency_constraint: bool = False
    neighborhood_column: str = "neighbourhood_group"
    max_neighborhood_price_deviation_pct: float = 25.0

    enable_room_type_fairness_constraint: bool = False
    room_type_column: str = "room_type"

    # numerical tolerance
    epsilon: float = 1e-6


# ============================================================
# OPTIMIZER
# ============================================================

class PriceOptimizer:
    """
    OR-Tools based discrete price optimization engine.

    Input:
        demand_price_curve.csv
        Each row = one candidate price option for one listing

    Output:
        optimized_prices.csv
        One selected price per listing
    """

    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.df = None
        self.solver = None
        self.x_vars: Dict[Tuple[str, int], pywraplp.Variable] = {}

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------
    def load_input_data(self) -> pd.DataFrame:
        logger.info("Loading demand-price curve data from: %s", self.config.input_curve_path)
        df = pd.read_csv(self.config.input_curve_path)

        required_columns = [
            self.config.listing_id_column,
            self.config.current_price_column,
            self.config.candidate_price_column,
            self.config.predicted_demand_column,
            self.config.expected_revenue_column,
        ]

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in input data: {missing_cols}")

        logger.info("Input curve data shape: %s", df.shape)

        # Add internal option index per listing
        df = df.copy()
        df["option_id"] = df.groupby(self.config.listing_id_column).cumcount()

        return df

    # --------------------------------------------------------
    # SOLVER INIT
    # --------------------------------------------------------
    def initialize_solver(self) -> None:
        logger.info("Initializing OR-Tools solver: %s", self.config.solver_name)
        self.solver = pywraplp.Solver.CreateSolver(self.config.solver_name)

        if not self.solver:
            raise RuntimeError(
                f"Failed to initialize OR-Tools solver with name: {self.config.solver_name}"
            )

        logger.info("Solver initialized successfully.")

    # --------------------------------------------------------
    # CREATE DECISION VARIABLES
    # --------------------------------------------------------
    def create_decision_variables(self) -> None:
        """
        Create binary variable x[i,j]:
        1 if listing i selects candidate option j
        """
        logger.info("Creating binary decision variables.")

        for _, row in self.df.iterrows():
            listing_id = row[self.config.listing_id_column]
            option_id = int(row["option_id"])

            var_name = f"x_{listing_id}_{option_id}"
            self.x_vars[(listing_id, option_id)] = self.solver.BoolVar(var_name)

        logger.info("Created %s decision variables.", len(self.x_vars))

    # --------------------------------------------------------
    # OBJECTIVE FUNCTION
    # --------------------------------------------------------
    def build_objective(self) -> None:
        """
        Maximize total expected revenue:
        sum(expected_revenue[i,j] * x[i,j])
        """
        logger.info("Building objective function: maximize total expected revenue")

        objective = self.solver.Objective()

        for _, row in self.df.iterrows():
            listing_id = row[self.config.listing_id_column]
            option_id = int(row["option_id"])
            expected_revenue = float(row[self.config.expected_revenue_column])

            objective.SetCoefficient(
                self.x_vars[(listing_id, option_id)],
                expected_revenue
            )

        objective.SetMaximization()
        logger.info("Objective function built successfully.")

    # --------------------------------------------------------
    # CORE CONSTRAINT: ONE PRICE PER LISTING
    # --------------------------------------------------------
    def add_one_price_per_listing_constraint(self) -> None:
        """
        For each listing:
        sum_j x[i,j] = 1
        """
        logger.info("Adding core constraint: exactly one price per listing")

        grouped = self.df.groupby(self.config.listing_id_column)

        for listing_id, group in grouped:
            constraint = self.solver.Constraint(1, 1, f"one_price_{listing_id}")

            for _, row in group.iterrows():
                option_id = int(row["option_id"])
                constraint.SetCoefficient(self.x_vars[(listing_id, option_id)], 1)

        logger.info("One-price-per-listing constraints added.")

    # --------------------------------------------------------
    # OPTIONAL CONSTRAINT: AVG PRICE CHANGE LIMIT
    # --------------------------------------------------------
    def add_average_price_change_constraint(self) -> None:
        """
        Limit average absolute % price change across all selected listings.

        Approximation:
        Since abs() is nonlinear, we precompute abs(price_change_pct) per option
        and constrain weighted average using selected binary variables.

        sum(abs_pct_ij * x_ij) / N <= threshold
        => sum(abs_pct_ij * x_ij) <= N * threshold
        """
        if not self.config.enable_avg_price_change_constraint:
            logger.info("Average price change constraint disabled.")
            return

        logger.info("Adding optional constraint: average price change limit")

        df = self.df.copy()
        df["abs_price_change_pct"] = (
            (df[self.config.candidate_price_column] - df[self.config.current_price_column]).abs()
            / df[self.config.current_price_column].replace(0, 1.0)
        ) * 100.0

        num_listings = df[self.config.listing_id_column].nunique()
        rhs = num_listings * self.config.max_average_price_change_pct

        constraint = self.solver.Constraint(
            -self.solver.infinity(),
            rhs,
            "avg_price_change_limit"
        )

        for _, row in df.iterrows():
            listing_id = row[self.config.listing_id_column]
            option_id = int(row["option_id"])
            abs_pct = float(row["abs_price_change_pct"])

            constraint.SetCoefficient(self.x_vars[(listing_id, option_id)], abs_pct)

        logger.info(
            "Average price change constraint added. Max avg change: %.2f%%",
            self.config.max_average_price_change_pct
        )

    # --------------------------------------------------------
    # OPTIONAL CONSTRAINT: NEIGHBORHOOD CONSISTENCY
    # --------------------------------------------------------
    def add_neighborhood_consistency_constraints(self) -> None:
        """
        Very simple first version:
        If neighborhood column exists, restrict selected prices to stay within
        neighborhood candidate min/max band expanded by deviation percentage.

        This is a soft business realism approximation.
        """
        if not self.config.enable_neighborhood_consistency_constraint:
            logger.info("Neighborhood consistency constraint disabled.")
            return

        if self.config.neighborhood_column not in self.df.columns:
            logger.warning(
                "Neighborhood column '%s' not found. Skipping neighborhood consistency constraint.",
                self.config.neighborhood_column
            )
            return

        logger.info("Adding optional neighborhood consistency constraints.")

        grouped = self.df.groupby(self.config.neighborhood_column)

        for neighborhood, group in grouped:
            neighborhood_min = float(group[self.config.current_price_column].min())
            neighborhood_max = float(group[self.config.current_price_column].max())

            lower_bound = neighborhood_min * (
                1 - self.config.max_neighborhood_price_deviation_pct / 100.0
            )
            upper_bound = neighborhood_max * (
                1 + self.config.max_neighborhood_price_deviation_pct / 100.0
            )

            for _, row in group.iterrows():
                listing_id = row[self.config.listing_id_column]
                option_id = int(row["option_id"])
                candidate_price = float(row[self.config.candidate_price_column])

                # If candidate violates allowed band, force variable = 0
                if candidate_price < lower_bound or candidate_price > upper_bound:
                    constraint = self.solver.Constraint(
                        0, 0, f"neighborhood_band_{listing_id}_{option_id}"
                    )
                    constraint.SetCoefficient(self.x_vars[(listing_id, option_id)], 1)

        logger.info("Neighborhood consistency constraints added.")

    # --------------------------------------------------------
    # OPTIONAL CONSTRAINT: ROOM-TYPE FAIRNESS
    # --------------------------------------------------------
    def add_room_type_fairness_constraints(self) -> None:
        """
        First simple business rule:
        Avoid extreme candidate prices within a room type.

        Here we disable options outside room-type observed current price band
        expanded by a safety factor.
        """
        if not self.config.enable_room_type_fairness_constraint:
            logger.info("Room-type fairness constraint disabled.")
            return

        if self.config.room_type_column not in self.df.columns:
            logger.warning(
                "Room type column '%s' not found. Skipping room-type fairness constraint.",
                self.config.room_type_column
            )
            return

        logger.info("Adding optional room-type fairness constraints.")

        grouped = self.df.groupby(self.config.room_type_column)

        for room_type, group in grouped:
            rt_min = float(group[self.config.current_price_column].min()) * 0.75
            rt_max = float(group[self.config.current_price_column].max()) * 1.25

            for _, row in group.iterrows():
                listing_id = row[self.config.listing_id_column]
                option_id = int(row["option_id"])
                candidate_price = float(row[self.config.candidate_price_column])

                if candidate_price < rt_min or candidate_price > rt_max:
                    constraint = self.solver.Constraint(
                        0, 0, f"roomtype_band_{listing_id}_{option_id}"
                    )
                    constraint.SetCoefficient(self.x_vars[(listing_id, option_id)], 1)

        logger.info("Room-type fairness constraints added.")

    # --------------------------------------------------------
    # SOLVE
    # --------------------------------------------------------
    def solve(self) -> int:
        logger.info("Starting optimization solve.")
        status = self.solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            logger.info("Optimization status: OPTIMAL")
        elif status == pywraplp.Solver.FEASIBLE:
            logger.warning("Optimization status: FEASIBLE (not proven optimal)")
        else:
            logger.error("Optimization failed. Status code: %s", status)

        return status

    # --------------------------------------------------------
    # EXTRACT SOLUTION
    # --------------------------------------------------------
    def extract_solution(self) -> pd.DataFrame:
        """
        Extract selected option for each listing.
        """
        logger.info("Extracting optimization solution.")

        selected_rows = []

        for _, row in self.df.iterrows():
            listing_id = row[self.config.listing_id_column]
            option_id = int(row["option_id"])

            value = self.x_vars[(listing_id, option_id)].solution_value()

            if value > 0.5:
                selected_rows.append(row.to_dict())

        result_df = pd.DataFrame(selected_rows)

        if result_df.empty:
            raise RuntimeError("No selected solution rows were extracted.")

        # Rename candidate price to recommended price
        result_df = result_df.rename(
            columns={self.config.candidate_price_column: "recommended_price"}
        )

        # Add improvement columns
        result_df["current_revenue_estimate"] = (
            result_df[self.config.current_price_column] *
            result_df[self.config.predicted_demand_column]
        )

        result_df["expected_revenue_change"] = (
            result_df[self.config.expected_revenue_column] -
            result_df["current_revenue_estimate"]
        )

        result_df["price_change"] = (
            result_df["recommended_price"] -
            result_df[self.config.current_price_column]
        )

        result_df["price_change_pct"] = (
            result_df["price_change"] /
            result_df[self.config.current_price_column].replace(0, 1.0)
        ) * 100.0

        # Optional simple reason
        result_df["reason"] = result_df.apply(self._generate_reason, axis=1)

        logger.info("Solution extracted successfully. Selected rows: %s", result_df.shape[0])

        return result_df

    # --------------------------------------------------------
    # SIMPLE EXPLANATION
    # --------------------------------------------------------
    def _generate_reason(self, row: pd.Series) -> str:
        """
        Basic explanation for business output.
        """
        price_change = row["price_change"]
        revenue_change = row["expected_revenue_change"]

        if price_change > 0 and revenue_change > 0:
            return "Higher price improves expected revenue."
        if price_change < 0 and revenue_change > 0:
            return "Lower price improves demand enough to increase expected revenue."
        if abs(price_change) <= self.config.epsilon:
            return "Current price already appears near-optimal."
        return "Selected as best feasible option under optimization rules."

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------
    def save_outputs(self, result_df: pd.DataFrame, status: int) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)

        logger.info("Saving optimized prices to: %s", self.config.output_prices_path)
        result_df.to_csv(self.config.output_prices_path, index=False)

        summary = {
            "solver_name": self.config.solver_name,
            "status_code": int(status),
            "num_input_rows": int(self.df.shape[0]),
            "num_listings": int(self.df[self.config.listing_id_column].nunique()),
            "num_selected_rows": int(result_df.shape[0]),
            "total_expected_revenue_after": float(result_df[self.config.expected_revenue_column].sum()),
            "total_estimated_revenue_before": float(result_df["current_revenue_estimate"].sum()),
            "total_expected_revenue_change": float(result_df["expected_revenue_change"].sum()),
            "average_price_change_pct": float(result_df["price_change_pct"].mean()),
            "optional_constraints": {
                "avg_price_change_enabled": self.config.enable_avg_price_change_constraint,
                "neighborhood_consistency_enabled": self.config.enable_neighborhood_consistency_constraint,
                "room_type_fairness_enabled": self.config.enable_room_type_fairness_constraint,
            }
        }

        logger.info("Saving optimization summary to: %s", self.config.output_summary_path)
        with open(self.config.output_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        logger.info("Optimization outputs saved successfully.")

    # --------------------------------------------------------
    # FULL RUN
    # --------------------------------------------------------
    def run(self) -> pd.DataFrame:
        logger.info("=" * 80)
        logger.info("PRICE OPTIMIZATION PIPELINE STARTED")
        logger.info("=" * 80)

        self.df = self.load_input_data()
        self.initialize_solver()
        self.create_decision_variables()
        self.build_objective()
        self.add_one_price_per_listing_constraint()
        self.add_average_price_change_constraint()
        self.add_neighborhood_consistency_constraints()
        self.add_room_type_fairness_constraints()

        status = self.solve()

        if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            raise RuntimeError(f"Optimization failed. Solver status code: {status}")

        result_df = self.extract_solution()
        self.save_outputs(result_df, status)

        logger.info("=" * 80)
        logger.info("PRICE OPTIMIZATION PIPELINE FINISHED")
        logger.info("=" * 80)

        return result_df


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    config = OptimizerConfig()
    optimizer = PriceOptimizer(config)
    optimizer.run()