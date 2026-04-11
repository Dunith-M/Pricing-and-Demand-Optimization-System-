from __future__ import annotations

from typing import Dict, Tuple
import pandas as pd
from ortools.linear_solver import pywraplp


# ============================================================
# LOGGER (OPTIONAL)
# ============================================================
try:
    from dynamic_pricing.logger import logger
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


# ============================================================
# CONSTRAINT FUNCTIONS
# ============================================================

class ConstraintBuilder:
    """
    Collection of reusable constraint functions.

    Each method:
    - receives solver, dataframe, and decision variables
    - applies constraints cleanly
    """

    def __init__(self, solver: pywraplp.Solver, df: pd.DataFrame,
                 x_vars: Dict[Tuple[str, int], pywraplp.Variable]):
        self.solver = solver
        self.df = df
        self.x_vars = x_vars

    # --------------------------------------------------------
    # 1. PRICE CHANGE LIMIT CONSTRAINT
    # --------------------------------------------------------
    def add_price_change_limit_constraint(self, max_pct_change: float = 20.0):
        """
        Limit % price change per listing.

        |new_price - current_price| <= threshold

        Since OR-Tools linear solver cannot handle abs(),
        we pre-filter invalid options.
        """
        logger.info("Applying price change limit constraint (±%.2f%%)", max_pct_change)

        for _, row in self.df.iterrows():
            listing_id = row["listing_id"]
            option_id = int(row["option_id"])

            current_price = float(row["current_price"])
            candidate_price = float(row["candidate_price"])

            if current_price == 0:
                continue

            pct_change = abs(candidate_price - current_price) / current_price * 100.0

            if pct_change > max_pct_change:
                # force x[i,j] = 0
                constraint = self.solver.Constraint(0, 0)
                constraint.SetCoefficient(self.x_vars[(listing_id, option_id)], 1)

        logger.info("Price change constraint applied.")

    # --------------------------------------------------------
    # 2. NEIGHBORHOOD CONSISTENCY
    # --------------------------------------------------------
    def add_neighborhood_consistency_constraint(self, deviation_pct: float = 25.0):
        """
        Keep prices within reasonable neighborhood range.

        candidate_price must be within:
        [min * (1 - deviation), max * (1 + deviation)]
        """
        if "neighbourhood_group" not in self.df.columns:
            logger.warning("neighbourhood_group not found. Skipping constraint.")
            return

        logger.info("Applying neighborhood consistency constraint (±%.2f%%)", deviation_pct)

        grouped = self.df.groupby("neighbourhood_group")

        for neighborhood, group in grouped:
            min_price = float(group["current_price"].min())
            max_price = float(group["current_price"].max())

            lower = min_price * (1 - deviation_pct / 100)
            upper = max_price * (1 + deviation_pct / 100)

            for _, row in group.iterrows():
                listing_id = row["listing_id"]
                option_id = int(row["option_id"])
                candidate_price = float(row["candidate_price"])

                if candidate_price < lower or candidate_price > upper:
                    constraint = self.solver.Constraint(0, 0)
                    constraint.SetCoefficient(self.x_vars[(listing_id, option_id)], 1)

        logger.info("Neighborhood constraint applied.")

    # --------------------------------------------------------
    # 3. ROOM TYPE HIERARCHY
    # --------------------------------------------------------
    def add_room_type_hierarchy_constraint(self):
        """
        Enforce:
        Entire home ≥ Private room ≥ Shared room

        This is a soft enforcement using filtering.
        """
        if "room_type" not in self.df.columns:
            logger.warning("room_type not found. Skipping constraint.")
            return

        logger.info("Applying room type hierarchy constraint.")

        # Define hierarchy
        hierarchy = {
            "Entire home/apt": 3,
            "Private room": 2,
            "Shared room": 1
        }

        # Compute average current price per room type
        avg_prices = self.df.groupby("room_type")["current_price"].mean().to_dict()

        for _, row in self.df.iterrows():
            listing_id = row["listing_id"]
            option_id = int(row["option_id"])

            room_type = row["room_type"]
            candidate_price = float(row["candidate_price"])

            if room_type not in hierarchy:
                continue

            for other_type, rank in hierarchy.items():
                if other_type not in avg_prices:
                    continue

                # if lower-ranked type has higher price → invalidate
                if hierarchy[room_type] < rank:
                    if candidate_price > avg_prices[other_type]:
                        constraint = self.solver.Constraint(0, 0)
                        constraint.SetCoefficient(self.x_vars[(listing_id, option_id)], 1)

        logger.info("Room-type hierarchy constraint applied.")

    # --------------------------------------------------------
    # 4. COMPETITOR CONSTRAINT (OPTIONAL)
    # --------------------------------------------------------
    def add_competitor_constraint(self, factor: float = 1.2):
        """
        candidate_price ≤ competitor_avg × factor

        Requires 'competitor_avg_price' column
        """
        if "competitor_avg_price" not in self.df.columns:
            logger.warning("competitor_avg_price not found. Skipping constraint.")
            return

        logger.info("Applying competitor constraint (factor = %.2f)", factor)

        for _, row in self.df.iterrows():
            listing_id = row["listing_id"]
            option_id = int(row["option_id"])

            candidate_price = float(row["candidate_price"])
            competitor_price = float(row["competitor_avg_price"])

            if candidate_price > competitor_price * factor:
                constraint = self.solver.Constraint(0, 0)
                constraint.SetCoefficient(self.x_vars[(listing_id, option_id)], 1)

        logger.info("Competitor constraint applied.")