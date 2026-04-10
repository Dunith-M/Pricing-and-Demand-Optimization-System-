from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import json


# ============================================================
# VARIABLE DEFINITIONS
# ============================================================

@dataclass
class DecisionVariable:
    """
    Represents the main optimization decision variable.
    In this project, the optimizer chooses a recommended price
    for each Airbnb listing.
    """
    name: str
    description: str
    variable_type: str  
    unit: str
    lower_bound_source: str
    upper_bound_source: str


# ============================================================
# OBJECTIVE DEFINITIONS
# ============================================================

@dataclass
class ObjectiveDefinition:
    """
    Defines what the optimization layer is trying to maximize or minimize.
    """
    name: str
    sense: str  # "maximize" or "minimize"
    mathematical_form: str
    business_meaning: str
    primary_metric: str
    secondary_metrics: List[str] = field(default_factory=list)


# ============================================================
# CONSTRAINT DEFINITIONS
# ============================================================

@dataclass
class ConstraintDefinition:
    """
    Represents one business/technical rule that the optimizer must respect.
    """
    name: str
    category: str
    description: str
    mathematical_form: str
    is_hard_constraint: bool = True
    required_inputs: List[str] = field(default_factory=list)
    notes: Optional[str] = None


# ============================================================
# ASSUMPTIONS
# ============================================================

@dataclass
class AssumptionDefinition:
    """
    Assumptions are important because your current optimizer depends
    on ML predictions and simulated demand responses.
    """
    name: str
    description: str
    risk_if_invalid: str


# ============================================================
# FULL PROBLEM DEFINITION
# ============================================================

@dataclass
class OptimizationProblemDefinition:
    """
    Full formal definition of the pricing optimization problem.
    """
    project_name: str
    version: str
    problem_statement: str
    decision_variable: DecisionVariable
    objective: ObjectiveDefinition
    constraints: List[ConstraintDefinition]
    assumptions: List[AssumptionDefinition]
    input_requirements: List[str]
    output_definition: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert full definition to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 4) -> str:
        """Convert full definition to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_to_json(self, output_path: str) -> None:
        """Save the problem definition to a JSON file."""
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=4)

    def summary(self) -> str:
        """
        Human-readable summary of the optimization problem.
        Useful for logging or debugging.
        """
        lines = [
            "=" * 80,
            f"PROJECT: {self.project_name}",
            f"VERSION: {self.version}",
            "-" * 80,
            "PROBLEM STATEMENT:",
            self.problem_statement,
            "-" * 80,
            "DECISION VARIABLE:",
            f"  Name: {self.decision_variable.name}",
            f"  Type: {self.decision_variable.variable_type}",
            f"  Description: {self.decision_variable.description}",
            "-" * 80,
            "OBJECTIVE:",
            f"  Name: {self.objective.name}",
            f"  Sense: {self.objective.sense}",
            f"  Formula: {self.objective.mathematical_form}",
            f"  Business Meaning: {self.objective.business_meaning}",
            "-" * 80,
            "CONSTRAINTS:"
        ]

        for idx, constraint in enumerate(self.constraints, start=1):
            lines.extend([
                f"  {idx}. {constraint.name}",
                f"     Category: {constraint.category}",
                f"     Type: {'Hard' if constraint.is_hard_constraint else 'Soft'}",
                f"     Formula: {constraint.mathematical_form}",
                f"     Description: {constraint.description}",
            ])

        lines.append("-" * 80)
        lines.append("ASSUMPTIONS:")

        for idx, assumption in enumerate(self.assumptions, start=1):
            lines.extend([
                f"  {idx}. {assumption.name}",
                f"     Description: {assumption.description}",
                f"     Risk: {assumption.risk_if_invalid}",
            ])

        lines.append("-" * 80)
        lines.append("REQUIRED INPUTS:")
        for item in self.input_requirements:
            lines.append(f"  - {item}")

        lines.append("-" * 80)
        lines.append("EXPECTED OUTPUTS:")
        for item in self.output_definition:
            lines.append(f"  - {item}")

        lines.append("=" * 80)

        return "\n".join(lines)


# ============================================================
# FACTORY FUNCTION FOR YOUR AIRBNB PROJECT
# ============================================================

def create_airbnb_pricing_problem_definition() -> OptimizationProblemDefinition:
    """
    Creates the formal optimization problem definition for the
    Airbnb dynamic pricing and demand optimization system.

    This does NOT solve the optimization problem.
    It only defines it clearly and consistently.
    """

    decision_variable = DecisionVariable(
        name="recommended_price_i",
        description=(
            "Recommended price for listing i selected by the optimization layer. "
            "This is the main controllable business decision."
        ),
        variable_type="continuous_or_discrete_candidate_price",
        unit="USD",
        lower_bound_source="listing-level or business-defined minimum allowed price",
        upper_bound_source="listing-level or business-defined maximum allowed price"
    )

    objective = ObjectiveDefinition(
        name="maximize_expected_revenue",
        sense="maximize",
        mathematical_form="maximize Σ (recommended_price_i × predicted_demand_i)",
        business_meaning=(
            "Choose prices that maximize expected revenue across listings while "
            "respecting pricing rules and business constraints."
        ),
        primary_metric="expected_revenue",
        secondary_metrics=[
            "predicted_occupancy",
            "expected_demand",
            "price_stability",
            "pricing_fairness"
        ]
    )

    constraints = [
        ConstraintDefinition(
            name="minimum_price_constraint",
            category="pricing_bound",
            description="Recommended price cannot go below the allowed minimum price.",
            mathematical_form="recommended_price_i >= min_price_i",
            is_hard_constraint=True,
            required_inputs=["recommended_price_i", "min_price_i"]
        ),
        ConstraintDefinition(
            name="maximum_price_constraint",
            category="pricing_bound",
            description="Recommended price cannot exceed the allowed maximum price.",
            mathematical_form="recommended_price_i <= max_price_i",
            is_hard_constraint=True,
            required_inputs=["recommended_price_i", "max_price_i"]
        ),
        ConstraintDefinition(
            name="price_change_limit_constraint",
            category="price_stability",
            description=(
                "Recommended price should not increase or decrease too sharply "
                "compared with the current listing price."
            ),
            mathematical_form="|recommended_price_i - current_price_i| <= max_price_change_i",
            is_hard_constraint=False,
            required_inputs=[
                "recommended_price_i",
                "current_price_i",
                "max_price_change_i"
            ],
            notes="Can be modeled as hard or soft later depending on feasibility."
        ),
        ConstraintDefinition(
            name="neighborhood_pricing_constraint",
            category="market_alignment",
            description=(
                "Recommended price should remain within an acceptable pricing range "
                "for the listing's neighborhood."
            ),
            mathematical_form=(
                "neighborhood_min_price_g <= recommended_price_i <= neighborhood_max_price_g"
            ),
            is_hard_constraint=False,
            required_inputs=[
                "recommended_price_i",
                "neighborhood_min_price_g",
                "neighborhood_max_price_g"
            ],
            notes="Useful to avoid market-misaligned pricing recommendations."
        ),
        ConstraintDefinition(
            name="room_type_pricing_constraint",
            category="business_logic",
            description=(
                "Recommended price must follow room-type-specific pricing logic or bounds."
            ),
            mathematical_form="room_type_min_r <= recommended_price_i <= room_type_max_r",
            is_hard_constraint=False,
            required_inputs=[
                "recommended_price_i",
                "room_type_min_r",
                "room_type_max_r"
            ]
        ),
        ConstraintDefinition(
            name="seasonal_pricing_constraint",
            category="seasonality",
            description=(
                "Recommended price should respect seasonal pricing policy "
                "for high/low demand periods."
            ),
            mathematical_form="recommended_price_i must satisfy seasonal policy rules",
            is_hard_constraint=False,
            required_inputs=[
                "recommended_price_i",
                "season",
                "seasonal_multiplier_or_rule"
            ],
            notes="This may later be implemented as rule-based pre-filtering or explicit constraints."
        )
    ]

    assumptions = [
        AssumptionDefinition(
            name="ml_predictions_are_price_sensitive",
            description=(
                "Predicted demand changes meaningfully when price changes."
            ),
            risk_if_invalid=(
                "If the ML model does not properly capture price-demand behavior, "
                "optimization recommendations will be misleading."
            )
        ),
        AssumptionDefinition(
            name="simulated_price_range_is_realistic",
            description=(
                "The candidate price range used during optimization is realistic "
                "and within the training-data distribution or acceptable business limits."
            ),
            risk_if_invalid=(
                "If candidate prices are unrealistic, demand predictions and "
                "recommendations may become unreliable."
            )
        ),
        AssumptionDefinition(
            name="other_features_remain_constant_during_simulation",
            description=(
                "When simulating alternative prices for a listing, non-price features "
                "are assumed to remain unchanged for that optimization run."
            ),
            risk_if_invalid=(
                "If other important features change at the same time, the estimated "
                "demand response may not match reality."
            )
        ),
        AssumptionDefinition(
            name="single_period_optimization",
            description=(
                "The current optimization is for a single pricing decision period, "
                "not a multi-day or multi-week dynamic control system."
            ),
            risk_if_invalid=(
                "If inter-temporal effects matter, a one-period optimizer may miss "
                "better long-term pricing strategies."
            )
        )
    ]

    input_requirements = [
        "listing_id",
        "current_price",
        "minimum_allowed_price",
        "maximum_allowed_price",
        "room_type",
        "neighbourhood_group",
        "seasonal_context",
        "engineered features used by ML model",
        "predicted demand for candidate prices",
        "business policy parameters"
    ]

    output_definition = [
        "listing_id",
        "current_price",
        "recommended_price",
        "predicted_demand_before",
        "predicted_demand_after",
        "expected_revenue_before",
        "expected_revenue_after",
        "expected_revenue_change",
        "reason_for_recommendation"
    ]

    return OptimizationProblemDefinition(
        project_name="Airbnb Dynamic Pricing and Demand Optimization System",
        version="1.0",
        problem_statement=(
            "For each Airbnb listing, choose an optimal recommended price that maximizes "
            "expected business value (starting with expected revenue) using ML-based demand "
            "predictions, while satisfying pricing bounds, business rules, neighborhood logic, "
            "room-type logic, and stability constraints."
        ),
        decision_variable=decision_variable,
        objective=objective,
        constraints=constraints,
        assumptions=assumptions,
        input_requirements=input_requirements,
        output_definition=output_definition
    )


# ============================================================
# OPTIONAL LOCAL TEST
# ============================================================

if __name__ == "__main__":
    problem_definition = create_airbnb_pricing_problem_definition()
    print(problem_definition.summary())