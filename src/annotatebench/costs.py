from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .core import AnnotationStrategy


@dataclass(frozen=True)
class CostScenario:
    name: str
    human_cost_per_label: float
    selection_cost_per_example: Mapping[AnnotationStrategy, float]
    source_name: str = "illustrative"
    source_url: str = ""
    checked_at: str = ""
    notes: str = ""


@dataclass(frozen=True)
class HumanCostCalibration:
    """Inputs for deriving per-label cost from public platform pricing."""

    source_name: str
    source_url: str
    checked_at: str
    hourly_reward_usd: float
    seconds_per_label: float
    platform_fee_rate: float
    redundancy: int = 1
    notes: str = ""

    @property
    def cost_per_label(self) -> float:
        labor_cost = self.hourly_reward_usd * (self.seconds_per_label / 3600)
        return labor_cost * (1 + self.platform_fee_rate) * self.redundancy


@dataclass(frozen=True)
class LlmTokenPricing:
    source_name: str
    source_url: str
    checked_at: str
    model: str
    input_usd_per_million_tokens: float
    output_usd_per_million_tokens: float
    notes: str = ""


def make_human_cost_scenario(
    name: str,
    calibration: HumanCostCalibration,
    selection_cost_per_example: Mapping[AnnotationStrategy, float] | None = None,
) -> CostScenario:
    return CostScenario(
        name=name,
        human_cost_per_label=calibration.cost_per_label,
        selection_cost_per_example=selection_cost_per_example or DEFAULT_SELECTION_COSTS,
        source_name=calibration.source_name,
        source_url=calibration.source_url,
        checked_at=calibration.checked_at,
        notes=calibration.notes,
    )


def estimate_llm_annotation_cost(
    input_tokens: int,
    output_tokens: int,
    pricing: LlmTokenPricing,
) -> float:
    """Estimate API spend from public per-token model pricing."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts must be non-negative.")
    input_cost = input_tokens * pricing.input_usd_per_million_tokens / 1_000_000
    output_cost = output_tokens * pricing.output_usd_per_million_tokens / 1_000_000
    return input_cost + output_cost


DEFAULT_SELECTION_COSTS: dict[AnnotationStrategy, float] = {
    AnnotationStrategy.RANDOM: 0.0,
    AnnotationStrategy.UNCERTAINTY_AL: 0.002,
    AnnotationStrategy.DIVERSITY_AL: 0.003,
    AnnotationStrategy.HYBRID_AL: 0.004,
    AnnotationStrategy.LLM_ANNOTATOR: 0.0,
}


HUMAN_COST_CALIBRATIONS: dict[str, HumanCostCalibration] = {
    "mturk_low": HumanCostCalibration(
        source_name="Amazon Mechanical Turk",
        source_url="https://www.mturk.com/pricing",
        checked_at="2026-07-12",
        hourly_reward_usd=8.00,
        seconds_per_label=15,
        platform_fee_rate=0.20,
        notes="Low-cost single-rater estimate using MTurk requester fee.",
    ),
    "prolific_base": HumanCostCalibration(
        source_name="Prolific",
        source_url="https://www.prolific.com/pricing",
        checked_at="2026-07-12",
        hourly_reward_usd=12.00,
        seconds_per_label=20,
        platform_fee_rate=0.428,
        notes="Recommended-pay single-rater estimate for short text classification.",
    ),
    "prolific_high": HumanCostCalibration(
        source_name="Prolific",
        source_url="https://www.prolific.com/pricing",
        checked_at="2026-07-12",
        hourly_reward_usd=18.00,
        seconds_per_label=30,
        platform_fee_rate=0.428,
        redundancy=2,
        notes="Higher-cost estimate with slower items and two independent labels.",
    ),
}


COST_SCENARIOS: dict[str, CostScenario] = {
    "low": make_human_cost_scenario(
        name="low",
        calibration=HUMAN_COST_CALIBRATIONS["mturk_low"],
    ),
    "base": make_human_cost_scenario(
        name="base",
        calibration=HUMAN_COST_CALIBRATIONS["prolific_base"],
    ),
    "high": make_human_cost_scenario(
        name="high",
        calibration=HUMAN_COST_CALIBRATIONS["prolific_high"],
    ),
}


def get_cost_scenarios(names: list[str] | tuple[str, ...] | None = None) -> list[CostScenario]:
    if names is None:
        return list(COST_SCENARIOS.values())
    scenarios: list[CostScenario] = []
    for name in names:
        if name not in COST_SCENARIOS:
            raise ValueError(f"Unknown cost scenario: {name}")
        scenarios.append(COST_SCENARIOS[name])
    return scenarios


def estimate_annotation_cost(
    strategy: AnnotationStrategy,
    n_labels: int,
    scenario: CostScenario,
) -> tuple[float, float, float]:
    """Return annotation, selection, and total cost for a strategy-budget point."""
    annotation_cost = n_labels * scenario.human_cost_per_label
    selection_cost = n_labels * scenario.selection_cost_per_example.get(strategy, 0.0)
    return annotation_cost, selection_cost, annotation_cost + selection_cost
