from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .core import AnnotationStrategy


@dataclass(frozen=True)
class CostScenario:
    name: str
    human_cost_per_label: float
    selection_cost_per_example: Mapping[AnnotationStrategy, float]


DEFAULT_SELECTION_COSTS: dict[AnnotationStrategy, float] = {
    AnnotationStrategy.RANDOM: 0.0,
    AnnotationStrategy.UNCERTAINTY_AL: 0.002,
    AnnotationStrategy.DIVERSITY_AL: 0.003,
    AnnotationStrategy.HYBRID_AL: 0.004,
    AnnotationStrategy.LLM_ANNOTATOR: 0.0,
}


COST_SCENARIOS: dict[str, CostScenario] = {
    "low": CostScenario(
        name="low",
        human_cost_per_label=0.03,
        selection_cost_per_example=DEFAULT_SELECTION_COSTS,
    ),
    "base": CostScenario(
        name="base",
        human_cost_per_label=0.10,
        selection_cost_per_example=DEFAULT_SELECTION_COSTS,
    ),
    "high": CostScenario(
        name="high",
        human_cost_per_label=0.30,
        selection_cost_per_example=DEFAULT_SELECTION_COSTS,
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
