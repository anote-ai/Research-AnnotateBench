"""Tests for scenario-based annotation cost estimates."""
from __future__ import annotations

import pytest

from annotatebench.core import AnnotationStrategy
from annotatebench.costs import (
    COST_SCENARIOS,
    LlmTokenPricing,
    estimate_annotation_cost,
    estimate_llm_annotation_cost,
    get_cost_scenarios,
)


def test_default_cost_scenarios_are_ordered():
    assert [scenario.name for scenario in get_cost_scenarios()] == ["low", "base", "high"]


def test_estimate_annotation_cost_includes_selection_overhead():
    annotation_cost, selection_cost, total_cost = estimate_annotation_cost(
        AnnotationStrategy.HYBRID_AL,
        100,
        COST_SCENARIOS["base"],
    )

    assert annotation_cost == pytest.approx(9.52)
    assert selection_cost == pytest.approx(0.4)
    assert total_cost == pytest.approx(9.92)


def test_cost_scenario_keeps_pricing_source_metadata():
    scenario = COST_SCENARIOS["base"]

    assert scenario.source_name == "Prolific"
    assert scenario.source_url == "https://www.prolific.com/pricing"
    assert scenario.checked_at == "2026-07-12"


def test_estimate_llm_annotation_cost_uses_token_prices():
    pricing = LlmTokenPricing(
        source_name="Example API",
        source_url="https://example.com/pricing",
        checked_at="2026-07-12",
        model="example-model",
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=4.0,
    )

    cost = estimate_llm_annotation_cost(2_000, 500, pricing)

    assert cost == pytest.approx(0.004)


def test_estimate_llm_annotation_cost_rejects_negative_counts():
    pricing = LlmTokenPricing(
        source_name="Example API",
        source_url="https://example.com/pricing",
        checked_at="2026-07-12",
        model="example-model",
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=4.0,
    )

    with pytest.raises(ValueError):
        estimate_llm_annotation_cost(-1, 0, pricing)


def test_unknown_cost_scenario_raises():
    with pytest.raises(ValueError):
        get_cost_scenarios(["not_a_scenario"])
