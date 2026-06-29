"""Tests for scenario-based annotation cost estimates."""
from __future__ import annotations

import pytest

from annotatebench.core import AnnotationStrategy
from annotatebench.costs import COST_SCENARIOS, estimate_annotation_cost, get_cost_scenarios


def test_default_cost_scenarios_are_ordered():
    assert [scenario.name for scenario in get_cost_scenarios()] == ["low", "base", "high"]


def test_estimate_annotation_cost_includes_selection_overhead():
    annotation_cost, selection_cost, total_cost = estimate_annotation_cost(
        AnnotationStrategy.HYBRID_AL,
        100,
        COST_SCENARIOS["base"],
    )

    assert annotation_cost == pytest.approx(10.0)
    assert selection_cost == pytest.approx(0.4)
    assert total_cost == pytest.approx(10.4)


def test_unknown_cost_scenario_raises():
    with pytest.raises(ValueError):
        get_cost_scenarios(["not_a_scenario"])
