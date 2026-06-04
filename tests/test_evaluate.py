"""Tests for annotatebench.evaluate."""

import pytest
from annotatebench.core import (
    AnnotationStrategy,
    NLPTask,
    ExperimentConfig,
    LearningCurve,
    LearningCurvePoint,
)
from annotatebench.evaluate import (
    pareto_frontier,
    area_under_learning_curve,
    cost_to_target_f1,
)


def test_pareto_frontier_basic():
    points = [(1.0, 0.5), (2.0, 0.6), (3.0, 0.55), (4.0, 0.7)]
    frontier = pareto_frontier(points)
    # (1.0, 0.5), (2.0, 0.6), (4.0, 0.7) are non-dominated
    assert (1.0, 0.5) in frontier
    assert (2.0, 0.6) in frontier
    assert (4.0, 0.7) in frontier
    assert (3.0, 0.55) not in frontier


def test_pareto_frontier_empty():
    assert pareto_frontier([]) == []


def test_pareto_frontier_single():
    assert pareto_frontier([(1.0, 0.8)]) == [(1.0, 0.8)]


def test_area_under_learning_curve():
    budgets = [100, 200, 400]
    f1s = [0.5, 0.6, 0.7]
    area = area_under_learning_curve(budgets, f1s)
    # Trapezoid: (200-100)*(0.5+0.6)/2 + (400-200)*(0.6+0.7)/2 = 55 + 130 = 185
    assert abs(area - 185.0) < 1e-9


def test_area_under_learning_curve_mismatch():
    with pytest.raises(ValueError):
        area_under_learning_curve([100, 200], [0.5])


def _make_curve(f1s: list[float], costs: list[float]) -> LearningCurve:
    config = ExperimentConfig(
        strategy=AnnotationStrategy.RANDOM,
        task=NLPTask.CLASSIFICATION,
        budget=0,
        model_type="test",
        dataset_name="test",
    )
    budgets = [50, 100, 250, 500, 1000][: len(f1s)]
    points = [LearningCurvePoint(b, f, c) for b, f, c in zip(budgets, f1s, costs)]
    return LearningCurve(config=config, points=points)


def test_cost_to_target_f1_interpolated():
    curve = _make_curve([0.5, 0.7], [10.0, 20.0])
    cost = cost_to_target_f1(curve, 0.6)
    # Midpoint between 0.5 and 0.7 => midpoint between 10 and 20 = 15
    assert cost is not None
    assert abs(cost - 15.0) < 1e-9


def test_cost_to_target_f1_not_achievable():
    curve = _make_curve([0.5, 0.65], [10.0, 20.0])
    cost = cost_to_target_f1(curve, 0.9)
    assert cost is None


def test_cost_to_target_f1_already_achieved():
    curve = _make_curve([0.8, 0.9], [5.0, 10.0])
    cost = cost_to_target_f1(curve, 0.75)
    assert cost == 5.0
