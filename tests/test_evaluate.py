"""Tests for annotatebench.evaluate — 8 tests."""
from __future__ import annotations

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from annotatebench.core import (
    AnnotationStrategy,
    NLPTask,
    BUDGET_LEVELS,
    ExperimentConfig,
    LearningCurvePoint,
    LearningCurve,
)
from annotatebench.evaluate import (
    pareto_frontier,
    area_under_learning_curve,
    cost_to_target_f1,
    strategy_comparison,
    budget_recommendation,
)


def _make_curve(strategy=AnnotationStrategy.RANDOM, f1s=None, costs=None):
    if f1s is None:
        f1s = [0.5, 0.6, 0.7, 0.8, 0.85]
    if costs is None:
        costs = [5.0, 10.0, 25.0, 50.0, 100.0]
    config = ExperimentConfig(
        strategy=strategy,
        task=NLPTask.CLASSIFICATION,
        budget=1000,
        model_type="bert",
        dataset_name="test",
    )
    pts = [LearningCurvePoint(budget=b, f1=f, cost_usd=c)
           for b, f, c in zip(BUDGET_LEVELS, f1s, costs)]
    return LearningCurve(config=config, points=pts)


def test_pareto_frontier_dominated_removed():
    # (10, 0.5) is dominated by (8, 0.7)
    pts = [(10.0, 0.5), (8.0, 0.7), (12.0, 0.6)]
    pf = pareto_frontier(pts)
    costs = [p[0] for p in pf]
    assert 10.0 not in costs
    assert 8.0 in costs


def test_pareto_frontier_all_non_dominated():
    # Each point has strictly lower cost or higher f1
    pts = [(5.0, 0.9), (10.0, 0.95), (20.0, 0.99)]
    pf = pareto_frontier(pts)
    assert len(pf) == 3


def test_area_under_learning_curve_monotone():
    budgets = [50, 100, 250, 500, 1000]
    f1s = [0.5, 0.6, 0.7, 0.8, 0.85]
    auc = area_under_learning_curve(budgets, f1s)
    assert auc > 0


def test_cost_to_target_f1_found():
    curve = _make_curve(f1s=[0.5, 0.6, 0.7, 0.8, 0.85])
    cost = cost_to_target_f1(curve, target_f1=0.75)
    assert cost is not None
    assert cost > 0


def test_cost_to_target_f1_not_found():
    curve = _make_curve(f1s=[0.5, 0.6, 0.7, 0.8, 0.85])
    cost = cost_to_target_f1(curve, target_f1=0.99)
    assert cost is None


def test_strategy_comparison_keys():
    curves = [
        _make_curve(AnnotationStrategy.RANDOM),
        _make_curve(AnnotationStrategy.UNCERTAINTY_AL),
    ]
    comp = strategy_comparison(curves)
    assert "random" in comp
    assert "uncertainty_al" in comp


def test_strategy_comparison_efficiency():
    curve = _make_curve(AnnotationStrategy.RANDOM, f1s=[0.5, 0.6, 0.7, 0.8, 0.85])
    comp = strategy_comparison([curve])
    info = comp["random"]
    expected_eff = info["best_f1"] / info["total_cost"]
    assert abs(info["efficiency"] - expected_eff) < 1e-9


def test_budget_recommendation():
    curve = _make_curve(AnnotationStrategy.RANDOM, f1s=[0.5, 0.6, 0.7, 0.8, 0.85])
    recs = budget_recommendation([curve], target_f1=0.75)
    assert "random" in recs
    assert recs["random"] is not None
