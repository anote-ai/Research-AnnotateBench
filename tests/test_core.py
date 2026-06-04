"""Tests for annotatebench.core — 10 tests."""
from __future__ import annotations

import math
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
    fit_learning_curve,
    predict_f1,
)


def _make_curve(f1s=None):
    if f1s is None:
        f1s = [0.5, 0.6, 0.7, 0.8, 0.85]
    config = ExperimentConfig(
        strategy=AnnotationStrategy.RANDOM,
        task=NLPTask.CLASSIFICATION,
        budget=1000,
        model_type="bert-base",
        dataset_name="test",
    )
    pts = [LearningCurvePoint(budget=b, f1=f, cost_usd=b * 0.1) for b, f in zip(BUDGET_LEVELS, f1s)]
    return LearningCurve(config=config, points=pts)


def test_annotation_strategy_values():
    assert AnnotationStrategy.RANDOM == "random"
    assert AnnotationStrategy.LLM_ANNOTATOR == "llm_annotator"


def test_nlp_task_values():
    assert NLPTask.CLASSIFICATION == "classification"
    assert NLPTask.INSTRUCTION_TUNING == "instruction_tuning"


def test_budget_levels_has_5_items():
    assert len(BUDGET_LEVELS) == 5
    assert BUDGET_LEVELS[0] == 50
    assert BUDGET_LEVELS[-1] == 1000


def test_learning_curve_point_efficiency():
    pt = LearningCurvePoint(budget=100, f1=0.8, cost_usd=10.0)
    assert abs(pt.efficiency - 0.08) < 1e-9


def test_learning_curve_max_f1():
    curve = _make_curve([0.5, 0.6, 0.7, 0.8, 0.85])
    assert abs(curve.max_f1() - 0.85) < 1e-9


def test_learning_curve_total_cost():
    curve = _make_curve()
    expected = sum(b * 0.1 for b in BUDGET_LEVELS)
    assert abs(curve.total_cost() - expected) < 1e-6


def test_learning_curve_f1_at_budget_hit():
    curve = _make_curve([0.5, 0.6, 0.7, 0.8, 0.85])
    assert abs(curve.f1_at_budget(100) - 0.6) < 1e-9


def test_learning_curve_f1_at_budget_miss():
    curve = _make_curve()
    assert curve.f1_at_budget(999) is None


def test_fit_learning_curve_power_law():
    # y = 0.1 * x^0.3
    a_true, b_true = 0.1, 0.3
    budgets = [50, 100, 250, 500, 1000]
    f1s = [a_true * (x ** b_true) for x in budgets]
    a_fit, b_fit = fit_learning_curve(budgets, f1s)
    assert abs(a_fit - a_true) < 0.01
    assert abs(b_fit - b_true) < 0.01


def test_predict_f1():
    val = predict_f1(0.1, 0.5, 100)
    expected = 0.1 * (100 ** 0.5)
    assert abs(val - expected) < 1e-9
