"""Tests for annotatebench.data — 5 tests."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from annotatebench.core import AnnotationStrategy, NLPTask, BUDGET_LEVELS, LearningCurve
from annotatebench.data import make_learning_curve, make_full_dataset


def test_make_learning_curve_returns_learning_curve():
    curve = make_learning_curve(AnnotationStrategy.RANDOM, NLPTask.CLASSIFICATION)
    assert isinstance(curve, LearningCurve)
    assert len(curve.points) == 5


def test_all_f1_in_zero_one():
    curve = make_learning_curve(AnnotationStrategy.LLM_ANNOTATOR, NLPTask.QA)
    for pt in curve.points:
        assert 0.0 <= pt.f1 <= 1.0


def test_llm_annotator_f1_at_50_greater_than_random():
    llm = make_learning_curve(AnnotationStrategy.LLM_ANNOTATOR, NLPTask.CLASSIFICATION, seed=42)
    rnd = make_learning_curve(AnnotationStrategy.RANDOM, NLPTask.CLASSIFICATION, seed=42)
    llm_f1 = llm.f1_at_budget(50)
    rnd_f1 = rnd.f1_at_budget(50)
    assert llm_f1 is not None and rnd_f1 is not None
    assert llm_f1 > rnd_f1


def test_make_full_dataset_length():
    curves = make_full_dataset(seed=42)
    assert len(curves) == 25


def test_budget_levels_match():
    curve = make_learning_curve(AnnotationStrategy.HYBRID_AL, NLPTask.NER)
    budgets = [pt.budget for pt in curve.points]
    assert budgets == BUDGET_LEVELS
