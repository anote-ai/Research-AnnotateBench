"""Tests for annotatebench.data."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from annotatebench.core import AnnotationStrategy, NLPTask, BUDGET_LEVELS
from annotatebench.data import (
    STRATEGY_F1_CURVES,
    make_learning_curve,
    make_full_dataset,
)


def test_make_learning_curve_has_all_budget_levels():
    curve = make_learning_curve(AnnotationStrategy.RANDOM, NLPTask.CLASSIFICATION, seed=1)
    budgets = [pt.budget for pt in curve.points]
    assert budgets == BUDGET_LEVELS


def test_make_learning_curve_f1_clamped_to_unit_interval():
    curve = make_learning_curve(AnnotationStrategy.HYBRID_AL, NLPTask.CLASSIFICATION, noise=5.0, seed=2)
    for pt in curve.points:
        assert 0.0 <= pt.f1 <= 1.0


def test_make_learning_curve_deterministic_given_seed():
    c1 = make_learning_curve(AnnotationStrategy.LLM_ANNOTATOR, NLPTask.NER, seed=7)
    c2 = make_learning_curve(AnnotationStrategy.LLM_ANNOTATOR, NLPTask.NER, seed=7)
    assert [pt.f1 for pt in c1.points] == [pt.f1 for pt in c2.points]


def test_make_full_dataset_size():
    curves = make_full_dataset(seed=42)
    assert len(curves) == len(list(AnnotationStrategy)) * len(list(NLPTask))


def test_ner_curve_distinct_from_classification():
    f1_class = STRATEGY_F1_CURVES[AnnotationStrategy.RANDOM.value]
    curve_ner = make_learning_curve(AnnotationStrategy.RANDOM, NLPTask.NER, noise=0.0, seed=0)
    assert [pt.f1 for pt in curve_ner.points] != f1_class


@pytest.mark.parametrize(
    "task",
    [NLPTask.QA, NLPTask.SUMMARIZATION, NLPTask.INSTRUCTION_TUNING],
)
def test_task_curves_distinct_from_classification_fallback(task):
    """Regression test: QA/Summarization/Instruction-Tuning previously fell
    back silently to the Classification curves. They must now have their own
    distinct reference curves for every strategy."""
    for strategy in AnnotationStrategy:
        class_curve = make_learning_curve(strategy, NLPTask.CLASSIFICATION, noise=0.0, seed=0)
        task_curve = make_learning_curve(strategy, task, noise=0.0, seed=0)
        class_f1s = [pt.f1 for pt in class_curve.points]
        task_f1s = [pt.f1 for pt in task_curve.points]
        assert task_f1s != class_f1s, f"{task} curve for {strategy} matches Classification fallback"


@pytest.mark.parametrize(
    "task",
    [NLPTask.QA, NLPTask.SUMMARIZATION, NLPTask.INSTRUCTION_TUNING],
)
def test_task_curves_monotonic_in_budget(task):
    """Reference curves should be non-decreasing in budget (more labels should
    not produce strictly worse expected F1 in the noise-free reference data)."""
    for strategy in AnnotationStrategy:
        curve = make_learning_curve(strategy, task, noise=0.0, seed=0)
        f1s = [pt.f1 for pt in curve.points]
        assert f1s == sorted(f1s)
