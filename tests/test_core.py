"""Tests for annotatebench.core."""

import pytest
from annotatebench.core import (
    AnnotationStrategy,
    NLPTask,
    BUDGET_LEVELS,
    ExperimentConfig,
    LearningCurvePoint,
    LearningCurve,
    fit_learning_curve,
)


def test_annotation_strategy_values():
    assert AnnotationStrategy.RANDOM.value == "random"
    assert AnnotationStrategy.UNCERTAINTY_AL.value == "uncertainty_al"
    assert AnnotationStrategy.DIVERSITY_AL.value == "diversity_al"
    assert AnnotationStrategy.HYBRID_AL.value == "hybrid_al"
    assert AnnotationStrategy.LLM_ANNOTATOR.value == "llm_annotator"


def test_nlp_task_values():
    assert NLPTask.CLASSIFICATION.value == "classification"
    assert NLPTask.NER.value == "ner"
    assert NLPTask.QA.value == "qa"
    assert NLPTask.SUMMARIZATION.value == "summarization"
    assert NLPTask.INSTRUCTION_TUNING.value == "instruction_tuning"


def test_budget_levels():
    assert BUDGET_LEVELS == [50, 100, 250, 500, 1000]
    assert len(BUDGET_LEVELS) == 5


def test_experiment_config_construction():
    config = ExperimentConfig(
        strategy=AnnotationStrategy.RANDOM,
        task=NLPTask.CLASSIFICATION,
        budget=100,
        model_type="bert-base",
        dataset_name="sst2",
    )
    assert config.strategy == AnnotationStrategy.RANDOM
    assert config.budget == 100


def test_learning_curve_point_construction():
    pt = LearningCurvePoint(budget=100, f1=0.72, cost_usd=5.0)
    assert pt.budget == 100
    assert pt.f1 == 0.72
    assert pt.cost_usd == 5.0


def test_learning_curve_construction():
    config = ExperimentConfig(
        strategy=AnnotationStrategy.HYBRID_AL,
        task=NLPTask.NER,
        budget=250,
        model_type="roberta",
        dataset_name="conll2003",
    )
    curve = LearningCurve(
        config=config,
        points=[
            LearningCurvePoint(50, 0.5, 2.0),
            LearningCurvePoint(100, 0.6, 4.0),
            LearningCurvePoint(250, 0.75, 10.0),
        ],
    )
    assert len(curve.points) == 3
    assert curve.config.task == NLPTask.NER


def test_fit_learning_curve_known():
    # y = 0.5 * x^0.3 — sample a few points
    a_true, b_true = 0.5, 0.3
    budgets = [50, 100, 250, 500, 1000]
    f1s = [a_true * b ** b_true for b in budgets]
    a_fit, b_fit = fit_learning_curve(budgets, f1s)
    assert abs(a_fit - a_true) < 0.05
    assert abs(b_fit - b_true) < 0.05


def test_fit_learning_curve_insufficient_data():
    with pytest.raises(ValueError, match="At least 2"):
        fit_learning_curve([100], [0.7])


def test_fit_learning_curve_mismatch():
    with pytest.raises(ValueError, match="same length"):
        fit_learning_curve([50, 100], [0.5])
