"""Tests for real text-classification pilot helpers."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from annotatebench.core import AnnotationStrategy, LearningCurve
from annotatebench.pilot import (
    TextClassificationDataset,
    load_text_classification_csv,
    run_text_classification_pilot,
    run_text_classification_pilot_table,
    select_budget_indices,
)


def _dataset() -> TextClassificationDataset:
    return TextClassificationDataset(
        name="toy-finance",
        train_texts=[
            "profits increased after strong revenue growth",
            "earnings beat analyst expectations",
            "shares rose after the positive forecast",
            "losses widened during the quarter",
            "revenue fell and margins declined",
            "the company warned about weak demand",
            "guidance stayed unchanged for the year",
            "analysts described the report as neutral",
        ],
        train_labels=[
            "positive",
            "positive",
            "positive",
            "negative",
            "negative",
            "negative",
            "neutral",
            "neutral",
        ],
        test_texts=[
            "revenue growth lifted profits",
            "demand weakened and revenue fell",
            "the outlook was unchanged",
        ],
        test_labels=["positive", "negative", "neutral"],
    )


def test_load_text_classification_csv(tmp_path):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    train.write_text("text,label\nhello,pos\nbad,neg\n", encoding="utf-8")
    test.write_text("text,label\nokay,pos\n", encoding="utf-8")

    dataset = load_text_classification_csv(train, test, dataset_name="fixture")

    assert dataset.name == "fixture"
    assert dataset.train_texts == ["hello", "bad"]
    assert dataset.test_labels == ["pos"]


def test_select_budget_indices_is_reproducible():
    dataset = _dataset()

    first = select_budget_indices(
        dataset.train_texts,
        dataset.train_labels,
        budget=4,
        strategy=AnnotationStrategy.RANDOM,
        seed=7,
    )
    second = select_budget_indices(
        dataset.train_texts,
        dataset.train_labels,
        budget=4,
        strategy=AnnotationStrategy.RANDOM,
        seed=7,
    )

    assert first == second
    assert len(first) == 4


def test_run_text_classification_pilot_returns_curves():
    curves = run_text_classification_pilot(
        _dataset(),
        strategies=[AnnotationStrategy.RANDOM, AnnotationStrategy.UNCERTAINTY_AL],
        budgets=[3, 5],
        seed=3,
    )

    assert len(curves) == 2
    assert all(isinstance(curve, LearningCurve) for curve in curves)
    assert all(len(curve.points) == 2 for curve in curves)
    assert all(0.0 <= point.f1 <= 1.0 for curve in curves for point in curve.points)
    assert curves[0].config.dataset_name == "toy-finance"


def test_run_text_classification_pilot_table_columns():
    results = run_text_classification_pilot_table(
        _dataset(),
        strategies=[AnnotationStrategy.RANDOM],
        budgets=[3],
        seeds=[0],
    )

    assert list(results.columns) == [
        "dataset",
        "strategy",
        "budget",
        "seed",
        "macro_f1",
        "accuracy",
        "cost_scenario",
        "cost_source",
        "cost_source_url",
        "cost_checked_at",
        "human_cost_per_label",
        "selection_cost_per_example",
        "annotation_cost",
        "selection_cost",
        "total_cost",
    ]
    assert len(results) == 3
    assert sorted(results["cost_scenario"].tolist()) == ["base", "high", "low"]
    assert set(results["cost_checked_at"].tolist()) == {"2026-07-12"}


def test_run_text_classification_pilot_table_can_select_cost_scenarios():
    results = run_text_classification_pilot_table(
        _dataset(),
        strategies=[AnnotationStrategy.HYBRID_AL],
        budgets=[3],
        seeds=[0],
        cost_scenarios=["base"],
    )

    assert len(results) == 1
    assert results.loc[0, "cost_scenario"] == "base"
    assert results.loc[0, "selection_cost"] > 0


def test_budget_larger_than_train_set_uses_all_examples():
    dataset = _dataset()

    selected = select_budget_indices(
        dataset.train_texts,
        dataset.train_labels,
        budget=100,
        strategy=AnnotationStrategy.DIVERSITY_AL,
    )

    assert selected == list(range(len(dataset.train_texts)))
