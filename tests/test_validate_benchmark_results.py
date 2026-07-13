"""Tests for benchmark result validation."""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from annotatebench.datasets import BENCHMARK_DATASETS
from validate_benchmark_results import (
    EXPECTED_BUDGETS,
    EXPECTED_COST_SCENARIOS,
    EXPECTED_SEEDS,
    EXPECTED_STRATEGIES,
    validate_benchmark_results,
)


def _complete_results() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in BENCHMARK_DATASETS:
        for strategy in EXPECTED_STRATEGIES:
            for budget in EXPECTED_BUDGETS:
                for seed in EXPECTED_SEEDS:
                    for cost_scenario in EXPECTED_COST_SCENARIOS:
                        rows.append(
                            {
                                "dataset": dataset,
                                "strategy": strategy,
                                "budget": budget,
                                "seed": seed,
                                "cost_scenario": cost_scenario,
                                "macro_f1": 0.5,
                            }
                        )
    return pd.DataFrame(rows)


def test_validate_benchmark_results_accepts_complete_grid():
    assert validate_benchmark_results(_complete_results()) == []


def test_validate_benchmark_results_rejects_missing_dataset():
    df = _complete_results()
    df = df[df["dataset"] != BENCHMARK_DATASETS[0]]

    problems = validate_benchmark_results(df)

    assert any("Missing datasets" in problem for problem in problems)


def test_validate_benchmark_results_flags_suspicious_perfect_scores():
    df = _complete_results()
    dataset = BENCHMARK_DATASETS[0]
    df.loc[(df["dataset"] == dataset) & (df["cost_scenario"] == "base"), "macro_f1"] = 1.0

    problems = validate_benchmark_results(df)

    assert any("suspicious constant near-perfect macro F1" in problem for problem in problems)
