"""Tests for paper-ready benchmark summary tables."""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from make_paper_summary_table import make_paper_summary_table


def test_make_paper_summary_table_selects_best_base_row(tmp_path):
    results = tmp_path / "benchmark_results.csv"
    pd.DataFrame(
        [
            {
                "dataset": "fixture",
                "strategy": "random",
                "budget": 50,
                "seed": 0,
                "macro_f1": 0.40,
                "accuracy": 0.50,
                "cost_scenario": "base",
                "total_cost": 5.0,
            },
            {
                "dataset": "fixture",
                "strategy": "hybrid_al",
                "budget": 100,
                "seed": 0,
                "macro_f1": 0.70,
                "accuracy": 0.80,
                "cost_scenario": "base",
                "total_cost": 10.4,
            },
            {
                "dataset": "fixture",
                "strategy": "random",
                "budget": 50,
                "seed": 0,
                "macro_f1": 0.99,
                "accuracy": 0.99,
                "cost_scenario": "high",
                "total_cost": 20.0,
            },
        ]
    ).to_csv(results, index=False)

    output = tmp_path / "paper_core_summary.csv"
    summary = make_paper_summary_table(results, output, cost_scenario="base")

    assert output.exists()
    assert list(summary.columns) == [
        "dataset",
        "best_strategy",
        "best_budget",
        "mean_macro_f1",
        "std_macro_f1",
        "ci_lower_95",
        "ci_upper_95",
        "mean_accuracy",
        "mean_total_cost",
        "n_seeds",
        "cost_scenario",
    ]
    assert summary.loc[0, "best_strategy"] == "hybrid_al"
    assert summary.loc[0, "best_budget"] == 100
    assert summary.loc[0, "mean_macro_f1"] == 0.7
    assert summary.loc[0, "cost_scenario"] == "base"
