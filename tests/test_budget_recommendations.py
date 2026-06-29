"""Tests for budget recommendation generation."""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from make_budget_recommendations import make_budget_recommendations


def test_make_budget_recommendations_filters_cost_scenario(tmp_path):
    results = tmp_path / "pilot_results.csv"
    pd.DataFrame(
        [
            {
                "dataset": "fixture",
                "strategy": "random",
                "budget": 50,
                "seed": 0,
                "macro_f1": 0.4,
                "accuracy": 0.5,
                "cost_scenario": "base",
                "total_cost": 5.0,
            },
            {
                "dataset": "fixture",
                "strategy": "hybrid_al",
                "budget": 100,
                "seed": 0,
                "macro_f1": 0.7,
                "accuracy": 0.8,
                "cost_scenario": "base",
                "total_cost": 10.4,
            },
            {
                "dataset": "fixture",
                "strategy": "random",
                "budget": 50,
                "seed": 0,
                "macro_f1": 0.4,
                "accuracy": 0.5,
                "cost_scenario": "high",
                "total_cost": 15.0,
            },
        ]
    ).to_csv(results, index=False)

    output = tmp_path / "recommendations.csv"
    recommendations = make_budget_recommendations(
        [results],
        output,
        targets=[0.6],
        cost_scenario="base",
    )

    assert output.exists()
    assert recommendations.loc[0, "cheapest_strategy"] == "hybrid_al"
    assert recommendations.loc[0, "estimated_cost"] == "10.40"
