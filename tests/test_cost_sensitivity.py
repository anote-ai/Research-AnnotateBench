from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from make_cost_sensitivity import make_cost_sensitivity
from make_figures import add_review_cost_sensitivity


def test_selection_overhead_factors_report_stable_frontier():
    results = pd.DataFrame(
        [
            {
                "dataset": "fixture",
                "strategy": "random",
                "budget": 50,
                "seed": seed,
                "macro_f1": 0.6,
                "annotation_cost": 5.0,
                "selection_cost": 0.0,
                "cost_scenario": "base",
            }
            for seed in range(2)
        ]
        + [
            {
                "dataset": "fixture",
                "strategy": "hybrid_al",
                "budget": 100,
                "seed": seed,
                "macro_f1": 0.8,
                "annotation_cost": 10.0,
                "selection_cost": 0.4,
                "cost_scenario": "base",
            }
            for seed in range(2)
        ]
    )

    output = make_cost_sensitivity(results, factors=[1, 2, 5, 10])

    assert output["frontier_stable_vs_1x"].all()
    assert set(output["overhead_factor"]) == {1, 2, 5, 10}


def test_ten_percent_review_cost_uses_full_precision_base_rate():
    summary = pd.DataFrame(
        [
            {
                "strategy": "llm_annotator",
                "budget": budget,
                "total_cost": 0.01,
                "macro_f1": 0.5,
                "macro_f1_std": 0.0,
                "cost_estimation_method": "measured",
            }
            for budget in [50, 100, 250]
        ]
    )

    output = add_review_cost_sensitivity(summary, review_fraction=0.1, human_cost_per_label=0.0952)
    reviewed = output[output["strategy"] == "llm_annotator_review_10pct"]

    assert reviewed["total_cost"].tolist() == pytest.approx([0.486, 0.962, 2.39])
    assert reviewed["frontier_eligible"].all()
    assert not output[output["strategy"] == "llm_annotator"]["frontier_eligible"].any()
