from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from make_matched_gold_summary import expected_selected_indices, make_matched_gold_tables


def test_budget_250_nested_draw_matches_independent_random_selection():
    assert len(expected_selected_indices("fixture", seed=2, budget=250)) == 250


def test_matched_gold_summary_uses_random_rows_and_sample_sd():
    gold = pd.DataFrame(
        [
            {
                "dataset": "fixture",
                "budget": 250,
                "seed": seed,
                "strategy": strategy,
                "macro_f1": score,
                "accuracy": score,
                "cost_scenario": "base",
            }
            for seed, random_score, best_score in [(0, 0.5, 0.7), (1, 0.6, 0.8), (2, 0.7, 0.9)]
            for strategy, score in [("random", random_score), ("hybrid_al", best_score)]
        ]
    )
    llm = pd.DataFrame(
        [
            {
                "dataset": "fixture",
                "budget": 250,
                "seed": seed,
                "strategy": "llm_annotator",
                "macro_f1": score,
                "accuracy": score,
                "llm_label_accuracy": 0.9,
                "cost_scenario": "base",
            }
            for seed, score in [(0, 0.4), (1, 0.65), (2, 0.8)]
        ]
    )

    per_seed, summary = make_matched_gold_tables(
        gold,
        llm,
        seeds=[0, 1, 2],
        budget=250,
        cost_scenario="base",
    )

    assert per_seed["matched_gold_macro_f1"].tolist() == [0.5, 0.6, 0.7]
    assert per_seed["best_selector_macro_f1"].tolist() == [0.7, 0.8, 0.9]
    assert per_seed["paired_gap"].tolist() == pytest.approx([-0.1, 0.05, 0.1])
    assert summary.loc[0, "paired_gap_sd"] == pytest.approx(
        pd.Series([-0.1, 0.05, 0.1]).std(ddof=1)
    )
