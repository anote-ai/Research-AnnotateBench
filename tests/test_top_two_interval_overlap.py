from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from make_top_two_interval_overlap import make_overlap_table


def test_top_two_overlap_uses_student_t_intervals():
    rows = []
    for strategy, scores in {
        "random": [0.50, 0.51, 0.52, 0.53, 0.54],
        "hybrid_al": [0.51, 0.52, 0.53, 0.54, 0.55],
    }.items():
        for seed, score in enumerate(scores):
            rows.append(
                {
                    "dataset": "fixture",
                    "strategy": strategy,
                    "budget": 50,
                    "seed": seed,
                    "macro_f1": score,
                    "accuracy": score,
                    "total_cost": 5.0,
                    "cost_scenario": "base",
                }
            )

    output = make_overlap_table(pd.DataFrame(rows))

    assert bool(output.loc[0, "intervals_overlap"])
