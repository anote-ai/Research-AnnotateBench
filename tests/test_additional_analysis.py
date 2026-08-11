from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_paired_strategy_stats import exact_sign_flip_p_value, make_statistics
from run_cumulative_trajectories import (
    SELECTION_STRATEGIES,
    build_trajectory,
    diversity_order,
)

from annotatebench.core import AnnotationStrategy
from annotatebench.pilot import select_budget_indices


def test_incremental_diversity_matches_reference_set() -> None:
    texts = [f"topic {index % 3} example {index}" for index in range(30)]
    expected = select_budget_indices(
        texts,
        [str(index % 3) for index in range(30)],
        budget=12,
        strategy=AnnotationStrategy.DIVERSITY_AL,
        seed=7,
    )
    assert sorted(diversity_order(texts, 12, 7)) == expected


def test_all_cumulative_trajectories_are_nested() -> None:
    texts = [f"class {index % 2} token {index}" for index in range(40)]
    labels = [str(index % 2) for index in range(40)]
    budgets = [5, 10, 20]
    for strategy in SELECTION_STRATEGIES:
        trajectory = build_trajectory(texts, labels, budgets, strategy, seed=3)
        assert set(trajectory[5]).issubset(trajectory[10])
        assert set(trajectory[10]).issubset(trajectory[20])


def test_paired_statistics_reports_exact_test_and_holm_adjustment() -> None:
    rows = []
    for seed in range(5):
        for strategy, score in (("random", 0.6), ("hybrid_al", 0.7)):
            rows.append(
                {
                    "dataset": "demo",
                    "strategy": strategy,
                    "budget": 100,
                    "seed": seed,
                    "macro_f1": score + seed * 0.001,
                    "cost_scenario": "base",
                    "total_cost": 10.0,
                }
            )
    result = make_statistics(pd.DataFrame(rows), bootstrap_resamples=500, seed=1)
    assert len(result) == 1
    assert result.loc[0, "paired_mean_difference"] == 0.1
    assert result.loc[0, "exact_sign_flip_p"] == 0.0625
    assert result.loc[0, "holm_adjusted_p"] == 0.0625


def test_exact_sign_flip_is_two_sided() -> None:
    assert exact_sign_flip_p_value(np.array([1.0, 1.0, 1.0])) == 0.25
