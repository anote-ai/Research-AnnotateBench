"""Tests for figure generation."""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from make_figures import make_figures


def test_make_figures_writes_pngs(tmp_path):
    results = tmp_path / "pilot_results.csv"
    pd.DataFrame(
        [
            {
                "dataset": "fixture",
                "strategy": "random",
                "budget": 5,
                "seed": 0,
                "macro_f1": 0.4,
                "accuracy": 0.5,
                "cost_scenario": "base",
                "annotation_cost": 0.5,
                "selection_cost": 0.0,
                "total_cost": 0.5,
            },
            {
                "dataset": "fixture",
                "strategy": "hybrid_al",
                "budget": 10,
                "seed": 0,
                "macro_f1": 0.6,
                "accuracy": 0.7,
                "cost_scenario": "base",
                "annotation_cost": 1.0,
                "selection_cost": 0.04,
                "total_cost": 1.0,
            },
        ]
    ).to_csv(results, index=False)

    written = make_figures(results, tmp_path / "figures")

    assert len(written) == 2
    assert all(path.exists() for path in written)
