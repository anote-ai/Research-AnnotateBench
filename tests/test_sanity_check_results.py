"""Tests for result sanity checks."""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from sanity_check_results import sanity_check_results


def test_sanity_check_results_flags_duplicate_keys_and_metric_range(tmp_path):
    results_path = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {
                "dataset": "fixture",
                "strategy": "random",
                "budget": 50,
                "seed": 0,
                "cost_scenario": "base",
                "macro_f1": 0.4,
            },
            {
                "dataset": "fixture",
                "strategy": "random",
                "budget": 50,
                "seed": 0,
                "cost_scenario": "base",
                "macro_f1": 1.2,
            },
        ]
    ).to_csv(results_path, index=False)

    report = sanity_check_results([results_path])

    assert any(report["check"] == "duplicate_keys")
    assert any(report["check"] == "macro_f1_range")
    assert any(report["level"] == "fail")


def test_sanity_check_results_compares_baseline_key_coverage(tmp_path):
    results_path = tmp_path / "current.csv"
    baseline_path = tmp_path / "baseline.csv"
    pd.DataFrame(
        [
            {
                "dataset": "fixture",
                "strategy": "random",
                "budget": 50,
                "seed": 0,
                "macro_f1": 0.4,
            }
        ]
    ).to_csv(results_path, index=False)
    pd.DataFrame(
        [
            {
                "dataset": "fixture",
                "strategy": "hybrid_al",
                "budget": 50,
                "seed": 0,
                "macro_f1": 0.5,
            }
        ]
    ).to_csv(baseline_path, index=False)

    report = sanity_check_results([results_path], baseline_path=baseline_path)

    coverage = report[report["check"] == "baseline_key_coverage"].iloc[0]
    assert coverage["level"] == "warning"
    assert "1 keys only in current" in coverage["detail"]
