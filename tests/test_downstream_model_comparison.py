"""Tests for downstream-model comparison summaries."""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from make_downstream_model_comparison import make_downstream_model_comparison


def test_make_downstream_model_comparison_adds_baseline_deltas(tmp_path):
    baseline = tmp_path / "baseline.csv"
    embeddings = tmp_path / "embeddings.csv"
    summary_csv = tmp_path / "summary.csv"
    best_csv = tmp_path / "best.csv"

    pd.DataFrame(
        [
            {
                "dataset": "fixture",
                "strategy": "random",
                "budget": 50,
                "seed": 0,
                "macro_f1": 0.50,
                "accuracy": 0.60,
                "cost_scenario": "base",
                "total_cost": 4.76,
            },
            {
                "dataset": "fixture",
                "strategy": "hybrid_al",
                "budget": 100,
                "seed": 0,
                "macro_f1": 0.70,
                "accuracy": 0.80,
                "cost_scenario": "base",
                "total_cost": 9.92,
            },
        ]
    ).to_csv(baseline, index=False)
    pd.DataFrame(
        [
            {
                "dataset": "fixture",
                "downstream_model": "sentence_transformer_logreg",
                "strategy": "random",
                "budget": 50,
                "seed": 0,
                "macro_f1": 0.75,
                "accuracy": 0.82,
                "cost_scenario": "base",
                "total_cost": 4.76,
            },
        ]
    ).to_csv(embeddings, index=False)

    summary, best = make_downstream_model_comparison(
        [baseline, embeddings],
        summary_csv,
        best_csv,
    )

    assert set(summary["downstream_model"]) == {
        "tfidf_logreg",
        "sentence_transformer_logreg",
    }
    embedding_best = best[best["downstream_model"] == "sentence_transformer_logreg"].iloc[0]
    assert embedding_best["delta_macro_f1_vs_baseline"] == 0.05
    assert summary_csv.exists()
    assert best_csv.exists()
