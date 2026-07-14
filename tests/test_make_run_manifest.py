"""Tests for run manifest generation."""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from make_run_manifest import make_run_manifest


def test_make_run_manifest_summarizes_result_file(tmp_path):
    results_path = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {
                "dataset": "fixture",
                "strategy": "random",
                "budget": 50,
                "seed": 0,
                "downstream_model": "sentence_transformer_logreg",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "macro_f1": 0.4,
            },
            {
                "dataset": "fixture",
                "strategy": "hybrid_al",
                "budget": 100,
                "seed": 1,
                "downstream_model": "sentence_transformer_logreg",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "macro_f1": 0.6,
            },
        ]
    ).to_csv(results_path, index=False)

    manifest = make_run_manifest([results_path])

    assert manifest.loc[0, "n_rows"] == 2
    assert manifest.loc[0, "n_datasets"] == 1
    assert manifest.loc[0, "n_strategies"] == 2
    assert manifest.loc[0, "macro_f1_min"] == 0.4
    assert manifest.loc[0, "macro_f1_max"] == 0.6
