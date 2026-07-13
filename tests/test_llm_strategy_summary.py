"""Tests for LLM strategy comparison summaries."""
from __future__ import annotations

import os
import subprocess
import sys

import pandas as pd


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_make_llm_strategy_summary_supports_multiple_seeds(tmp_path):
    gold_path = tmp_path / "gold.csv"
    llm_path = tmp_path / "llm.csv"
    output_path = tmp_path / "summary.csv"
    pd.DataFrame(
        [
            {
                "dataset": "fixture",
                "strategy": "random",
                "budget": 50,
                "seed": 0,
                "macro_f1": 0.5,
                "accuracy": 0.6,
                "cost_scenario": "base",
            },
            {
                "dataset": "fixture",
                "strategy": "hybrid_al",
                "budget": 50,
                "seed": 1,
                "macro_f1": 0.7,
                "accuracy": 0.8,
                "cost_scenario": "base",
            },
        ]
    ).to_csv(gold_path, index=False)
    pd.DataFrame(
        [
            {
                "dataset": "fixture",
                "strategy": "llm_annotator",
                "budget": 50,
                "seed": 0,
                "macro_f1": 0.4,
                "accuracy": 0.5,
                "cost_scenario": "base",
                "llm_label_accuracy": 0.9,
                "llm_label_macro_f1": 0.9,
                "llm_ece": 0.1,
                "llm_lari": 0.8,
                "model_name": "fixture-model",
                "prompt_version": "fixture-prompt",
            },
            {
                "dataset": "fixture",
                "strategy": "llm_annotator",
                "budget": 50,
                "seed": 1,
                "macro_f1": 0.6,
                "accuracy": 0.7,
                "cost_scenario": "base",
                "llm_label_accuracy": 0.8,
                "llm_label_macro_f1": 0.8,
                "llm_ece": 0.2,
                "llm_lari": 0.6,
                "model_name": "fixture-model",
                "prompt_version": "fixture-prompt",
            },
        ]
    ).to_csv(llm_path, index=False)

    subprocess.run(
        [
            sys.executable,
            os.path.join(ROOT, "scripts", "make_llm_strategy_summary.py"),
            "--gold-results",
            str(gold_path),
            "--llm-results",
            str(llm_path),
            "--seeds",
            "0,1",
            "--output-csv",
            str(output_path),
        ],
        cwd=ROOT,
        check=True,
    )

    summary = pd.read_csv(output_path)
    assert summary["seed"].tolist() == [0, 1]
    assert summary["best_gold_macro_f1"].tolist() == [0.5, 0.7]
