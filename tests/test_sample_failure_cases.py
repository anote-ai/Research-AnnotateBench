"""Tests for qualitative failure-case sampling."""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from sample_failure_cases import sample_failure_cases


def test_sample_failure_cases_samples_incorrect_rows_by_group(tmp_path):
    annotations_path = tmp_path / "annotations.csv"
    pd.DataFrame(
        [
            {
                "dataset_name": "fixture",
                "example_id": "a",
                "model_name": "model-a",
                "prompt_version": "prompt-a",
                "gold_label": "yes",
                "predicted_label": "no",
                "confidence": 0.9,
                "correct": False,
            },
            {
                "dataset_name": "fixture",
                "example_id": "b",
                "model_name": "model-a",
                "prompt_version": "prompt-a",
                "gold_label": "yes",
                "predicted_label": "yes",
                "confidence": 0.8,
                "correct": True,
            },
            {
                "dataset_name": "fixture",
                "example_id": "c",
                "model_name": "model-b",
                "prompt_version": "prompt-a",
                "gold_label": "no",
                "predicted_label": "yes",
                "confidence": 0.7,
                "correct": False,
            },
        ]
    ).to_csv(annotations_path, index=False)

    samples = sample_failure_cases([annotations_path], n_per_group=1, seed=0)

    assert samples["example_id"].tolist() == ["a", "c"]
    assert set(samples["source_file"]) == {str(annotations_path)}
