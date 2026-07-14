from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIELDNAMES = [
    "run_id",
    "dataset_name",
    "split",
    "seed",
    "task_type",
    "example_id",
    "difficulty_bucket",
    "annotator_type",
    "annotator_id",
    "model_name",
    "temperature",
    "prompt_version",
    "replicate_id",
    "gold_label",
    "predicted_label",
    "confidence",
    "correct",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cost_usd",
    "failure_category",
    "rationale",
    "notes",
    "raw_response",
]


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _row(example_id: str, annotator_id: str, gold: str, predicted: str) -> dict[str, object]:
    return {
        "run_id": "run-1",
        "dataset_name": "fixture",
        "split": "test",
        "seed": 0,
        "task_type": "classification",
        "example_id": example_id,
        "difficulty_bucket": "",
        "annotator_type": "llm",
        "annotator_id": annotator_id,
        "model_name": "model",
        "temperature": "0",
        "prompt_version": "prompt",
        "replicate_id": 0,
        "gold_label": gold,
        "predicted_label": predicted,
        "confidence": 0.8,
        "correct": gold == predicted,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost_usd": "",
        "failure_category": "",
        "rationale": "",
        "notes": "",
        "raw_response": "{}",
    }


def test_statistical_appendix_writes_metric_ci_and_insufficient_agreement_note(tmp_path):
    annotations_path = tmp_path / "annotations.csv"
    appendix_path = tmp_path / "appendix.csv"
    significance_path = tmp_path / "significance.csv"
    _write_rows(
        annotations_path,
        [
            _row("ex-1", "llm-a", "positive", "positive"),
            _row("ex-2", "llm-a", "negative", "positive"),
        ],
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_statistical_appendix.py"),
            "--annotations",
            str(annotations_path),
            "--output-csv",
            str(appendix_path),
            "--significance-csv",
            str(significance_path),
            "--n-resamples",
            "50",
        ],
        cwd=ROOT,
        check=True,
    )

    appendix_rows = list(csv.DictReader(appendix_path.open(newline="", encoding="utf-8")))
    significance_rows = list(csv.DictReader(significance_path.open(newline="", encoding="utf-8")))

    assert {"accuracy", "macro_f1", "ece", "lari"}.issubset({row["metric"] for row in appendix_rows})
    assert any(row["metric"] == "gwet_ac1" and row["notes"] == "insufficient_annotators" for row in appendix_rows)
    assert significance_rows == []


def test_statistical_appendix_writes_pairwise_agreement_and_significance(tmp_path):
    annotations_path = tmp_path / "annotations.csv"
    appendix_path = tmp_path / "appendix.csv"
    significance_path = tmp_path / "significance.csv"
    _write_rows(
        annotations_path,
        [
            _row("ex-1", "llm-a", "positive", "positive"),
            _row("ex-2", "llm-a", "negative", "negative"),
            _row("ex-1", "llm-b", "positive", "positive"),
            _row("ex-2", "llm-b", "negative", "positive"),
        ],
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_statistical_appendix.py"),
            "--annotations",
            str(annotations_path),
            "--output-csv",
            str(appendix_path),
            "--significance-csv",
            str(significance_path),
            "--n-resamples",
            "50",
        ],
        cwd=ROOT,
        check=True,
    )

    appendix_rows = list(csv.DictReader(appendix_path.open(newline="", encoding="utf-8")))
    significance_rows = list(csv.DictReader(significance_path.open(newline="", encoding="utf-8")))

    agreement_rows = [row for row in appendix_rows if row["metric"] == "gwet_ac1"]
    assert agreement_rows
    assert agreement_rows[0]["group_key"] == "annotator_pair"
    assert significance_rows
    assert significance_rows[0]["metric"] == "accuracy"
    assert significance_rows[0]["p_value_bonferroni"]
