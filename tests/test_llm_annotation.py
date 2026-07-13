from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from annotatebench.llm import (
    build_user_prompt,
    dry_run_annotation,
    load_prompt,
    normalize_annotation,
    normalize_label,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_phrasebank_fixture(path: Path) -> None:
    path.write_text(
        "text,label\n"
        "profits increased,positive\n"
        "shares rose,positive\n"
        "earnings beat estimates,positive\n"
        "revenue growth accelerated,positive\n"
        "the outlook improved,positive\n"
        "losses widened,negative\n"
        "sales fell,negative\n"
        "margins declined,negative\n"
        "the warning hurt shares,negative\n"
        "demand weakened,negative\n"
        "guidance unchanged,neutral\n"
        "the board met,neutral\n"
        "the report was factual,neutral\n"
        "the company repeated guidance,neutral\n"
        "analysts waited for updates,neutral\n",
        encoding="utf-8",
    )


def test_generic_prompt_loads_and_formats():
    system_prompt, user_template = load_prompt(ROOT / "prompts", "text_classification_v1_zero_shot_json")

    assert "text classification annotator" in system_prompt
    user_prompt = build_user_prompt(
        user_template,
        text="Stocks rose after earnings.",
        dataset_name="fixture",
        label_names=["negative", "positive"],
    )

    assert "fixture" in user_prompt
    assert "- negative" in user_prompt
    assert "Stocks rose" in user_prompt


def test_generic_prompt_includes_dataset_label_descriptions():
    _, user_template = load_prompt(ROOT / "prompts", "text_classification_v1_zero_shot_json")

    user_prompt = build_user_prompt(
        user_template,
        text="A charming and funny film.",
        dataset_name="sst2",
        label_names=["0", "1"],
    )

    assert "0: negative movie-review sentiment" in user_prompt
    assert "1: positive movie-review sentiment" in user_prompt


def test_yelp_prompt_uses_one_two_label_mapping():
    _, user_template = load_prompt(ROOT / "prompts", "text_classification_v1_zero_shot_json")

    user_prompt = build_user_prompt(
        user_template,
        text="Great food and fast service.",
        dataset_name="yelp_polarity",
        label_names=["1", "2"],
    )

    assert "1: negative review sentiment" in user_prompt
    assert "2: positive review sentiment" in user_prompt


def test_twenty_newsgroups_prompt_includes_category_meanings():
    _, user_template = load_prompt(ROOT / "prompts", "text_classification_v1_zero_shot_json")

    user_prompt = build_user_prompt(
        user_template,
        text="The graphics driver fails under X.",
        dataset_name="twenty_newsgroups",
        label_names=["1", "5", "14"],
    )

    assert "1: comp.graphics, computer graphics" in user_prompt
    assert "5: comp.windows.x, X Window System" in user_prompt
    assert "14: sci.space, space and astronomy" in user_prompt


def test_prompt_label_preflight_runs_for_financial_phrasebank(tmp_path):
    data_path = tmp_path / "phrasebank.csv"
    _write_phrasebank_fixture(data_path)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_llm_prompt_labels.py"),
            "--datasets",
            "financial_phrasebank",
            "--financial-phrasebank-path",
            str(data_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "OK financial_phrasebank" in result.stdout


def test_normalize_label_accepts_case_insensitive_exact_labels():
    assert normalize_label("Positive", ["negative", "positive"]) == "positive"
    assert normalize_label("Sci/Tech", ["World", "Sci/Tech"]) == "Sci/Tech"


def test_normalize_label_rejects_unknown_label():
    with pytest.raises(ValueError):
        normalize_label("maybe", ["negative", "positive"])


def test_normalize_annotation_accepts_dry_run_annotation():
    parsed = normalize_annotation(dry_run_annotation(["negative", "positive"]), ["negative", "positive"])

    assert parsed.predicted_label == "negative"
    assert parsed.confidence == pytest.approx(0.34)
    assert parsed.input_tokens == 0
    assert parsed.output_tokens == 0
    assert parsed.total_tokens == 0
    assert parsed.cost_usd is None


def test_run_llm_annotation_dry_run_writes_summary(tmp_path):
    data_path = tmp_path / "phrasebank.csv"
    _write_phrasebank_fixture(data_path)
    annotations_path = tmp_path / "annotations.csv"
    summary_path = tmp_path / "summary.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_llm_annotation.py"),
            "--financial-phrasebank-path",
            str(data_path),
            "--limit",
            "2",
            "--dry-run",
            "--output-csv",
            str(annotations_path),
            "--summary-csv",
            str(summary_path),
        ],
        cwd=ROOT,
        check=True,
    )

    with summary_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["dataset_name"] == "financial_phrasebank"
    assert row["n_examples"] == "2"
    assert row["input_tokens"] == "0"
    assert row["output_tokens"] == "0"
    assert row["total_tokens"] == "0"
    assert "lari" in row

    with annotations_path.open(newline="", encoding="utf-8") as handle:
        annotation_row = next(csv.DictReader(handle))
    assert annotation_row["input_tokens"] == "0"
    assert annotation_row["output_tokens"] == "0"
    assert annotation_row["total_tokens"] == "0"


def test_run_llm_strategy_benchmark_dry_run_writes_llm_rows(tmp_path):
    data_path = tmp_path / "phrasebank.csv"
    _write_phrasebank_fixture(data_path)
    output_path = tmp_path / "benchmark_with_llm.csv"
    annotation_dir = tmp_path / "annotations"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_llm_strategy_benchmark.py"),
            "--datasets",
            "financial_phrasebank",
            "--budgets",
            "2",
            "--seeds",
            "0",
            "--cost-scenarios",
            "base",
            "--financial-phrasebank-path",
            str(data_path),
            "--dry-run",
            "--annotation-dir",
            str(annotation_dir),
            "--output-csv",
            str(output_path),
        ],
        cwd=ROOT,
        check=True,
    )

    with output_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    assert row["strategy"] == "llm_annotator"
    assert row["dataset"] == "financial_phrasebank"
    assert row["budget"] == "2"
    assert row["downstream_model"] == "tfidf_logreg"
    assert "llm_lari" in row
    assert list(annotation_dir.glob("*.csv"))


def test_run_llm_strategy_benchmark_resumes_cached_annotations(tmp_path):
    data_path = tmp_path / "phrasebank.csv"
    _write_phrasebank_fixture(data_path)
    output_path = tmp_path / "benchmark_with_llm.csv"
    annotation_dir = tmp_path / "annotations"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_llm_strategy_benchmark.py"),
        "--datasets",
        "financial_phrasebank",
        "--budgets",
        "2",
        "--seeds",
        "0",
        "--cost-scenarios",
        "base",
        "--financial-phrasebank-path",
        str(data_path),
        "--dry-run",
        "--annotation-dir",
        str(annotation_dir),
        "--output-csv",
        str(output_path),
    ]

    subprocess.run(command, cwd=ROOT, check=True)
    cache_path = next(annotation_dir.glob("*.csv"))
    rows = list(csv.DictReader(cache_path.open(newline="", encoding="utf-8")))
    rows[0]["predicted_label"] = "positive"
    rows[0]["confidence"] = "0.91"
    with cache_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    subprocess.run(command, cwd=ROOT, check=True)
    resumed_rows = list(csv.DictReader(cache_path.open(newline="", encoding="utf-8")))

    assert resumed_rows[0]["predicted_label"] == "positive"
    assert resumed_rows[0]["confidence"] == "0.91"


def test_run_llm_strategy_benchmark_uses_nested_seed_cache(tmp_path):
    data_path = tmp_path / "phrasebank.csv"
    _write_phrasebank_fixture(data_path)
    output_path = tmp_path / "benchmark_with_llm.csv"
    annotation_dir = tmp_path / "annotations"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_llm_strategy_benchmark.py"),
            "--datasets",
            "financial_phrasebank",
            "--budgets",
            "2,3",
            "--seeds",
            "0",
            "--cost-scenarios",
            "base",
            "--financial-phrasebank-path",
            str(data_path),
            "--dry-run",
            "--annotation-dir",
            str(annotation_dir),
            "--output-csv",
            str(output_path),
        ],
        cwd=ROOT,
        check=True,
    )

    cache_files = list(annotation_dir.glob("*.csv"))
    assert len(cache_files) == 1
    cache_rows = list(csv.DictReader(cache_files[0].open(newline="", encoding="utf-8")))
    assert len(cache_rows) == 3

    result_rows = list(csv.DictReader(output_path.open(newline="", encoding="utf-8")))
    assert [row["budget"] for row in result_rows] == ["2", "3"]
