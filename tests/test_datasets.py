"""Tests for public dataset loaders."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from annotatebench.datasets import (
    BENCHMARK_DATASETS,
    _cap_examples,
    _normalize_hf_label,
    load_banking77,
    load_benchmark_dataset,
    load_financial_phrasebank,
    load_trec,
)


def test_load_financial_phrasebank_from_original_text(tmp_path):
    path = tmp_path / "Sentences_AllAgree.txt"
    path.write_text(
        "\n".join(
            [
                "Profit rose in the quarter@positive",
                "The company issued no forecast change@neutral",
                "Revenue declined sharply@negative",
                "Margins improved year over year@positive",
                "The report was in line with expectations@neutral",
                "Shares fell after the warning@negative",
                "Demand remained stable@neutral",
                "Earnings exceeded estimates@positive",
                "Losses widened during the year@negative",
            ]
        ),
        encoding="latin-1",
    )

    dataset = load_financial_phrasebank(path, test_size=0.34, seed=7)

    assert dataset.name == "financial_phrasebank"
    assert dataset.train_texts
    assert dataset.test_texts
    assert sorted(set(dataset.train_labels + dataset.test_labels)) == ["negative", "neutral", "positive"]


def test_load_trec_from_local_splits(tmp_path):
    train_path = tmp_path / "trec_train.csv"
    test_path = tmp_path / "trec_test.csv"
    train_path.write_text(
        "text,coarse_label\n"
        "What is the capital of France?,5\n"
        "Who wrote Hamlet?,3\n",
        encoding="utf-8",
    )
    test_path.write_text(
        "text,coarse_label\n"
        "Where is the Eiffel Tower?,4\n",
        encoding="utf-8",
    )

    dataset = load_trec(train_path, test_path)

    assert dataset.name == "trec"
    assert dataset.train_labels == ["NUM", "HUM"]
    assert dataset.test_labels == ["LOC"]
    assert dataset.label_names == ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"]


def test_load_banking77_from_local_splits(tmp_path):
    train_path = tmp_path / "banking77_train.csv"
    test_path = tmp_path / "banking77_test.csv"
    train_path.write_text(
        "text,category\n"
        "How do I reset my PIN?,cash_withdrawal\n"
        "My card has not arrived,card_arrival\n",
        encoding="utf-8",
    )
    test_path.write_text(
        "text,category\n"
        "Can I change my cash withdrawal limit?,cash_withdrawal\n",
        encoding="utf-8",
    )

    dataset = load_banking77(train_path, test_path)

    assert dataset.name == "banking77"
    assert dataset.train_labels == ["cash_withdrawal", "card_arrival"]
    assert dataset.test_labels == ["cash_withdrawal"]


def test_benchmark_registry_has_ten_unique_datasets():
    assert len(BENCHMARK_DATASETS) == 10
    assert len(set(BENCHMARK_DATASETS)) == 10
    assert "financial_phrasebank" in BENCHMARK_DATASETS
    assert "twenty_newsgroups" in BENCHMARK_DATASETS


def test_cap_examples_is_reproducible_and_stratified():
    texts = [f"text {idx}" for idx in range(12)]
    labels = ["a"] * 6 + ["b"] * 6

    capped_texts, capped_labels = _cap_examples(texts, labels, max_examples=6, seed=3)
    capped_texts_again, capped_labels_again = _cap_examples(texts, labels, max_examples=6, seed=3)

    assert capped_texts == capped_texts_again
    assert capped_labels == capped_labels_again
    assert len(capped_texts) == 6
    assert set(capped_labels) == {"a", "b"}


def test_normalize_hf_label_uses_class_names():
    assert _normalize_hf_label(1, ["negative", "positive"]) == "positive"
    assert _normalize_hf_label("custom", None) == "custom"


def test_load_benchmark_dataset_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown benchmark dataset"):
        load_benchmark_dataset("not_a_dataset")
