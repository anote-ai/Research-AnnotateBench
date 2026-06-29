"""Tests for public dataset loaders."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from annotatebench.datasets import load_banking77, load_financial_phrasebank, load_trec


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
