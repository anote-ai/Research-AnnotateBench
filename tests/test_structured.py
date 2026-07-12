from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from annotatebench import (
    NERExample,
    QAExample,
    RelationExtractionExample,
    SummarizationExample,
    bio_tags_to_spans,
    entity_level_f1,
    mean_rouge_l_f1,
    qa_exact_match,
    qa_scores,
    qa_token_f1,
    relation_macro_f1,
    rouge_l_f1,
)
from annotatebench.core import NLPTask


def test_structured_examples_capture_annotation_units():
    ner = NERExample(
        tokens=["EU", "rejects", "German", "call"],
        labels=["B-ORG", "O", "B-MISC", "O"],
    )
    qa = QAExample(question="Who rejected the call?", context="EU rejects German call.", answers=["EU"])
    summary = SummarizationExample(document="A long document.", summary="A short summary.")
    relation = RelationExtractionExample(
        context="Alice founded Acme.",
        head_entity="Alice",
        tail_entity="Acme",
        relation="founder",
    )

    assert ner.tokens[0] == "EU"
    assert qa.answers == ["EU"]
    assert summary.summary == "A short summary."
    assert relation.relation == "founder"
    assert NLPTask.RELATION_EXTRACTION == "relation_extraction"


def test_ner_example_requires_aligned_tokens_and_labels():
    with pytest.raises(ValueError):
        NERExample(tokens=["EU"], labels=["B-ORG", "O"])


def test_bio_tags_to_spans_extracts_entities():
    labels = ["B-PER", "I-PER", "O", "B-ORG", "I-ORG"]

    assert bio_tags_to_spans(labels) == {("PER", 0, 2), ("ORG", 3, 5)}


def test_entity_level_f1_requires_exact_span_match():
    gold = [["B-PER", "I-PER", "O", "B-ORG"]]
    predicted = [["B-PER", "O", "O", "B-ORG"]]

    assert entity_level_f1(gold, predicted) == pytest.approx(0.5)


def test_qa_metrics_use_squad_style_normalization():
    assert qa_exact_match("The Eiffel Tower!", ["eiffel tower"]) == 1.0
    assert qa_token_f1("Eiffel Tower Paris", ["Eiffel Tower"]) == pytest.approx(0.8)


def test_qa_scores_average_over_examples():
    scores = qa_scores(["Paris", "wrong answer"], [["Paris"], ["right answer"]])

    assert scores["exact_match"] == pytest.approx(0.5)
    assert scores["token_f1"] == pytest.approx(0.75)


def test_rouge_l_f1_scores_summarization_overlap():
    assert rouge_l_f1("cat sat", "cat sat on mat") == pytest.approx(2 / 3)
    assert mean_rouge_l_f1(["cat sat", "blue sky"], ["cat sat", "green field"]) == pytest.approx(
        0.5
    )


def test_relation_macro_f1_wraps_classification_metric():
    gold = ["founder", "no_relation", "founder", "no_relation"]
    predicted = ["founder", "no_relation", "no_relation", "no_relation"]

    assert relation_macro_f1(gold, predicted) == pytest.approx(11 / 15)
