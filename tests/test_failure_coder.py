from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from annotatebench.taxonomy import FailureCategory, FailureCode, summarize_failure_codes


def test_summarize_failure_codes_counts_all_taxonomy_categories():
    codes = [
        FailureCode("ex-1", FailureCategory.LABEL_AMBIGUITY, confidence=0.7),
        FailureCode("ex-2", FailureCategory.LABEL_AMBIGUITY, confidence=0.8),
        FailureCode("ex-3", FailureCategory.SCHEMA_CONFUSION, confidence=0.9),
    ]

    summary = summarize_failure_codes(codes)

    assert summary[FailureCategory.LABEL_AMBIGUITY] == 2
    assert summary[FailureCategory.SCHEMA_CONFUSION] == 1
    assert summary[FailureCategory.KNOWLEDGE_GAP] == 0


def test_failure_code_validates_confidence():
    with pytest.raises(ValueError):
        FailureCode("ex-1", FailureCategory.OVERCONFIDENT_HALLUCINATION, confidence=1.1)
