from __future__ import annotations

import json
import os
import sys

import pytest

from annotatebench.cost_estimation import (
    estimate_total_cost,
    estimate_total_cost_from_strata,
    stratified_sample_indices,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from make_llm_api_cost_summary import token_counts


def test_stratified_sample_is_reproducible_and_balanced() -> None:
    lengths = list(range(90))
    first = stratified_sample_indices(lengths, sample_size=30, random_state=7)
    second = stratified_sample_indices(lengths, sample_size=30, random_state=7)
    assert first == second
    assert len(first) == 30
    assert [sum(index in range(start, start + 30) for index in first) for start in (0, 30, 60)] == [10, 10, 10]


def test_estimate_total_cost_recovers_constant_cost() -> None:
    lengths = list(range(90))
    sampled = stratified_sample_indices(lengths, sample_size=30)
    estimate = estimate_total_cost(lengths, sampled, [0.01] * 30, n_resamples=100)
    assert estimate.mean == pytest.approx(0.9)
    assert estimate.lower_95 == pytest.approx(0.9)
    assert estimate.upper_95 == pytest.approx(0.9)


def test_estimate_total_cost_rejects_zero_placeholder() -> None:
    with pytest.raises(ValueError, match="positive"):
        estimate_total_cost([1, 2, 3], [0, 1, 2], [0.01, 0.0, 0.01])


def test_estimate_total_cost_from_external_strata() -> None:
    estimate = estimate_total_cost_from_strata(
        [0] * 5 + [1] * 3 + [2] * 2,
        [0, 1, 2],
        [0.01, 0.02, 0.03],
        n_resamples=50,
    )
    assert estimate.mean == pytest.approx(0.17)


def test_token_counts_recovers_usage_from_raw_response() -> None:
    row = {
        "input_tokens": "0",
        "output_tokens": "0",
        "total_tokens": "0",
        "raw_response": json.dumps(
            {"usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}}
        ),
    }
    assert token_counts(row) == (11, 3, 14)
