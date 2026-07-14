from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from annotatebench.metrics import (
    bonferroni_significant,
    bootstrap_confidence_interval,
    group_metric_variance,
    gwet_ac1,
    paired_bootstrap_p_value,
)


def test_bootstrap_confidence_interval_reports_estimate_and_bounds():
    interval = bootstrap_confidence_interval(
        [0.2, 0.4, 0.6, 0.8],
        n_resamples=200,
        random_state=7,
    )

    assert interval.estimate == pytest.approx(0.5)
    assert interval.lower <= interval.estimate <= interval.upper
    assert interval.confidence_level == 0.95
    assert interval.n_resamples == 200


def test_clustered_bootstrap_resamples_annotator_clusters():
    interval = bootstrap_confidence_interval(
        [0.0, 0.0, 1.0, 1.0],
        clusters=["ann-a", "ann-a", "ann-b", "ann-b"],
        n_resamples=200,
        random_state=11,
    )

    assert interval.estimate == pytest.approx(0.5)
    assert interval.lower == pytest.approx(0.0)
    assert interval.upper == pytest.approx(1.0)


def test_gwet_ac1_handles_skewed_label_distribution():
    score = gwet_ac1(
        ["yes", "yes", "yes", "yes"],
        ["yes", "yes", "yes", "yes"],
        labels=["yes", "no"],
    )

    assert score == pytest.approx(1.0)


def test_gwet_ac1_penalizes_disagreement():
    score = gwet_ac1(
        ["yes", "yes", "no", "no"],
        ["yes", "no", "yes", "no"],
        labels=["yes", "no"],
    )

    assert score == pytest.approx(0.0)


def test_paired_bootstrap_p_value_detects_consistent_improvement():
    p_value = paired_bootstrap_p_value(
        baseline=[0.1, 0.2, 0.3, 0.4],
        candidate=[0.3, 0.4, 0.5, 0.6],
        n_resamples=200,
        random_state=3,
    )

    assert p_value < 0.05


def test_bonferroni_significant_applies_corrected_threshold():
    assert bonferroni_significant([0.001, 0.02, 0.2], alpha=0.05) == [True, False, False]


def test_group_metric_variance_summarizes_annotator_pool_variance():
    rows = [
        {"pool": "us", "accuracy": 0.8},
        {"pool": "us", "accuracy": 1.0},
        {"pool": "non_us", "accuracy": 0.6},
        {"pool": "non_us", "accuracy": 0.6},
    ]

    variances = group_metric_variance(rows, group_key="pool", metric_key="accuracy")

    assert variances["us"] == pytest.approx(0.02)
    assert variances["non_us"] == pytest.approx(0.0)
