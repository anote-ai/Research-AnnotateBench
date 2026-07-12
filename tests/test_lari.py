from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from annotatebench.metrics import calibration_score, expected_calibration_error, lari_score


def test_expected_calibration_error_zero_for_perfectly_calibrated_bins():
    ece = expected_calibration_error(
        confidences=[0.5, 0.5, 1.0, 1.0],
        correctness=[True, False, True, True],
        n_bins=2,
    )

    assert ece == 0.0


def test_expected_calibration_error_penalizes_overconfidence():
    ece = expected_calibration_error(
        confidences=[0.9, 0.9, 0.9, 0.9],
        correctness=[True, False, False, False],
        n_bins=10,
    )

    assert ece == pytest.approx(0.65)


def test_lari_score_multiplies_f1_by_calibration_score():
    lari = lari_score(
        f1=0.8,
        confidences=[0.9, 0.9, 0.9, 0.9],
        correctness=[True, False, False, False],
        n_bins=10,
    )

    assert lari == pytest.approx(0.8 * 0.35)
    assert calibration_score([0.9], [True]) == pytest.approx(0.9)


def test_lari_validates_inputs():
    with pytest.raises(ValueError):
        expected_calibration_error([1.2], [True])
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [True, False])
    with pytest.raises(ValueError):
        lari_score(1.2, [0.5], [True])
