from __future__ import annotations

from typing import Sequence


def expected_calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[bool | int],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error for annotation predictions.

    Args:
        confidences: Model confidence for each annotation, in [0, 1].
        correctness: Whether each annotation matches the gold label.
        n_bins: Number of equal-width confidence bins.
    """
    if len(confidences) != len(correctness):
        raise ValueError("confidences and correctness must have the same length.")
    if not confidences:
        raise ValueError("At least one prediction is required.")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    bins: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for confidence, is_correct in zip(confidences, correctness):
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("All confidences must be in [0, 1].")
        bin_index = min(int(confidence * n_bins), n_bins - 1)
        bins[bin_index].append((confidence, float(bool(is_correct))))

    total = len(confidences)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        mean_confidence = sum(item[0] for item in bucket) / len(bucket)
        mean_accuracy = sum(item[1] for item in bucket) / len(bucket)
        ece += (len(bucket) / total) * abs(mean_accuracy - mean_confidence)
    return ece


def calibration_score(
    confidences: Sequence[float],
    correctness: Sequence[bool | int],
    n_bins: int = 10,
) -> float:
    """Return 1 - ECE, clamped to [0, 1]."""
    ece = expected_calibration_error(confidences, correctness, n_bins=n_bins)
    return max(0.0, min(1.0, 1.0 - ece))


def lari_score(
    f1: float,
    confidences: Sequence[float],
    correctness: Sequence[bool | int],
    n_bins: int = 10,
) -> float:
    """Compute LARI = F1 * (1 - ECE)."""
    if not 0.0 <= f1 <= 1.0:
        raise ValueError("f1 must be in [0, 1].")
    return f1 * calibration_score(confidences, correctness, n_bins=n_bins)
