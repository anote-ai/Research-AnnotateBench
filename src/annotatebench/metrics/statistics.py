from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, Mapping, Sequence

import numpy as np


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values))


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    n_resamples: int


def bootstrap_confidence_interval(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = _mean,
    *,
    clusters: Sequence[Hashable] | None = None,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = None,
) -> BootstrapInterval:
    """Estimate a percentile bootstrap confidence interval.

    When clusters are provided, resampling happens at the cluster level. This is
    the annotator-level bootstrap needed when multiple labels come from the
    same annotator.
    """
    _validate_bootstrap_inputs(values, clusters, n_resamples, confidence_level)

    rng = np.random.default_rng(random_state)
    sample = np.asarray(values, dtype=float)
    estimate = float(statistic(sample.tolist()))
    draws = np.empty(n_resamples, dtype=float)

    if clusters is None:
        for draw_index in range(n_resamples):
            indices = rng.integers(0, len(sample), size=len(sample))
            draws[draw_index] = float(statistic(sample[indices].tolist()))
    else:
        cluster_ids = list(dict.fromkeys(clusters))
        cluster_to_indices = _cluster_indices(clusters)
        for draw_index in range(n_resamples):
            chosen = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
            indices = [idx for cluster in chosen for idx in cluster_to_indices[cluster]]
            draws[draw_index] = float(statistic(sample[indices].tolist()))

    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapInterval(
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        confidence_level=confidence_level,
        n_resamples=n_resamples,
    )


def paired_bootstrap_p_value(
    baseline: Sequence[float],
    candidate: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = _mean,
    *,
    clusters: Sequence[Hashable] | None = None,
    n_resamples: int = 1000,
    random_state: int | None = None,
) -> float:
    """Return a two-sided paired bootstrap p-value for candidate - baseline."""
    if len(baseline) != len(candidate):
        raise ValueError("baseline and candidate must have the same length.")
    _validate_bootstrap_inputs(baseline, clusters, n_resamples, 0.95)

    rng = np.random.default_rng(random_state)
    base = np.asarray(baseline, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    diffs = np.empty(n_resamples, dtype=float)

    if clusters is None:
        for draw_index in range(n_resamples):
            indices = rng.integers(0, len(base), size=len(base))
            diffs[draw_index] = _statistic_delta(base, cand, indices, statistic)
    else:
        cluster_ids = list(dict.fromkeys(clusters))
        cluster_to_indices = _cluster_indices(clusters)
        for draw_index in range(n_resamples):
            chosen = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
            indices = [idx for cluster in chosen for idx in cluster_to_indices[cluster]]
            diffs[draw_index] = _statistic_delta(base, cand, indices, statistic)

    p_lower = (np.count_nonzero(diffs <= 0.0) + 1) / (n_resamples + 1)
    p_upper = (np.count_nonzero(diffs >= 0.0) + 1) / (n_resamples + 1)
    return float(min(1.0, 2.0 * min(p_lower, p_upper)))


def bonferroni_significant(
    p_values: Sequence[float],
    *,
    alpha: float = 0.05,
) -> list[bool]:
    if len(p_values) == 0:
        raise ValueError("At least one p-value is required.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    for p_value in p_values:
        if not 0.0 <= p_value <= 1.0:
            raise ValueError("p-values must be in [0, 1].")

    threshold = alpha / len(p_values)
    return [p_value < threshold for p_value in p_values]


def gwet_ac1(
    rater_a: Sequence[Hashable],
    rater_b: Sequence[Hashable],
    *,
    labels: Sequence[Hashable] | None = None,
) -> float:
    """Compute Gwet's AC1 agreement coefficient for two nominal raters."""
    if len(rater_a) != len(rater_b):
        raise ValueError("rater_a and rater_b must have the same length.")
    if len(rater_a) == 0:
        raise ValueError("At least one paired rating is required.")

    observed = sum(a == b for a, b in zip(rater_a, rater_b)) / len(rater_a)
    label_values = list(labels) if labels is not None else list(dict.fromkeys([*rater_a, *rater_b]))
    if len(label_values) < 2:
        return 1.0 if observed == 1.0 else 0.0

    total_assignments = 2 * len(rater_a)
    probabilities = []
    for label in label_values:
        count = sum(value == label for value in rater_a) + sum(value == label for value in rater_b)
        probabilities.append(count / total_assignments)

    chance = sum(prob * (1.0 - prob) for prob in probabilities) / (len(label_values) - 1)
    if chance >= 1.0:
        return 0.0
    return float((observed - chance) / (1.0 - chance))


def group_metric_variance(
    rows: Sequence[Mapping[str, object]],
    *,
    group_key: str,
    metric_key: str,
) -> dict[Hashable, float]:
    groups: dict[Hashable, list[float]] = {}
    for row in rows:
        groups.setdefault(row[group_key], []).append(float(row[metric_key]))

    return {
        group: float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
        for group, values in groups.items()
    }


def _cluster_indices(clusters: Sequence[Hashable]) -> dict[Hashable, list[int]]:
    cluster_to_indices: dict[Hashable, list[int]] = {}
    for index, cluster in enumerate(clusters):
        cluster_to_indices.setdefault(cluster, []).append(index)
    return cluster_to_indices


def _statistic_delta(
    baseline: np.ndarray,
    candidate: np.ndarray,
    indices: Sequence[int],
    statistic: Callable[[Sequence[float]], float],
) -> float:
    return float(statistic(candidate[list(indices)].tolist()) - statistic(baseline[list(indices)].tolist()))


def _validate_bootstrap_inputs(
    values: Sequence[float],
    clusters: Sequence[Hashable] | None,
    n_resamples: int,
    confidence_level: float,
) -> None:
    if len(values) == 0:
        raise ValueError("At least one value is required.")
    if clusters is not None and len(clusters) != len(values):
        raise ValueError("clusters must have the same length as values.")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1).")
