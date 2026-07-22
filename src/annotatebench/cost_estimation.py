"""Reproducible stratified estimates for LLM annotation costs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostEstimate:
    mean: float
    lower_95: float
    upper_95: float
    sample_size: int


def length_strata(lengths: list[int]) -> np.ndarray:
    """Assign deterministic short/medium/long strata by stable rank."""
    if not lengths:
        raise ValueError("At least one length is required.")
    order = np.argsort(np.asarray(lengths), kind="stable")
    strata = np.empty(len(lengths), dtype=int)
    for stratum, indices in enumerate(np.array_split(order, 3)):
        strata[indices] = stratum
    return strata


def stratified_sample_indices(
    lengths: list[int],
    *,
    sample_size: int = 30,
    random_state: int = 0,
) -> list[int]:
    """Sample as evenly as possible from length tertiles."""
    if sample_size <= 0 or sample_size > len(lengths):
        raise ValueError("sample_size must be between 1 and the population size.")
    strata = length_strata(lengths)
    rng = np.random.default_rng(random_state)
    base, remainder = divmod(sample_size, 3)
    selected: list[int] = []
    for stratum in range(3):
        candidates = np.flatnonzero(strata == stratum)
        take = min(len(candidates), base + (1 if stratum < remainder else 0))
        selected.extend(int(value) for value in rng.choice(candidates, size=take, replace=False))
    if len(selected) < sample_size:
        remaining = sorted(set(range(len(lengths))) - set(selected))
        selected.extend(int(value) for value in rng.choice(remaining, size=sample_size - len(selected), replace=False))
    return sorted(selected)


def estimate_total_cost(
    population_lengths: list[int],
    sampled_indices: list[int],
    sampled_costs: list[float],
    *,
    n_resamples: int = 2000,
    random_state: int = 0,
) -> CostEstimate:
    """Estimate a population total with stratified bootstrap intervals."""
    if len(sampled_indices) != len(sampled_costs) or not sampled_indices:
        raise ValueError("Sample indices and costs must have the same non-zero length.")
    if any(cost <= 0 for cost in sampled_costs):
        raise ValueError("Observed API costs must be positive; zero placeholders are invalid.")
    strata = length_strata(population_lengths)
    sample_strata = strata[np.asarray(sampled_indices)]
    costs = np.asarray(sampled_costs, dtype=float)
    population_counts = np.bincount(strata, minlength=3)

    def total(values: np.ndarray) -> float:
        return float(sum(population_counts[s] * values[sample_strata == s].mean() for s in range(3)))

    point = total(costs)
    rng = np.random.default_rng(random_state)
    draws = []
    for _ in range(n_resamples):
        resampled = costs.copy()
        for stratum in range(3):
            positions = np.flatnonzero(sample_strata == stratum)
            resampled[positions] = rng.choice(costs[positions], size=len(positions), replace=True)
        draws.append(total(resampled))
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return CostEstimate(point, float(lower), float(upper), len(sampled_indices))


def estimate_total_cost_from_strata(
    population_strata: list[int],
    sampled_strata: list[int],
    sampled_costs: list[float],
    *,
    n_resamples: int = 2000,
    random_state: int = 0,
) -> CostEstimate:
    """Estimate a total when strata were defined on a larger sampling frame."""
    if len(sampled_strata) != len(sampled_costs) or not sampled_costs:
        raise ValueError("Sample strata and costs must have the same non-zero length.")
    if any(cost <= 0 for cost in sampled_costs):
        raise ValueError("Observed API costs must be positive; zero placeholders are invalid.")
    population_counts = np.bincount(np.asarray(population_strata), minlength=3)
    sample_strata_array = np.asarray(sampled_strata)
    costs = np.asarray(sampled_costs, dtype=float)
    if any(not np.any(sample_strata_array == stratum) for stratum in np.flatnonzero(population_counts)):
        raise ValueError("Every populated stratum requires at least one sampled cost.")

    def total(values: np.ndarray) -> float:
        return float(
            sum(
                population_counts[stratum] * values[sample_strata_array == stratum].mean()
                for stratum in range(3)
                if population_counts[stratum]
            )
        )

    point = total(costs)
    rng = np.random.default_rng(random_state)
    draws = []
    for _ in range(n_resamples):
        resampled = costs.copy()
        for stratum in range(3):
            positions = np.flatnonzero(sample_strata_array == stratum)
            if len(positions):
                resampled[positions] = rng.choice(costs[positions], size=len(positions), replace=True)
        draws.append(total(resampled))
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return CostEstimate(point, float(lower), float(upper), len(sampled_costs))
