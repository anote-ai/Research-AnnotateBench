"""Evaluation utilities for AnnotateBench."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from annotatebench.core import LearningCurve


def pareto_frontier(
    points: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Return the Pareto-optimal (non-dominated) subset of (cost, f1) points.

    A point dominates another if it has lower cost AND higher or equal f1,
    or equal cost AND strictly higher f1.
    """
    if not points:
        return []
    # Sort by cost ascending, then f1 descending
    sorted_pts = sorted(points, key=lambda p: (p[0], -p[1]))
    frontier: list[tuple[float, float]] = []
    max_f1 = float("-inf")
    for cost, f1 in sorted_pts:
        if f1 > max_f1:
            frontier.append((cost, f1))
            max_f1 = f1
    return frontier


def area_under_learning_curve(
    budgets: list[int], f1_scores: list[float]
) -> float:
    """Trapezoidal integration of the learning curve."""
    if len(budgets) != len(f1_scores):
        raise ValueError("budgets and f1_scores must have the same length.")
    if len(budgets) < 2:  # noqa: PLR2004
        return 0.0
    pairs = sorted(zip(budgets, f1_scores))
    area = 0.0
    for i in range(1, len(pairs)):
        x0, y0 = pairs[i - 1]
        x1, y1 = pairs[i]
        area += (x1 - x0) * (y0 + y1) / 2.0
    return area


def cost_to_target_f1(
    curve: "LearningCurve", target_f1: float
) -> float | None:
    """Interpolate the cost (budget) needed to reach target_f1.

    Returns None if the target is not achievable within the observed range.
    """
    points = sorted(curve.points, key=lambda p: p.budget)
    for i in range(1, len(points)):
        p0, p1 = points[i - 1], points[i]
        if p0.f1 <= target_f1 <= p1.f1:
            # Linear interpolation in budget space
            if p1.f1 == p0.f1:
                return float(p0.cost_usd)
            t = (target_f1 - p0.f1) / (p1.f1 - p0.f1)
            return p0.cost_usd + t * (p1.cost_usd - p0.cost_usd)
    # Check if already achieved at first point
    if points and points[0].f1 >= target_f1:
        return points[0].cost_usd
    return None


def strategy_comparison(curves: list["LearningCurve"]) -> dict:
    """Return the best strategy per task at each budget level.

    Returns a nested dict: result[task][budget] = best_strategy_name.
    """
    # Group by (task, budget) -> list of (f1, strategy)
    from collections import defaultdict

    task_budget_scores: dict = defaultdict(lambda: defaultdict(list))
    for curve in curves:
        task = curve.config.task.value
        strategy = curve.config.strategy.value
        for pt in curve.points:
            task_budget_scores[task][pt.budget].append((pt.f1, strategy))

    result: dict = {}
    for task, budget_map in task_budget_scores.items():
        result[task] = {}
        for budget, scores in budget_map.items():
            best_strategy = max(scores, key=lambda x: x[0])[1]
            result[task][budget] = best_strategy
    return result
