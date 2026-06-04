from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .core import LearningCurve


def pareto_frontier(
    points: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """Return Pareto-optimal (cost, f1) points: minimize cost, maximize f1.

    A point dominates another if it has <= cost AND >= f1 (strictly better in at least one).
    """
    if not points:
        return []

    # Sort by cost ascending, then f1 descending
    sorted_pts = sorted(points, key=lambda p: (p[0], -p[1]))

    pareto: List[Tuple[float, float]] = []
    max_f1_so_far = -1.0

    for cost, f1 in sorted_pts:
        if f1 > max_f1_so_far:
            pareto.append((cost, f1))
            max_f1_so_far = f1

    return pareto


def area_under_learning_curve(
    budgets: List[int], f1_scores: List[float]
) -> float:
    """Trapezoidal area under the learning curve."""
    if len(budgets) < 2:
        return 0.0
    area = 0.0
    for i in range(1, len(budgets)):
        dx = budgets[i] - budgets[i - 1]
        avg_y = (f1_scores[i] + f1_scores[i - 1]) / 2.0
        area += dx * avg_y
    return area


def cost_to_target_f1(
    curve: LearningCurve, target_f1: float
) -> Optional[float]:
    """Return the interpolated cost at which F1 first reaches target_f1.

    Uses linear interpolation between consecutive curve points.
    Returns None if target is never reached.
    """
    pts = sorted(curve.points, key=lambda p: p.budget)

    # Check if any point meets the target
    for i, pt in enumerate(pts):
        if pt.f1 >= target_f1:
            if i == 0:
                return pt.cost_usd
            prev = pts[i - 1]
            if prev.f1 >= target_f1:
                return prev.cost_usd
            # Linear interpolation on f1 vs cost
            t = (target_f1 - prev.f1) / max(pt.f1 - prev.f1, 1e-12)
            return prev.cost_usd + t * (pt.cost_usd - prev.cost_usd)

    return None


def strategy_comparison(
    curves: List[LearningCurve],
) -> Dict[str, Dict]:
    """Compare strategies: {strategy_name: {best_f1, total_cost, auc, efficiency}}."""
    from .core import BUDGET_LEVELS, area_under_learning_curve

    result: Dict[str, Dict] = {}
    for curve in curves:
        strategy_name = curve.config.strategy.value
        best_f1 = curve.max_f1()
        total_cost = curve.total_cost()
        budgets = [pt.budget for pt in sorted(curve.points, key=lambda p: p.budget)]
        f1s = [pt.f1 for pt in sorted(curve.points, key=lambda p: p.budget)]
        auc = area_under_learning_curve(budgets, f1s)
        efficiency = best_f1 / max(total_cost, 1e-9)
        result[strategy_name] = {
            "best_f1": best_f1,
            "total_cost": total_cost,
            "auc": auc,
            "efficiency": efficiency,
        }
    return result


def budget_recommendation(
    curves: List[LearningCurve], target_f1: float = 0.8
) -> Dict[str, Optional[float]]:
    """Return {strategy: min_cost_to_reach_target or None}."""
    result: Dict[str, Optional[float]] = {}
    for curve in curves:
        strategy_name = curve.config.strategy.value
        cost = cost_to_target_f1(curve, target_f1)
        result[strategy_name] = cost
    return result
