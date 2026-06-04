from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .core import CostModel, LearningCurve


def pareto_frontier(
    points: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """Return Pareto-optimal (cost, f1) points: minimize cost, maximize f1."""
    if not points:
        return []

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
    """Return the interpolated cost at which F1 first reaches target_f1."""
    pts = sorted(curve.points, key=lambda p: p.budget)

    for i, pt in enumerate(pts):
        if pt.f1 >= target_f1:
            if i == 0:
                return pt.cost_usd
            prev = pts[i - 1]
            if prev.f1 >= target_f1:
                return prev.cost_usd
            t = (target_f1 - prev.f1) / max(pt.f1 - prev.f1, 1e-12)
            return prev.cost_usd + t * (pt.cost_usd - prev.cost_usd)

    return None


def strategy_comparison(
    curves: List[LearningCurve],
) -> Dict[str, Dict]:
    """Compare strategies: {strategy_name: {best_f1, total_cost, auc, efficiency}}."""
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


def cost_efficiency_curve(
    curve: LearningCurve,
) -> List[Tuple[float, float]]:
    """Return a list of (cumulative_cost_usd, f1) pairs sorted by cost.

    Useful for plotting how F1 improves as dollars are spent.
    """
    pts = sorted(curve.points, key=lambda p: p.budget)
    result: List[Tuple[float, float]] = []
    cumulative_cost = 0.0
    for pt in pts:
        cumulative_cost += pt.cost_usd
        result.append((cumulative_cost, pt.f1))
    return result


def annotation_roi(
    curve: LearningCurve,
    cost_model: CostModel,
    baseline_f1: float = 0.0,
) -> Dict[str, float]:
    """Compute return-on-investment metrics for an annotation strategy.

    Args:
        curve: The learning curve for the strategy.
        cost_model: A CostModel describing annotation costs.
        baseline_f1: F1 of a baseline model with zero annotations.

    Returns:
        Dict with keys:
            f1_gain: max_f1 - baseline_f1
            total_cost_usd: total annotation cost from the CostModel
            f1_per_dollar: f1_gain / total_cost_usd
            cost_per_f1_point: total_cost_usd / f1_gain (INF if no gain)
    """
    total_budget = sum(pt.budget for pt in curve.points)
    total_cost = cost_model.total_api_cost(total_budget)
    max_f1 = curve.max_f1()
    f1_gain = max(max_f1 - baseline_f1, 0.0)
    f1_per_dollar = f1_gain / max(total_cost, 1e-9)
    cost_per_f1_point = total_cost / max(f1_gain, 1e-9)
    return {
        "f1_gain": f1_gain,
        "total_cost_usd": total_cost,
        "f1_per_dollar": f1_per_dollar,
        "cost_per_f1_point": cost_per_f1_point,
    }
