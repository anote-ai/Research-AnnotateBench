#!/usr/bin/env python3
"""Demo script for annotatebench package."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from annotatebench.data import make_full_dataset
from annotatebench.evaluate import (
    strategy_comparison,
    pareto_frontier,
    budget_recommendation,
)
from annotatebench.core import BUDGET_LEVELS


def main() -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
    except ImportError:
        console = None

    curves = make_full_dataset(seed=42)
    print(f"Generated {len(curves)} learning curves (5 strategies × 5 tasks)\n")

    # Strategy comparison (aggregate across tasks for first task per strategy)
    first_per_strategy = []
    seen = set()
    for c in curves:
        if c.config.strategy not in seen:
            first_per_strategy.append(c)
            seen.add(c.config.strategy)

    comparison = strategy_comparison(first_per_strategy)

    print("Strategy Comparison (CLASSIFICATION task):")
    if console:
        table = Table(title="Strategy Comparison")
        table.add_column("Strategy")
        table.add_column("Best F1", justify="right")
        table.add_column("Total Cost", justify="right")
        table.add_column("AUC", justify="right")
        table.add_column("Efficiency", justify="right")
        for strategy, info in sorted(comparison.items(), key=lambda x: -x[1]["best_f1"]):
            table.add_row(
                strategy,
                f"{info['best_f1']:.4f}",
                f"${info['total_cost']:.2f}",
                f"{info['auc']:.1f}",
                f"{info['efficiency']:.4f}",
            )
        console.print(table)
    else:
        for strategy, info in sorted(comparison.items(), key=lambda x: -x[1]["best_f1"]):
            print(f"  {strategy}: best_f1={info['best_f1']:.4f}, cost=${info['total_cost']:.2f}, AUC={info['auc']:.1f}")

    # Pareto frontier
    print("\nPareto Frontier (cost vs F1):")
    pts = [(info["total_cost"], info["best_f1"]) for info in comparison.values()]
    pf = pareto_frontier(pts)
    for cost, f1 in pf:
        print(f"  cost=${cost:.2f}, F1={f1:.4f}")

    # Budget recommendation
    print("\nBudget recommendation for target F1=0.85:")
    recs = budget_recommendation(first_per_strategy, target_f1=0.85)
    for strategy, cost in recs.items():
        cost_str = f"${cost:.2f}" if cost is not None else "Not achievable"
        print(f"  {strategy}: {cost_str}")


if __name__ == "__main__":
    main()
