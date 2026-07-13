#!/usr/bin/env python3
"""Create compact benchmark summary tables from row-level benchmark results."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/benchmark_results.csv")
    parser.add_argument("--cost-scenario", default="base")
    parser.add_argument("--best-overall-csv", default="results/benchmark_best_overall.csv")
    parser.add_argument("--best-by-strategy-csv", default="results/benchmark_best_by_strategy.csv")
    return parser.parse_args()


def summarize_benchmark(
    results_csv: str | Path,
    best_overall_csv: str | Path,
    best_by_strategy_csv: str | Path,
    cost_scenario: str = "base",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(results_csv)
    required = {"dataset", "strategy", "budget", "seed", "macro_f1", "accuracy", "total_cost"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")
    if "cost_scenario" in df.columns:
        df = df[df["cost_scenario"] == cost_scenario].copy()
    if df.empty:
        raise ValueError(f"No rows found for cost scenario: {cost_scenario}")

    summary = (
        df.groupby(["dataset", "strategy", "budget"], as_index=False)
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            mean_accuracy=("accuracy", "mean"),
            mean_total_cost=("total_cost", "mean"),
        )
        .sort_values(
            ["dataset", "mean_macro_f1", "mean_total_cost"],
            ascending=[True, False, True],
        )
    )
    summary["std_macro_f1"] = summary["std_macro_f1"].fillna(0.0)
    summary["mean_total_cost"] = summary["mean_total_cost"].round(2)

    best_by_strategy = (
        summary.sort_values(
            ["dataset", "strategy", "mean_macro_f1", "mean_total_cost"],
            ascending=[True, True, False, True],
        )
        .groupby(["dataset", "strategy"], as_index=False)
        .head(1)
        .sort_values(
            ["dataset", "mean_macro_f1", "mean_total_cost"],
            ascending=[True, False, True],
        )
    )

    best_overall = (
        summary.sort_values(
            ["dataset", "mean_macro_f1", "mean_total_cost"],
            ascending=[True, False, True],
        )
        .groupby("dataset", as_index=False)
        .head(1)
    )

    best_by_strategy_path = Path(best_by_strategy_csv)
    best_overall_path = Path(best_overall_csv)
    best_by_strategy_path.parent.mkdir(parents=True, exist_ok=True)
    best_overall_path.parent.mkdir(parents=True, exist_ok=True)
    best_by_strategy.to_csv(best_by_strategy_path, index=False)
    best_overall.to_csv(best_overall_path, index=False)
    return best_overall, best_by_strategy


def main() -> None:
    args = parse_args()
    best_overall, best_by_strategy = summarize_benchmark(
        args.results,
        args.best_overall_csv,
        args.best_by_strategy_csv,
        args.cost_scenario,
    )
    print(f"Wrote {len(best_overall)} rows to {args.best_overall_csv}")
    print(f"Wrote {len(best_by_strategy)} rows to {args.best_by_strategy_csv}")


if __name__ == "__main__":
    main()
