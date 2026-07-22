#!/usr/bin/env python3
"""Create the paper-ready best strategy table from benchmark results."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/benchmark_results.csv")
    parser.add_argument("--output-csv", default="results/paper_core_summary.csv")
    parser.add_argument("--cost-scenario", default="base")
    return parser.parse_args()


def make_paper_summary_table(
    results_csv: str | Path,
    output_csv: str | Path,
    cost_scenario: str = "base",
) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    required = {
        "dataset",
        "strategy",
        "budget",
        "seed",
        "macro_f1",
        "accuracy",
        "total_cost",
    }
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
            n_seeds=("seed", "nunique"),
        )
        .sort_values(
            ["dataset", "mean_macro_f1", "mean_total_cost"],
            ascending=[True, False, True],
        )
    )
    summary["std_macro_f1"] = summary["std_macro_f1"].fillna(0.0)
    summary["ci_half_width_95"] = 1.96 * summary["std_macro_f1"] / summary["n_seeds"].pow(0.5)
    best = summary.groupby("dataset", as_index=False).head(1).copy()
    best = best.rename(
        columns={
            "strategy": "best_strategy",
            "budget": "best_budget",
        }
    )
    best["cost_scenario"] = cost_scenario
    best["mean_macro_f1"] = best["mean_macro_f1"].round(4)
    best["std_macro_f1"] = best["std_macro_f1"].round(4)
    best["ci_lower_95"] = (best["mean_macro_f1"] - best["ci_half_width_95"]).clip(lower=0).round(4)
    best["ci_upper_95"] = (best["mean_macro_f1"] + best["ci_half_width_95"]).clip(upper=1).round(4)
    best["mean_accuracy"] = best["mean_accuracy"].round(4)
    best["mean_total_cost"] = best["mean_total_cost"].round(2)
    best = best[
        [
            "dataset",
            "best_strategy",
            "best_budget",
            "mean_macro_f1",
            "std_macro_f1",
            "ci_lower_95",
            "ci_upper_95",
            "mean_accuracy",
            "mean_total_cost",
            "n_seeds",
            "cost_scenario",
        ]
    ].reset_index(drop=True)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(output_path, index=False)
    return best


def main() -> None:
    args = parse_args()
    summary = make_paper_summary_table(args.results, args.output_csv, args.cost_scenario)
    print(f"Wrote {len(summary)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
