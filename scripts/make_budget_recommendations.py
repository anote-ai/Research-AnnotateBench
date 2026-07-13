#!/usr/bin/env python3
"""Create budget recommendation tables from benchmark CSV results."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", default=["results/benchmark_results.csv"])
    parser.add_argument("--output-csv", default="results/budget_recommendations.csv")
    parser.add_argument("--targets", default="0.50,0.60,0.70,0.75,0.80")
    parser.add_argument("--cost-scenario", default="base")
    return parser.parse_args()


def make_budget_recommendations(
    result_paths: list[str | Path],
    output_csv: str | Path,
    targets: list[float],
    cost_scenario: str = "base",
) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in result_paths]
    df = pd.concat(frames, ignore_index=True)
    if "cost_scenario" in df.columns:
        df = df[df["cost_scenario"] == cost_scenario].copy()
    if df.empty:
        raise ValueError("No result rows available for budget recommendations.")

    summary = (
        df.groupby(["dataset", "strategy", "budget"], as_index=False)
        .agg(macro_f1=("macro_f1", "mean"), total_cost=("total_cost", "mean"))
    )
    rows: list[dict[str, object]] = []
    for dataset, dataset_df in summary.groupby("dataset"):
        for target in targets:
            eligible = dataset_df[dataset_df["macro_f1"] >= target].sort_values(
                ["total_cost", "macro_f1"],
                ascending=[True, False],
            )
            if eligible.empty:
                rows.append(
                    {
                        "dataset": dataset,
                        "target_f1": target,
                        "cost_scenario": cost_scenario,
                        "cheapest_strategy": "",
                        "required_budget": "",
                        "estimated_cost": "",
                        "mean_macro_f1": "",
                    }
                )
                continue
            best = eligible.iloc[0]
            rows.append(
                {
                    "dataset": dataset,
                    "target_f1": target,
                    "cost_scenario": cost_scenario,
                    "cheapest_strategy": best["strategy"],
                    "required_budget": int(best["budget"]),
                    "estimated_cost": f"{best['total_cost']:.2f}",
                    "mean_macro_f1": f"{best['macro_f1']:.4f}",
                }
            )

    result = pd.DataFrame(rows)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    args = parse_args()
    targets = [float(value) for value in args.targets.split(",") if value.strip()]
    result = make_budget_recommendations(
        args.results,
        args.output_csv,
        targets,
        args.cost_scenario,
    )
    print(f"Wrote {len(result)} recommendations to {args.output_csv}")


if __name__ == "__main__":
    main()
