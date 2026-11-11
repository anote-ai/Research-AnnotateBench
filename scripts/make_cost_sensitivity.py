#!/usr/bin/env python3
"""Audit Pareto-frontier stability under scaled selection overheads."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/benchmark_results.csv")
    parser.add_argument("--cost-scenario", default="base")
    parser.add_argument("--factors", default="1,2,5,10")
    parser.add_argument("--output-csv", default="results/selection_overhead_sensitivity.csv")
    return parser.parse_args()


def pareto_members(frame: pd.DataFrame) -> set[tuple[str, int]]:
    members: set[tuple[str, int]] = set()
    for row in frame.itertuples():
        dominated = (
            (frame["adjusted_total_cost"] <= row.adjusted_total_cost)
            & (frame["macro_f1"] >= row.macro_f1)
            & (
                (frame["adjusted_total_cost"] < row.adjusted_total_cost)
                | (frame["macro_f1"] > row.macro_f1)
            )
        ).any()
        if not dominated:
            members.add((str(row.strategy), int(row.budget)))
    return members


def make_cost_sensitivity(
    results: pd.DataFrame,
    *,
    factors: list[float],
    cost_scenario: str = "base",
) -> pd.DataFrame:
    required = {
        "dataset",
        "strategy",
        "budget",
        "macro_f1",
        "annotation_cost",
        "selection_cost",
        "cost_scenario",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")
    source = results[results["cost_scenario"] == cost_scenario].copy()
    if source.empty:
        raise ValueError(f"No rows for cost scenario: {cost_scenario}")

    output: list[pd.DataFrame] = []
    baseline: dict[str, set[tuple[str, int]]] = {}
    for factor in factors:
        adjusted = source.copy()
        adjusted["adjusted_total_cost"] = (
            adjusted["annotation_cost"] + factor * adjusted["selection_cost"]
        )
        summary = (
            adjusted.groupby(["dataset", "strategy", "budget"], as_index=False)
            .agg(
                macro_f1=("macro_f1", "mean"),
                adjusted_total_cost=("adjusted_total_cost", "mean"),
            )
        )
        summary["overhead_factor"] = factor
        summary["is_pareto"] = False
        summary["frontier_stable_vs_1x"] = False
        for dataset, indices in summary.groupby("dataset").groups.items():
            dataset_frame = summary.loc[indices]
            members = pareto_members(dataset_frame)
            if factor == 1:
                baseline[str(dataset)] = members
            summary.loc[indices, "is_pareto"] = [
                (str(row.strategy), int(row.budget)) in members
                for row in dataset_frame.itertuples()
            ]
            summary.loc[indices, "frontier_stable_vs_1x"] = (
                members == baseline.get(str(dataset), members)
            )
        output.append(summary)
    return pd.concat(output, ignore_index=True)


def main() -> None:
    args = parse_args()
    factors = [float(value) for value in args.factors.split(",") if value.strip()]
    output = make_cost_sensitivity(
        pd.read_csv(args.results),
        factors=factors,
        cost_scenario=args.cost_scenario,
    )
    path = Path(args.output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    print(f"Wrote {len(output)} rows to {path}")


if __name__ == "__main__":
    main()
