#!/usr/bin/env python3
"""Report whether the top two configuration intervals overlap per dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from make_paper_summary_table import summarize_configurations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/benchmark_results.csv")
    parser.add_argument("--cost-scenario", default="base")
    parser.add_argument("--output-csv", default="results/top_two_interval_overlap.csv")
    return parser.parse_args()


def make_overlap_table(results: pd.DataFrame, cost_scenario: str = "base") -> pd.DataFrame:
    summary = summarize_configurations(results, cost_scenario)
    rows: list[dict[str, object]] = []
    for dataset, frame in summary.groupby("dataset"):
        top = frame.head(2).reset_index(drop=True)
        if len(top) != 2:
            raise ValueError(f"Dataset {dataset} has fewer than two configurations.")
        overlap = max(top.loc[0, "ci_lower_95"], top.loc[1, "ci_lower_95"]) <= min(
            top.loc[0, "ci_upper_95"], top.loc[1, "ci_upper_95"]
        )
        rows.append(
            {
                "dataset": dataset,
                "top_strategy": top.loc[0, "strategy"],
                "top_budget": int(top.loc[0, "budget"]),
                "top_mean_macro_f1": top.loc[0, "mean_macro_f1"],
                "top_ci_lower_95": top.loc[0, "ci_lower_95"],
                "top_ci_upper_95": top.loc[0, "ci_upper_95"],
                "runner_strategy": top.loc[1, "strategy"],
                "runner_budget": int(top.loc[1, "budget"]),
                "runner_mean_macro_f1": top.loc[1, "mean_macro_f1"],
                "runner_ci_lower_95": top.loc[1, "ci_lower_95"],
                "runner_ci_upper_95": top.loc[1, "ci_upper_95"],
                "intervals_overlap": bool(overlap),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    output = make_overlap_table(pd.read_csv(args.results), args.cost_scenario)
    path = Path(args.output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    print(f"Wrote {len(output)} rows to {path}; overlaps={int(output['intervals_overlap'].sum())}")


if __name__ == "__main__":
    main()
