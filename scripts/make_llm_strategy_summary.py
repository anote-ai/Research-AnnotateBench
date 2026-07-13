#!/usr/bin/env python3
"""Summarize LLM annotator rows against the best gold-label strategy."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-results", default="results/benchmark_results.csv")
    parser.add_argument("--llm-results", nargs="+", required=True)
    parser.add_argument("--cost-scenario", default="base")
    parser.add_argument("--seed", type=int, default=None, help="Deprecated alias for a single seed.")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--output-csv", default="results/llm_strategy_seed0_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [args.seed] if args.seed is not None else [
        int(value) for value in args.seeds.split(",") if value.strip()
    ]
    llm = pd.concat([pd.read_csv(path) for path in args.llm_results], ignore_index=True)
    llm = llm[(llm["cost_scenario"] == args.cost_scenario) & (llm["seed"].isin(seeds))].copy()
    if llm.empty:
        raise SystemExit("No LLM rows matched the requested cost scenario and seeds.")

    gold = pd.read_csv(args.gold_results)
    gold = gold[
        (gold["cost_scenario"] == args.cost_scenario)
        & (gold["seed"].isin(seeds))
        & (gold["dataset"].isin(llm["dataset"].unique()))
        & (gold["budget"].isin(llm["budget"].unique()))
    ].copy()
    if gold.empty:
        raise SystemExit("No gold-label rows matched the LLM datasets, budgets, scenario, and seed.")

    best_indices = gold.groupby(["dataset", "budget", "seed"])["macro_f1"].idxmax()
    best_gold = gold.loc[
        best_indices,
        ["dataset", "budget", "seed", "strategy", "macro_f1", "accuracy"],
    ].rename(
        columns={
            "strategy": "best_gold_strategy",
            "macro_f1": "best_gold_macro_f1",
            "accuracy": "best_gold_accuracy",
        }
    )
    summary = llm.merge(best_gold, on=["dataset", "budget", "seed"], how="left")
    summary["macro_f1_gap_vs_best_gold"] = summary["macro_f1"] - summary["best_gold_macro_f1"]
    columns = [
        "dataset",
        "budget",
        "seed",
        "strategy",
        "macro_f1",
        "best_gold_macro_f1",
        "macro_f1_gap_vs_best_gold",
        "best_gold_strategy",
        "accuracy",
        "best_gold_accuracy",
        "llm_label_accuracy",
        "llm_label_macro_f1",
        "llm_ece",
        "llm_lari",
        "model_name",
        "prompt_version",
    ]
    summary = summary[columns].sort_values(["dataset", "budget"]).reset_index(drop=True)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"Wrote {len(summary)} rows to {output_path}")


if __name__ == "__main__":
    main()
