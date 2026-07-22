#!/usr/bin/env python3
"""Validate benchmark result coverage and flag suspicious result patterns."""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from annotatebench.datasets import BENCHMARK_DATASETS


EXPECTED_STRATEGIES = {"random", "uncertainty_al", "diversity_al", "hybrid_al"}
EXPECTED_BUDGETS = {50, 100, 250, 500, 1000}
EXPECTED_SEEDS = {0, 1, 2, 3, 4}
EXPECTED_COST_SCENARIOS = {"low", "base", "high"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/benchmark_results.csv")
    return parser.parse_args()


def validate_benchmark_results(df: pd.DataFrame) -> list[str]:
    expected_datasets = set(BENCHMARK_DATASETS)
    actual_datasets = set(df["dataset"])
    expected_rows = (
        len(expected_datasets)
        * len(EXPECTED_STRATEGIES)
        * len(EXPECTED_BUDGETS)
        * len(EXPECTED_SEEDS)
        * len(EXPECTED_COST_SCENARIOS)
    )

    problems: list[str] = []
    if len(df) != expected_rows:
        problems.append(f"Expected {expected_rows} rows, found {len(df)}.")
    missing = sorted(expected_datasets - actual_datasets)
    extra = sorted(actual_datasets - expected_datasets)
    if missing:
        problems.append(f"Missing datasets: {missing}.")
    if extra:
        problems.append(f"Unexpected datasets: {extra}.")

    keys = ["dataset", "strategy", "budget", "seed", "cost_scenario"]
    duplicate_count = int(df.duplicated(keys).sum())
    if duplicate_count:
        problems.append(f"Found {duplicate_count} duplicate condition rows.")

    for dataset, dataset_df in df.groupby("dataset"):
        if set(dataset_df["strategy"]) != EXPECTED_STRATEGIES:
            problems.append(f"{dataset}: strategy coverage mismatch.")
        if set(dataset_df["budget"]) != EXPECTED_BUDGETS:
            problems.append(f"{dataset}: budget coverage mismatch.")
        if set(dataset_df["seed"]) != EXPECTED_SEEDS:
            problems.append(f"{dataset}: seed coverage mismatch.")
        if set(dataset_df["cost_scenario"]) != EXPECTED_COST_SCENARIOS:
            problems.append(f"{dataset}: cost scenario coverage mismatch.")

        base = dataset_df[dataset_df["cost_scenario"] == "base"]
        if len(base) and base["macro_f1"].nunique() == 1 and float(base["macro_f1"].iloc[0]) >= 0.99:
            problems.append(f"{dataset}: suspicious constant near-perfect macro F1.")

    return problems


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.results)
    actual_datasets = set(df["dataset"])
    problems = validate_benchmark_results(df)

    print(f"Rows: {len(df)}")
    print(f"Datasets: {sorted(actual_datasets)}")
    if problems:
        print("Validation failed:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print("Validation passed.")


if __name__ == "__main__":
    main()
