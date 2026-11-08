#!/usr/bin/env python3
"""Build the budget-250 LLM comparison with a sample-matched gold baseline."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from annotatebench.core import AnnotationStrategy
from annotatebench.pilot import select_budget_indices

from run_llm_strategy_benchmark import nested_random_indices


DEFAULT_POOL_SIZES = {
    "twenty_newsgroups": 1177,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-results", default="results/benchmark_results.csv")
    parser.add_argument("--llm-results", nargs="+", required=True)
    parser.add_argument(
        "--annotation-dir",
        nargs="+",
        default=["results/llm_annotations"],
        help="Cache directories in base-to-overlay order; later directories take precedence.",
    )
    parser.add_argument("--budget", type=int, default=250)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--cost-scenario", default="base")
    parser.add_argument("--output-per-seed", default="results/llm_matched_gold_per_seed.csv")
    parser.add_argument("--output-summary", default="results/llm_matched_gold_summary.csv")
    parser.add_argument(
        "--skip-cache-audit",
        action="store_true",
        help="Skip row-level cache validation (intended only for synthetic tests).",
    )
    return parser.parse_args()


def expected_selected_indices(dataset: str, seed: int, budget: int) -> list[int]:
    pool_size = DEFAULT_POOL_SIZES.get(dataset, 1200)
    nested = sorted(nested_random_indices(pool_size, budget, seed)[:budget])
    independent = select_budget_indices(
        [str(i) for i in range(pool_size)],
        [str(i % 2) for i in range(pool_size)],
        budget,
        AnnotationStrategy.RANDOM,
        seed=seed,
    )
    if nested != independent:
        raise AssertionError(
            f"Budget-{budget} nested and independent random selections differ for {dataset}, seed {seed}."
        )
    return nested


def audit_annotation_cache(
    annotation_dir: str | Path | list[str | Path],
    datasets: list[str],
    seeds: list[int],
    budget: int,
) -> None:
    roots = [Path(annotation_dir)] if isinstance(annotation_dir, (str, Path)) else [Path(p) for p in annotation_dir]
    problems: list[str] = []
    for dataset in datasets:
        for seed in seeds:
            candidates = [root / f"{dataset}_llm_annotator_seed{seed}.csv" for root in roots]
            existing = [path for path in candidates if path.exists()]
            if not existing:
                if seed == 0:
                    continue
                problems.append(f"missing cache: {candidates[-1]}")
                continue
            path = existing[-1]
            frame = pd.read_csv(path)
            frame["index"] = frame["example_id"].str.rsplit("_", n=1).str[-1].astype(int)
            expected = set(expected_selected_indices(dataset, seed, budget))
            selected = frame[frame["index"].isin(expected)].copy()
            if set(selected["index"]) != expected:
                missing = sorted(expected.difference(selected["index"]))
                problems.append(f"{dataset} seed {seed}: missing selected IDs {missing[:5]}")
            contaminated = selected[selected["notes"].fillna("").eq("dry_run")]
            for row in contaminated.itertuples():
                problems.append(
                    f"{dataset} seed {seed} {row.example_id}: non-production note {row.notes!r}"
                )
            for row in selected.itertuples():
                try:
                    raw = json.loads(row.raw_response)
                except (TypeError, json.JSONDecodeError):
                    continue
                response_model = raw.get("model") if isinstance(raw, dict) else None
                if response_model and response_model != "gpt-4o-mini-2024-07-18":
                    problems.append(
                        f"{dataset} seed {seed} {row.example_id}: unexpected response model {response_model}"
                    )
    if problems:
        raise ValueError("Annotation-cache audit failed:\n- " + "\n- ".join(problems))


def make_matched_gold_tables(
    gold: pd.DataFrame,
    llm: pd.DataFrame,
    *,
    seeds: list[int],
    budget: int,
    cost_scenario: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    llm = llm[
        (llm["cost_scenario"] == cost_scenario)
        & (llm["seed"].isin(seeds))
        & (llm["budget"] == budget)
    ].copy()
    gold = gold[
        (gold["cost_scenario"] == cost_scenario)
        & (gold["seed"].isin(seeds))
        & (gold["budget"] == budget)
        & (gold["dataset"].isin(llm["dataset"].unique()))
    ].copy()
    if llm.empty or gold.empty:
        raise ValueError("No matching LLM or gold rows were found.")

    matched = gold[gold["strategy"] == AnnotationStrategy.RANDOM.value][
        ["dataset", "budget", "seed", "macro_f1", "accuracy"]
    ].rename(
        columns={
            "macro_f1": "matched_gold_macro_f1",
            "accuracy": "matched_gold_accuracy",
        }
    )
    best_indices = gold.groupby(["dataset", "budget", "seed"])["macro_f1"].idxmax()
    best = gold.loc[best_indices, ["dataset", "budget", "seed", "strategy", "macro_f1"]].rename(
        columns={
            "strategy": "best_selector_strategy",
            "macro_f1": "best_selector_macro_f1",
        }
    )
    per_seed = (
        llm.merge(matched, on=["dataset", "budget", "seed"], validate="one_to_one")
        .merge(best, on=["dataset", "budget", "seed"], validate="one_to_one")
        .rename(columns={"macro_f1": "llm_macro_f1"})
    )
    per_seed["paired_gap"] = per_seed["llm_macro_f1"] - per_seed["matched_gold_macro_f1"]
    per_seed = per_seed[
        [
            "dataset",
            "budget",
            "seed",
            "llm_macro_f1",
            "matched_gold_macro_f1",
            "paired_gap",
            "best_selector_macro_f1",
            "best_selector_strategy",
            "llm_label_accuracy",
        ]
    ].sort_values(["dataset", "seed"]).reset_index(drop=True)

    summary = (
        per_seed.groupby(["dataset", "budget"], as_index=False)
        .agg(
            llm_macro_f1=("llm_macro_f1", "mean"),
            matched_gold_macro_f1=("matched_gold_macro_f1", "mean"),
            paired_gap=("paired_gap", "mean"),
            paired_gap_sd=("paired_gap", "std"),
            best_selector_macro_f1=("best_selector_macro_f1", "mean"),
            llm_label_accuracy=("llm_label_accuracy", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values("dataset")
        .reset_index(drop=True)
    )
    return per_seed, summary


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    llm = pd.concat([pd.read_csv(path) for path in args.llm_results], ignore_index=True)
    gold = pd.read_csv(args.gold_results)
    datasets = sorted(llm.loc[llm["budget"] == args.budget, "dataset"].unique())
    if not args.skip_cache_audit:
        audit_annotation_cache(args.annotation_dir, datasets, seeds, args.budget)
    per_seed, summary = make_matched_gold_tables(
        gold,
        llm,
        seeds=seeds,
        budget=args.budget,
        cost_scenario=args.cost_scenario,
    )
    per_seed_path = Path(args.output_per_seed)
    summary_path = Path(args.output_summary)
    per_seed_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(per_seed_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {len(per_seed)} per-seed rows to {per_seed_path}")
    print(f"Wrote {len(summary)} summary rows to {summary_path}")


if __name__ == "__main__":
    main()
