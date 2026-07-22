#!/usr/bin/env python3
"""Validate stratified sampling and attach measured/estimated costs to LLM rows."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from annotatebench.cost_estimation import (
    estimate_total_cost,
    estimate_total_cost_from_strata,
    length_strata,
    stratified_sample_indices,
)
from annotatebench.datasets import load_benchmark_dataset
from run_llm_strategy_benchmark import nested_random_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llm-results",
        nargs="+",
        default=[
            "results/benchmark_results_with_llm_seed0_all_datasets.csv",
            "results/benchmark_results_with_llm_seed1_2_all_datasets.csv",
        ],
    )
    parser.add_argument("--gold-results", default="results/benchmark_results.csv")
    parser.add_argument("--annotation-dir", default="results/llm_annotations")
    parser.add_argument("--sample-manifest", default="results/llm_cost_seed0_sample_manifest.csv")
    parser.add_argument("--sample-dir", default="results/llm_cost_samples")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--input-usd-per-million-tokens", type=float, default=0.15)
    parser.add_argument("--output-usd-per-million-tokens", type=float, default=0.60)
    parser.add_argument("--price-checked-at", default="2026-07-22")
    parser.add_argument("--price-source-url", default="https://developers.openai.com/api/docs/models/gpt-4o-mini")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--financial-phrasebank-path")
    parser.add_argument("--trec-train-path")
    parser.add_argument("--trec-test-path")
    parser.add_argument("--banking77-train-path")
    parser.add_argument("--banking77-test-path")
    parser.add_argument("--output-csv", default="results/benchmark_results_cost_unified.csv")
    parser.add_argument("--validation-csv", default="results/llm_cost_sampling_validation.csv")
    return parser.parse_args()


def row_costs(frame: pd.DataFrame, input_price: float, output_price: float) -> np.ndarray:
    input_tokens = frame["input_tokens"].fillna(0).astype(float).to_numpy(copy=True)
    output_tokens = frame["output_tokens"].fillna(0).astype(float).to_numpy(copy=True)
    for index, raw_response in enumerate(frame.get("raw_response", pd.Series(dtype=str)).fillna("")):
        if input_tokens[index] > 0 or output_tokens[index] > 0 or not raw_response:
            continue
        try:
            usage = json.loads(raw_response).get("usage", {})
            input_tokens[index] = float(usage.get("prompt_tokens") or 0)
            output_tokens[index] = float(usage.get("completion_tokens") or 0)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    explicit_no_call = np.zeros(len(frame), dtype=bool)
    for status_column in ("annotation_status", "notes"):
        if status_column in frame:
            explicit_no_call |= frame[status_column].fillna("").eq("dry_run").to_numpy()
    missing = (input_tokens + output_tokens <= 0) & ~explicit_no_call
    if missing.any():
        identifiers = frame.loc[missing, "example_id"].astype(str).tolist() if "example_id" in frame else []
        raise ValueError(f"Token logs are missing for {identifiers}; zero-cost placeholders are not allowed.")
    return (
        input_tokens * input_price / 1_000_000
        + output_tokens * output_price / 1_000_000
    )


def validate_sample_size(annotation_dir: Path, sample_size: int, input_price: float, output_price: float) -> pd.DataFrame:
    rows = []
    for path in sorted(annotation_dir.glob("*_llm_annotator_seed[12].csv")):
        frame = pd.read_csv(path)
        frame = frame[(frame["input_tokens"].fillna(0) > 0) & (frame["output_tokens"].fillna(0) > 0)]
        frame = frame.drop_duplicates("example_id", keep="last").reset_index(drop=True)
        lengths = frame["input_tokens"].astype(int).tolist()
        sampled = stratified_sample_indices(lengths, sample_size=sample_size, random_state=0)
        costs = row_costs(frame, input_price, output_price)
        estimate = estimate_total_cost(lengths, sampled, costs[sampled].tolist(), random_state=0)
        actual = float(costs.sum())
        rows.append(
            {
                "dataset": frame.loc[0, "dataset_name"],
                "seed": int(frame.loc[0, "seed"]),
                "sample_size": sample_size,
                "estimated_cost": estimate.mean,
                "actual_cost": actual,
                "absolute_percentage_error": abs(estimate.mean - actual) / actual,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.input_usd_per_million_tokens <= 0 or args.output_usd_per_million_tokens <= 0:
        raise SystemExit("Token prices must be positive.")
    annotation_dir = Path(args.annotation_dir)
    validation = validate_sample_size(
        annotation_dir,
        args.sample_size,
        args.input_usd_per_million_tokens,
        args.output_usd_per_million_tokens,
    )
    if validation.empty:
        raise SystemExit("No complete seed-1/2 annotation logs were found.")
    chosen_size = args.sample_size
    if validation["absolute_percentage_error"].mean() > 0.10 and chosen_size < 50:
        chosen_size = 50
        validation = validate_sample_size(
            annotation_dir,
            chosen_size,
            args.input_usd_per_million_tokens,
            args.output_usd_per_million_tokens,
        )
    validation_path = Path(args.validation_csv)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation.to_csv(validation_path, index=False)
    if validation["absolute_percentage_error"].mean() > 0.10:
        raise SystemExit("The 50-row validation still exceeds the 10% mean error threshold.")

    manifest_path = Path(args.sample_manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Seed-0 sample manifest is missing: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    llm = pd.concat([pd.read_csv(path) for path in args.llm_results], ignore_index=True, sort=False)
    estimates: dict[tuple[str, int, int], dict[str, object]] = {}
    budgets = sorted(int(value) for value in llm["budget"].unique())
    for dataset_name in sorted(llm["dataset"].unique()):
        dataset = load_benchmark_dataset(
            str(dataset_name),
            download=args.download_data,
            financial_phrasebank_path=args.financial_phrasebank_path,
            trec_train_path=args.trec_train_path,
            trec_test_path=args.trec_test_path,
            banking77_train_path=args.banking77_train_path,
            banking77_test_path=args.banking77_test_path,
        )
        for seed in (1, 2):
            log = pd.read_csv(annotation_dir / f"{dataset_name}_llm_annotator_seed{seed}.csv")
            log["_has_usage"] = (
                (log["input_tokens"].fillna(0) > 0)
                | (log["output_tokens"].fillna(0) > 0)
                | log["raw_response"].fillna("").str.contains('"usage"', regex=False)
            )
            log = log.sort_values("_has_usage").drop_duplicates("example_id", keep="last").set_index("example_id")
            pool = nested_random_indices(len(dataset.train_texts), max(budgets), seed)
            pool_ids = [f"{dataset_name}_train_{index}" for index in sorted(pool)]
            selected_log = log.loc[pool_ids].copy()
            selected_log["_cost"] = row_costs(
                selected_log.reset_index(),
                args.input_usd_per_million_tokens,
                args.output_usd_per_million_tokens,
            )
            for budget in budgets:
                ids = [f"{dataset_name}_train_{index}" for index in sorted(pool[:budget])]
                total = float(selected_log.loc[ids, "_cost"].sum())
                estimates[(dataset_name, seed, budget)] = {
                    "total_cost": total,
                    "cost_estimation_method": "measured",
                    "cost_sample_size": budget,
                    "estimated_cost_lower_95": total,
                    "estimated_cost_upper_95": total,
                }

        sample_path = Path(args.sample_dir) / f"{dataset_name}_llm_cost_sample_seed0.csv"
        if not sample_path.exists():
            raise SystemExit(f"Seed-0 API sample is missing: {sample_path}")
        sample = pd.read_csv(sample_path).drop_duplicates("example_id").set_index("example_id")
        manifest_group = manifest[manifest["dataset"] == dataset_name]
        sample = sample.loc[manifest_group["example_id"]]
        sample_costs = row_costs(sample.reset_index(), args.input_usd_per_million_tokens, args.output_usd_per_million_tokens)
        full_pool = nested_random_indices(len(dataset.train_texts), max(budgets), 0)
        full_lengths = [len(dataset.train_texts[index].split()) for index in full_pool]
        full_strata = length_strata(full_lengths)
        sample_strata = [
            {"short": 0, "medium": 1, "long": 2}[value]
            for value in manifest_group["length_stratum"]
        ]
        for budget in budgets:
            estimate = estimate_total_cost_from_strata(
                full_strata[:budget].tolist(), sample_strata, sample_costs.tolist(), random_state=0
            )
            estimates[(dataset_name, 0, budget)] = {
                "total_cost": estimate.mean,
                "cost_estimation_method": "estimated_from_stratified_sample",
                "cost_sample_size": estimate.sample_size,
                "estimated_cost_lower_95": estimate.lower_95,
                "estimated_cost_upper_95": estimate.upper_95,
            }

    for index, row in llm.iterrows():
        values = estimates[(str(row["dataset"]), int(row["seed"]), int(row["budget"]))]
        for key, value in values.items():
            llm.loc[index, key] = value
        llm.loc[index, "annotation_cost"] = values["total_cost"]
        llm.loc[index, "selection_cost"] = 0.0
        llm.loc[index, "cost_source"] = "OpenAI API usage"
        llm.loc[index, "price_checked_at"] = args.price_checked_at
        llm.loc[index, "price_source_url"] = args.price_source_url
    gold = pd.read_csv(args.gold_results)
    gold["cost_estimation_method"] = "human_cost_scenario"
    gold["cost_sample_size"] = gold["budget"]
    gold["estimated_cost_lower_95"] = gold["total_cost"]
    gold["estimated_cost_upper_95"] = gold["total_cost"]
    combined = pd.concat([gold, llm], ignore_index=True, sort=False)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_csv, index=False)
    print(f"Validation mean error: {validation['absolute_percentage_error'].mean():.3%} with n={chosen_size}")
    print(f"Wrote {len(combined)} unified rows to {args.output_csv}")


if __name__ == "__main__":
    main()
