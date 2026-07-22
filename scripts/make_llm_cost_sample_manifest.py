#!/usr/bin/env python3
"""Create a reproducible seed-0 length-stratified LLM cost sample manifest."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from annotatebench.cost_estimation import length_strata, stratified_sample_indices
from annotatebench.datasets import BENCHMARK_DATASETS, load_benchmark_dataset
from run_llm_strategy_benchmark import nested_random_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(BENCHMARK_DATASETS))
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--max-budget", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--financial-phrasebank-path")
    parser.add_argument("--trec-train-path")
    parser.add_argument("--trec-test-path")
    parser.add_argument("--banking77-train-path")
    parser.add_argument("--banking77-test-path")
    parser.add_argument("--output-csv", default="results/llm_cost_seed0_sample_manifest.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for name in [value.strip() for value in args.datasets.split(",") if value.strip()]:
        dataset = load_benchmark_dataset(
            name,
            download=args.download_data,
            financial_phrasebank_path=args.financial_phrasebank_path,
            trec_train_path=args.trec_train_path,
            trec_test_path=args.trec_test_path,
            banking77_train_path=args.banking77_train_path,
            banking77_test_path=args.banking77_test_path,
        )
        pool = nested_random_indices(len(dataset.train_texts), args.max_budget, args.seed)
        texts = [dataset.train_texts[index] for index in pool]
        lengths = [len(text.split()) for text in texts]
        strata = length_strata(lengths)
        sample = stratified_sample_indices(lengths, sample_size=args.sample_size, random_state=args.seed)
        for pool_position in sample:
            dataset_index = pool[pool_position]
            rows.append(
                {
                    "dataset": dataset.name,
                    "seed": args.seed,
                    "pool_position": pool_position,
                    "dataset_index": dataset_index,
                    "example_id": f"{dataset.name}_train_{dataset_index}",
                    "length_stratum": ["short", "medium", "long"][int(strata[pool_position])],
                    "approx_text_tokens": lengths[pool_position],
                    "population_size": len(pool),
                }
            )
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows)} sample rows to {output}")


if __name__ == "__main__":
    main()
