#!/usr/bin/env python3
"""Run the 10-dataset text-classification benchmark."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from annotatebench.core import AnnotationStrategy, BUDGET_LEVELS
from annotatebench.datasets import BENCHMARK_DATASETS, load_benchmark_dataset
from annotatebench.pilot import (
    DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    DOWNSTREAM_MODEL_SENTENCE_TRANSFORMER_LOGREG,
    DOWNSTREAM_MODEL_TFIDF_LOGREG,
    run_text_classification_pilot_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(BENCHMARK_DATASETS))
    parser.add_argument("--budgets", default=",".join(str(budget) for budget in BUDGET_LEVELS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--cost-scenarios", default="low,base,high")
    parser.add_argument("--max-train-examples", type=int, default=1200)
    parser.add_argument("--max-test-examples", type=int, default=1000)
    parser.add_argument("--financial-phrasebank-path")
    parser.add_argument("--trec-train-path")
    parser.add_argument("--trec-test-path")
    parser.add_argument("--banking77-train-path")
    parser.add_argument("--banking77-test-path")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument(
        "--downstream-model",
        choices=[DOWNSTREAM_MODEL_TFIDF_LOGREG, DOWNSTREAM_MODEL_SENTENCE_TRANSFORMER_LOGREG],
        default=DOWNSTREAM_MODEL_TFIDF_LOGREG,
    )
    parser.add_argument("--sentence-transformer-model", default=DEFAULT_SENTENCE_TRANSFORMER_MODEL)
    parser.add_argument("--output-csv", default="results/benchmark_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    budgets = [int(value) for value in args.budgets.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    cost_scenarios = [value.strip() for value in args.cost_scenarios.split(",") if value.strip()]
    strategies = [
        AnnotationStrategy.RANDOM,
        AnnotationStrategy.UNCERTAINTY_AL,
        AnnotationStrategy.DIVERSITY_AL,
        AnnotationStrategy.HYBRID_AL,
    ]

    frames: list[pd.DataFrame] = []
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for dataset_name in dataset_names:
        print(f"Loading {dataset_name}...")
        dataset = load_benchmark_dataset(
            dataset_name,
            financial_phrasebank_path=args.financial_phrasebank_path,
            trec_train_path=args.trec_train_path,
            trec_test_path=args.trec_test_path,
            banking77_train_path=args.banking77_train_path,
            banking77_test_path=args.banking77_test_path,
            download=args.download_data,
            max_train_examples=args.max_train_examples,
            max_test_examples=args.max_test_examples,
        )
        active_budgets = [budget for budget in budgets if budget <= len(dataset.train_texts)]
        if not active_budgets:
            active_budgets = [len(dataset.train_texts)]
        print(
            f"Running {dataset.name}: train={len(dataset.train_texts)}, "
            f"test={len(dataset.test_texts)}, budgets={active_budgets}"
        )
        frame = run_text_classification_pilot_table(
            dataset,
            strategies=strategies,
            budgets=active_budgets,
            seeds=seeds,
            cost_scenarios=cost_scenarios,
            downstream_model=args.downstream_model,
            sentence_transformer_model=args.sentence_transformer_model,
        )
        frames.append(frame)
        partial = pd.concat(frames, ignore_index=True)
        partial.to_csv(output_path, index=False)
        print(f"Saved partial results: {len(partial)} rows to {output_path}")

    result = pd.concat(frames, ignore_index=True)
    result.to_csv(output_path, index=False)
    print(f"Wrote {len(result)} rows to {output_path}")


if __name__ == "__main__":
    main()
