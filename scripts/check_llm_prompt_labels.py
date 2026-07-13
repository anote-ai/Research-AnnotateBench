#!/usr/bin/env python3
"""Validate LLM prompt label meanings against benchmark dataset labels."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from annotatebench.datasets import BENCHMARK_DATASETS, load_benchmark_dataset
from annotatebench.llm import DATASET_LABEL_DESCRIPTIONS, dataset_label_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(BENCHMARK_DATASETS))
    parser.add_argument("--max-train-examples", type=int, default=1200)
    parser.add_argument("--max-test-examples", type=int, default=1000)
    parser.add_argument("--financial-phrasebank-path")
    parser.add_argument("--trec-train-path")
    parser.add_argument("--trec-test-path")
    parser.add_argument("--banking77-train-path")
    parser.add_argument("--banking77-test-path")
    parser.add_argument("--download-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    failures: list[str] = []
    for dataset_name in dataset_names:
        try:
            dataset = load_benchmark_dataset(
                dataset_name,
                max_train_examples=args.max_train_examples,
                max_test_examples=args.max_test_examples,
                financial_phrasebank_path=args.financial_phrasebank_path,
                trec_train_path=args.trec_train_path,
                trec_test_path=args.trec_test_path,
                banking77_train_path=args.banking77_train_path,
                banking77_test_path=args.banking77_test_path,
                download=args.download_data,
            )
        except Exception as exc:
            print(f"ERROR {dataset_name}: could not load dataset: {exc}")
            failures.append(f"{dataset_name}: could not load dataset: {exc}")
            continue
        labels = dataset_label_names(dataset)
        descriptions = DATASET_LABEL_DESCRIPTIONS.get(dataset.name, {})
        missing = [label for label in labels if str(label) not in descriptions]
        numeric_labels = [label for label in labels if str(label).isdigit()]
        status = "OK"
        if numeric_labels and missing:
            status = "FAIL"
            failures.append(
                f"{dataset.name}: numeric labels missing descriptions: {missing}; labels={labels}"
            )
        elif len(labels) > 10 and missing:
            status = "WARN"
        print(f"{status} {dataset.name}: {len(labels)} labels")
        print("  labels:", ", ".join(str(label) for label in labels))
        if descriptions:
            print("  described:", ", ".join(str(label) for label in labels if str(label) in descriptions))
        if missing:
            print("  missing descriptions:", ", ".join(str(label) for label in missing))

    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
