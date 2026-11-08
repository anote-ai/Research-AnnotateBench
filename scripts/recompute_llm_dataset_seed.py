#!/usr/bin/env python3
"""Recompute one dataset/seed's benchmark rows from a row-level annotation cache."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from annotatebench.datasets import load_benchmark_dataset
from annotatebench.llm import dataset_label_names
from annotatebench.metrics.lari import expected_calibration_error, lari_score
from annotatebench.pilot import fit_and_score_text_classifier

from run_llm_strategy_benchmark import nested_random_indices


def complete_annotation_cost(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        return None
    return float(numeric.sum())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-results", required=True)
    parser.add_argument("--annotation-cache", required=True)
    parser.add_argument("--output-results", required=True)
    parser.add_argument("--financial-phrasebank-path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = pd.read_csv(args.source_results)
    cache = pd.read_csv(args.annotation_cache)
    dataset = load_benchmark_dataset(
        args.dataset,
        financial_phrasebank_path=args.financial_phrasebank_path,
    )
    target_mask = (results["dataset"] == args.dataset) & (results["seed"] == args.seed)
    budgets = sorted(results.loc[target_mask, "budget"].unique())
    if not budgets:
        raise SystemExit("No source result rows matched the requested dataset and seed.")
    selected_pool = nested_random_indices(len(dataset.train_texts), max(budgets), args.seed)
    cache["index"] = cache["example_id"].str.rsplit("_", n=1).str[-1].astype(int)
    cache_by_index = cache.set_index("index", verify_integrity=True)
    label_names = dataset_label_names(dataset)

    for budget in budgets:
        selected = sorted(selected_pool[: int(budget)])
        annotations = cache_by_index.loc[selected]
        bad = annotations[annotations["notes"].fillna("").eq("dry_run")]
        if not bad.empty:
            raise SystemExit(f"Dry-run rows remain in budget {budget}: {bad['example_id'].tolist()}")
        predicted = annotations["predicted_label"].astype(str).tolist()
        gold = [dataset.train_labels[index] for index in selected]
        train_texts = [dataset.train_texts[index] for index in selected]
        macro_f1, accuracy = fit_and_score_text_classifier(
            train_texts,
            predicted,
            dataset.test_texts,
            dataset.test_labels,
            seed=args.seed,
        )
        label_macro_f1 = f1_score(gold, predicted, labels=label_names, average="macro")
        correctness = [actual == guess for actual, guess in zip(gold, predicted)]
        confidence = annotations["confidence"].astype(float).tolist()
        annotation_cost = complete_annotation_cost(annotations["cost_usd"])
        budget_mask = target_mask & (results["budget"] == budget)
        results.loc[budget_mask, "macro_f1"] = macro_f1
        results.loc[budget_mask, "accuracy"] = accuracy
        results.loc[budget_mask, "llm_label_accuracy"] = accuracy_score(gold, predicted)
        results.loc[budget_mask, "llm_label_macro_f1"] = label_macro_f1
        results.loc[budget_mask, "llm_ece"] = expected_calibration_error(confidence, correctness)
        results.loc[budget_mask, "llm_lari"] = lari_score(label_macro_f1, confidence, correctness)
        if annotation_cost is not None:
            results.loc[budget_mask, "annotation_cost"] = annotation_cost
            results.loc[budget_mask, "total_cost"] = annotation_cost
            for column in ["estimated_cost_lower_95", "estimated_cost_upper_95"]:
                if column in results.columns:
                    results.loc[budget_mask, column] = annotation_cost

    output = Path(args.output_results)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print(f"Wrote recomputed results to {output}")


if __name__ == "__main__":
    main()
