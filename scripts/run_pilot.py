#!/usr/bin/env python3
"""Run a real text-classification pilot."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from annotatebench.core import AnnotationStrategy, BUDGET_LEVELS
from annotatebench.evaluate import pareto_frontier, strategy_comparison
from annotatebench.datasets import load_banking77, load_financial_phrasebank, load_trec
from annotatebench.pilot import (
    load_text_classification_csv,
    run_text_classification_pilot,
    run_text_classification_pilot_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["csv", "financial_phrasebank", "trec", "banking77"], default="csv")
    parser.add_argument("--train-csv")
    parser.add_argument("--test-csv")
    parser.add_argument("--financial-phrasebank-path")
    parser.add_argument("--trec-train-path")
    parser.add_argument("--trec-test-path")
    parser.add_argument("--banking77-train-path")
    parser.add_argument("--banking77-test-path")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument(
        "--budgets",
        default=",".join(str(b) for b in BUDGET_LEVELS),
        help="Comma-separated annotation budgets.",
    )
    parser.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated random seeds for CSV output.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cost-per-sample", type=float, default=0.10)
    parser.add_argument(
        "--cost-scenarios",
        default="low,base,high",
        help="Comma-separated cost scenarios to include in CSV output.",
    )
    parser.add_argument("--output-csv", default="results/pilot_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budgets = [int(value) for value in args.budgets.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    cost_scenarios = [value.strip() for value in args.cost_scenarios.split(",") if value.strip()]
    if args.dataset == "financial_phrasebank":
        dataset = load_financial_phrasebank(
            args.financial_phrasebank_path,
            seed=args.seed,
            download=args.download_data,
        )
    elif args.dataset == "trec":
        dataset = load_trec(
            args.trec_train_path,
            args.trec_test_path,
            download=args.download_data,
        )
    elif args.dataset == "banking77":
        dataset = load_banking77(
            args.banking77_train_path,
            args.banking77_test_path,
            download=args.download_data,
        )
    else:
        if not args.train_csv or not args.test_csv:
            raise SystemExit("--train-csv and --test-csv are required when --dataset csv")
        dataset = load_text_classification_csv(
            args.train_csv,
            args.test_csv,
            text_column=args.text_column,
            label_column=args.label_column,
            dataset_name=args.dataset_name,
        )
    budgets = [budget for budget in budgets if budget <= len(dataset.train_texts)]
    if not budgets:
        budgets = [len(dataset.train_texts)]
    curves = run_text_classification_pilot(
        dataset,
        strategies=[
            AnnotationStrategy.RANDOM,
            AnnotationStrategy.UNCERTAINTY_AL,
            AnnotationStrategy.DIVERSITY_AL,
            AnnotationStrategy.HYBRID_AL,
        ],
        budgets=budgets,
        seed=seeds[0],
        cost_per_sample=args.cost_per_sample,
    )
    result_table = run_text_classification_pilot_table(
        dataset,
        strategies=[
            AnnotationStrategy.RANDOM,
            AnnotationStrategy.UNCERTAINTY_AL,
            AnnotationStrategy.DIVERSITY_AL,
            AnnotationStrategy.HYBRID_AL,
        ],
        budgets=budgets,
        seeds=seeds,
        cost_per_sample=args.cost_per_sample,
        cost_scenarios=cost_scenarios,
    )
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_table.to_csv(output_path, index=False)

    print(f"Dataset: {dataset.name}")
    print(f"Train: {len(dataset.train_texts)} examples, Test: {len(dataset.test_texts)} examples")
    print(f"Saved results: {output_path}")
    print(f"Cost scenarios: {', '.join(cost_scenarios)}")
    print("\nLearning curves:")
    for curve in curves:
        values = ", ".join(f"{pt.budget}: {pt.f1:.4f}" for pt in curve.points)
        print(f"  {curve.config.strategy.value}: {values}")

    print("\nStrategy comparison:")
    comparison = strategy_comparison(curves)
    for strategy, info in sorted(comparison.items(), key=lambda item: -item[1]["best_f1"]):
        print(
            f"  {strategy}: best_f1={info['best_f1']:.4f}, "
            f"cost=${info['total_cost']:.2f}, AUC={info['auc']:.1f}"
        )

    print("\nPareto frontier:")
    points = [(info["total_cost"], info["best_f1"]) for info in comparison.values()]
    for cost, f1 in pareto_frontier(points):
        print(f"  cost=${cost:.2f}, F1={f1:.4f}")


if __name__ == "__main__":
    main()
