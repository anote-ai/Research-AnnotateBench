#!/usr/bin/env python3
"""Measure fixed-label selection runtime and utility as the unlabeled pool grows."""
from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from annotatebench.core import AnnotationStrategy
from annotatebench.pilot import fit_and_score_text_classifier, select_budget_indices


SELECTION_STRATEGIES = (
    AnnotationStrategy.RANDOM,
    AnnotationStrategy.UNCERTAINTY_AL,
    AnnotationStrategy.DIVERSITY_AL,
    AnnotationStrategy.HYBRID_AL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-sizes", default="1200,5000,10000")
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument(
        "--output-csv",
        default="results/pool_scalability_results.csv",
    )
    parser.add_argument(
        "--output-figure",
        default="figures/pool_scalability_runtime.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pool_sizes = sorted(int(value) for value in args.pool_sizes.split(",") if value.strip())
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    from datasets import load_dataset

    train_split = load_dataset("fancyzhx/ag_news", split="train")
    test_split = load_dataset("fancyzhx/ag_news", split="test")
    train_split = train_split.shuffle(seed=42).select(range(max(pool_sizes)))
    test_split = test_split.shuffle(seed=42).select(range(1000))
    label_names = train_split.features["label"].names
    train_texts = list(train_split["text"])
    train_labels = [label_names[value] for value in train_split["label"]]
    test_texts = list(test_split["text"])
    test_labels = [label_names[value] for value in test_split["label"]]
    if len(train_texts) < max(pool_sizes):
        raise ValueError(
            f"AG News loader returned {len(train_texts)} examples; "
            f"need {max(pool_sizes)}"
        )
    rows: list[dict[str, object]] = []
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    for pool_size in pool_sizes:
        texts = train_texts[:pool_size]
        labels = train_labels[:pool_size]
        for seed in seeds:
            for strategy in SELECTION_STRATEGIES:
                started = perf_counter()
                selected = select_budget_indices(
                    texts,
                    labels,
                    budget=args.budget,
                    strategy=strategy,
                    seed=seed,
                )
                selection_seconds = perf_counter() - started
                macro_f1, accuracy = fit_and_score_text_classifier(
                    [texts[index] for index in selected],
                    [labels[index] for index in selected],
                    test_texts,
                    test_labels,
                    seed=seed,
                )
                rows.append(
                    {
                        "dataset": "ag_news",
                        "pool_size": pool_size,
                        "label_budget": args.budget,
                        "strategy": strategy.value,
                        "seed": seed,
                        "selection_seconds": selection_seconds,
                        "macro_f1": macro_f1,
                        "accuracy": accuracy,
                        "machine": platform.machine(),
                        "python_version": platform.python_version(),
                    }
                )
        pd.DataFrame(rows).to_csv(output, index=False)
        print(f"Saved scalability results through pool size {pool_size}")
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    summary = (
        result.groupby(["pool_size", "strategy"], as_index=False)
        .agg(
            mean_seconds=("selection_seconds", "mean"),
            std_seconds=("selection_seconds", "std"),
        )
    )
    figure = Path(args.output_figure)
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6.4, 3.8))
    for strategy, group in summary.groupby("strategy"):
        group = group.sort_values("pool_size")
        axis.errorbar(
            group["pool_size"],
            group["mean_seconds"],
            yerr=group["std_seconds"].fillna(0),
            marker="o",
            capsize=3,
            label=strategy,
        )
    axis.set_yscale("log")
    axis.set_xlabel("Unlabeled pool size")
    axis.set_ylabel("Selection time (seconds, log scale)")
    axis.set_title(f"AG News selection runtime at a {args.budget}-label budget")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure, dpi=220)
    plt.close(fig)
    print(f"Wrote {len(result)} rows to {output}")
    print(f"Wrote runtime figure to {figure}")


if __name__ == "__main__":
    main()
