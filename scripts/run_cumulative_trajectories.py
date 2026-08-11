#!/usr/bin/env python3
"""Run nested cumulative acquisition trajectories for three paper datasets."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_distances

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from annotatebench.core import AnnotationStrategy, BUDGET_LEVELS
from annotatebench.costs import estimate_annotation_cost, get_cost_scenarios
from annotatebench.datasets import load_benchmark_dataset
from annotatebench.pilot import fit_and_score_text_classifier


DEFAULT_DATASETS = ("financial_phrasebank", "trec", "yelp_polarity")
SELECTION_STRATEGIES = (
    AnnotationStrategy.RANDOM,
    AnnotationStrategy.UNCERTAINTY_AL,
    AnnotationStrategy.DIVERSITY_AL,
    AnnotationStrategy.HYBRID_AL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--budgets", default=",".join(str(value) for value in BUDGET_LEVELS))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--financial-phrasebank-path")
    parser.add_argument("--trec-train-path")
    parser.add_argument("--trec-test-path")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument(
        "--output-csv",
        default="results/cumulative_trajectory_results.csv",
    )
    return parser.parse_args()


def diversity_order(texts: list[str], budget: int, seed: int) -> list[int]:
    vectors = TfidfVectorizer(min_df=1).fit_transform(texts)
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(len(texts)))]
    minimum = cosine_distances(vectors, vectors[selected]).ravel()
    minimum[selected] = -1.0
    while len(selected) < budget:
        jitter = rng.random(len(texts)) * 1e-12
        next_index = int(np.argmax(minimum + jitter))
        selected.append(next_index)
        candidate = cosine_distances(vectors, vectors[[next_index]]).ravel()
        minimum = np.minimum(minimum, candidate)
        minimum[selected] = -1.0
    return selected


def uncertainty_expand(
    texts: list[str],
    labels: list[str],
    selected: list[int],
    target_budget: int,
    rng: np.random.Generator,
    seed: int,
) -> list[int]:
    if len(selected) >= target_budget:
        return selected[:target_budget]
    remaining = [index for index in range(len(texts)) if index not in set(selected)]
    selected_labels = [labels[index] for index in selected]
    if len(set(selected_labels)) < 2:
        additions = rng.choice(
            remaining,
            size=target_budget - len(selected),
            replace=False,
        ).tolist()
        return selected + additions
    model = LogisticRegression(max_iter=1000, random_state=seed)
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    train_vectors = vectorizer.fit_transform([texts[index] for index in selected])
    model.fit(train_vectors, selected_labels)
    probabilities = model.predict_proba(
        vectorizer.transform([texts[index] for index in remaining])
    )
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1)
    tie_breakers = rng.random(len(remaining))
    ranked = [
        index
        for _, _, index in sorted(
            zip(entropy, tie_breakers, remaining),
            reverse=True,
        )
    ]
    return selected + ranked[: target_budget - len(selected)]


def build_trajectory(
    texts: list[str],
    labels: list[str],
    budgets: list[int],
    strategy: AnnotationStrategy,
    seed: int,
) -> dict[int, list[int]]:
    maximum = max(budgets)
    rng = np.random.default_rng(seed)
    if strategy == AnnotationStrategy.RANDOM:
        order = rng.permutation(len(texts))[:maximum].tolist()
        return {budget: order[:budget] for budget in budgets}
    if strategy == AnnotationStrategy.DIVERSITY_AL:
        order = diversity_order(texts, maximum, seed)
        return {budget: order[:budget] for budget in budgets}
    if strategy == AnnotationStrategy.UNCERTAINTY_AL:
        selected = rng.choice(len(texts), size=min(budgets), replace=False).tolist()
    elif strategy == AnnotationStrategy.HYBRID_AL:
        selected = diversity_order(texts, min(budgets), seed)
    else:
        raise ValueError(f"Unsupported cumulative strategy: {strategy}")
    trajectory = {min(budgets): list(selected)}
    for budget in budgets[1:]:
        selected = uncertainty_expand(texts, labels, selected, budget, rng, seed)
        trajectory[budget] = list(selected)
    return trajectory


def main() -> None:
    args = parse_args()
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    budgets = sorted(int(value) for value in args.budgets.split(",") if value.strip())
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    strategies = list(SELECTION_STRATEGIES)
    base_cost = get_cost_scenarios(["base"])[0]
    frames: list[pd.DataFrame] = []
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    for dataset_name in datasets:
        dataset = load_benchmark_dataset(
            dataset_name,
            financial_phrasebank_path=args.financial_phrasebank_path,
            trec_train_path=args.trec_train_path,
            trec_test_path=args.trec_test_path,
            download=args.download_data,
            max_train_examples=1200,
            max_test_examples=1000,
        )
        active_budgets = [value for value in budgets if value <= len(dataset.train_texts)]
        rows: list[dict[str, object]] = []
        for seed in seeds:
            for strategy in strategies:
                trajectory = build_trajectory(
                    dataset.train_texts,
                    dataset.train_labels,
                    active_budgets,
                    strategy,
                    seed,
                )
                previous: set[int] = set()
                for budget in active_budgets:
                    selected = trajectory[budget]
                    if not previous.issubset(selected):
                        raise ValueError("Cumulative trajectory is not nested")
                    previous = set(selected)
                    macro_f1, accuracy = fit_and_score_text_classifier(
                        [dataset.train_texts[index] for index in selected],
                        [dataset.train_labels[index] for index in selected],
                        dataset.test_texts,
                        dataset.test_labels,
                        seed=seed,
                    )
                    annotation_cost, selection_cost, total_cost = estimate_annotation_cost(
                        strategy,
                        len(selected),
                        base_cost,
                    )
                    rows.append(
                        {
                            "dataset": dataset.name,
                            "strategy": strategy.value,
                            "budget": len(selected),
                            "seed": seed,
                            "macro_f1": macro_f1,
                            "accuracy": accuracy,
                            "annotation_cost": annotation_cost,
                            "selection_cost": selection_cost,
                            "total_cost": total_cost,
                            "trajectory_type": "nested_cumulative",
                        }
                    )
        frames.append(pd.DataFrame(rows))
        pd.concat(frames, ignore_index=True).to_csv(output, index=False)
        print(f"Saved cumulative results through {dataset.name}")
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(output, index=False)
    print(f"Wrote {len(result)} rows to {output}")


if __name__ == "__main__":
    main()
