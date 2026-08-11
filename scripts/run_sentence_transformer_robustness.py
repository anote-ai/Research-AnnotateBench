#!/usr/bin/env python3
"""Run missing robustness seeds with one embedding pass per dataset."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from annotatebench.core import AnnotationStrategy, BUDGET_LEVELS
from annotatebench.costs import estimate_annotation_cost, get_cost_scenarios
from annotatebench.datasets import BENCHMARK_DATASETS, load_benchmark_dataset
from annotatebench.pilot import (
    DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    DOWNSTREAM_MODEL_SENTENCE_TRANSFORMER_LOGREG,
    select_budget_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(BENCHMARK_DATASETS))
    parser.add_argument("--budgets", default=",".join(str(b) for b in BUDGET_LEVELS))
    parser.add_argument("--seeds", default="3,4")
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
        "--sentence-transformer-model",
        default=DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    )
    parser.add_argument(
        "--output-csv",
        default="results/benchmark_results_sentence_transformer_logreg_5seed.csv",
    )
    return parser.parse_args()


def _fit_embedding_classifier(
    train_vectors: np.ndarray,
    train_labels: list[str],
    test_vectors: np.ndarray,
    test_labels: list[str],
    selected: list[int],
    seed: int,
) -> tuple[float, float]:
    selected_labels = [train_labels[index] for index in selected]
    if len(set(selected_labels)) < 2:
        classifier = DummyClassifier(strategy="most_frequent")
    else:
        classifier = LogisticRegression(max_iter=1000, random_state=seed)
    classifier.fit(train_vectors[selected], selected_labels)
    predictions = classifier.predict(test_vectors)
    return (
        float(f1_score(test_labels, predictions, average="macro")),
        float(accuracy_score(test_labels, predictions)),
    )


def main() -> None:
    args = parse_args()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Install the sentence-transformers optional dependency first."
        ) from exc

    dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    budgets = [int(value) for value in args.budgets.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    scenarios = get_cost_scenarios(
        [value.strip() for value in args.cost_scenarios.split(",") if value.strip()]
    )
    strategies = [
        AnnotationStrategy.RANDOM,
        AnnotationStrategy.UNCERTAINTY_AL,
        AnnotationStrategy.DIVERSITY_AL,
        AnnotationStrategy.HYBRID_AL,
    ]
    encoder = SentenceTransformer(args.sentence_transformer_model)
    frames: list[pd.DataFrame] = []
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for dataset_name in dataset_names:
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
        print(
            f"Encoding {dataset.name}: train={len(dataset.train_texts)}, "
            f"test={len(dataset.test_texts)}"
        )
        train_vectors = np.asarray(
            encoder.encode(dataset.train_texts, show_progress_bar=False)
        )
        test_vectors = np.asarray(
            encoder.encode(dataset.test_texts, show_progress_bar=False)
        )
        rows: list[dict[str, object]] = []
        for seed in seeds:
            for strategy in strategies:
                for budget in active_budgets:
                    selected = select_budget_indices(
                        dataset.train_texts,
                        dataset.train_labels,
                        budget=budget,
                        strategy=strategy,
                        seed=seed,
                    )
                    macro_f1, accuracy = _fit_embedding_classifier(
                        train_vectors,
                        dataset.train_labels,
                        test_vectors,
                        dataset.test_labels,
                        selected,
                        seed,
                    )
                    for scenario in scenarios:
                        annotation_cost, selection_cost, total_cost = (
                            estimate_annotation_cost(
                                strategy,
                                len(selected),
                                scenario,
                            )
                        )
                        rows.append(
                            {
                                "dataset": dataset.name,
                                "strategy": strategy.value,
                                "budget": len(selected),
                                "seed": seed,
                                "downstream_model": (
                                    DOWNSTREAM_MODEL_SENTENCE_TRANSFORMER_LOGREG
                                ),
                                "embedding_model": args.sentence_transformer_model,
                                "macro_f1": macro_f1,
                                "accuracy": accuracy,
                                "cost_scenario": scenario.name,
                                "cost_source": scenario.source_name,
                                "cost_source_url": scenario.source_url,
                                "cost_checked_at": scenario.checked_at,
                                "human_cost_per_label": scenario.human_cost_per_label,
                                "selection_cost_per_example": (
                                    scenario.selection_cost_per_example.get(
                                        strategy, 0.0
                                    )
                                ),
                                "annotation_cost": annotation_cost,
                                "selection_cost": selection_cost,
                                "total_cost": total_cost,
                            }
                        )
        frames.append(pd.DataFrame(rows))
        pd.concat(frames, ignore_index=True).to_csv(output_path, index=False)
        print(f"Saved partial results to {output_path}")

    result = pd.concat(frames, ignore_index=True)
    result.to_csv(output_path, index=False)
    print(f"Wrote {len(result)} rows to {output_path}")


if __name__ == "__main__":
    main()
