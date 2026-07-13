from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.pipeline import Pipeline

from .core import (
    AnnotationStrategy,
    BUDGET_LEVELS,
    ExperimentConfig,
    LearningCurve,
    LearningCurvePoint,
    NLPTask,
)
from .costs import CostScenario, estimate_annotation_cost, get_cost_scenarios

DOWNSTREAM_MODEL_TFIDF_LOGREG = "tfidf_logreg"
DOWNSTREAM_MODEL_SENTENCE_TRANSFORMER_LOGREG = "sentence_transformer_logreg"
DEFAULT_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUPPORTED_DOWNSTREAM_MODELS = {
    DOWNSTREAM_MODEL_TFIDF_LOGREG,
    DOWNSTREAM_MODEL_SENTENCE_TRANSFORMER_LOGREG,
}


@dataclass(frozen=True)
class TextClassificationDataset:
    name: str
    train_texts: List[str]
    train_labels: List[str]
    test_texts: List[str]
    test_labels: List[str]
    label_names: List[str] | None = None


def load_text_classification_csv(
    train_path: str | Path,
    test_path: str | Path,
    text_column: str = "text",
    label_column: str = "label",
    dataset_name: str | None = None,
) -> TextClassificationDataset:
    """Load a labeled text classification train/test split from CSV files."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    for column in (text_column, label_column):
        if column not in train_df.columns:
            raise ValueError(f"Missing column in train CSV: {column}")
        if column not in test_df.columns:
            raise ValueError(f"Missing column in test CSV: {column}")

    name = dataset_name or Path(train_path).stem
    return TextClassificationDataset(
        name=name,
        train_texts=train_df[text_column].astype(str).tolist(),
        train_labels=train_df[label_column].astype(str).tolist(),
        test_texts=test_df[text_column].astype(str).tolist(),
        test_labels=test_df[label_column].astype(str).tolist(),
    )


def run_text_classification_pilot(
    dataset: TextClassificationDataset,
    strategies: Sequence[AnnotationStrategy] | None = None,
    budgets: Sequence[int] = BUDGET_LEVELS,
    seed: int = 42,
    cost_per_sample: float = 0.10,
    cost_scenario: CostScenario | None = None,
    downstream_model: str = DOWNSTREAM_MODEL_TFIDF_LOGREG,
    sentence_transformer_model: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL,
) -> List[LearningCurve]:
    """Run budgeted text-classification experiments on a gold-labeled split."""
    if not dataset.train_texts or not dataset.test_texts:
        raise ValueError("Dataset must include non-empty train and test splits.")
    if len(dataset.train_texts) != len(dataset.train_labels):
        raise ValueError("Train texts and labels must have the same length.")
    if len(dataset.test_texts) != len(dataset.test_labels):
        raise ValueError("Test texts and labels must have the same length.")
    if not budgets or any(budget <= 0 for budget in budgets):
        raise ValueError("Budgets must be positive integers.")

    active_strategies = list(strategies or (
        AnnotationStrategy.RANDOM,
        AnnotationStrategy.UNCERTAINTY_AL,
        AnnotationStrategy.DIVERSITY_AL,
        AnnotationStrategy.HYBRID_AL,
    ))

    curves: List[LearningCurve] = []
    active_cost_scenario = cost_scenario or get_cost_scenarios(["base"])[0]
    for strategy in active_strategies:
        points: List[LearningCurvePoint] = []
        for budget in budgets:
            selected = select_budget_indices(
                dataset.train_texts,
                dataset.train_labels,
                budget=budget,
                strategy=strategy,
                seed=seed,
            )
            f1 = _fit_and_score(
                dataset,
                selected,
                seed=seed,
                downstream_model=downstream_model,
                sentence_transformer_model=sentence_transformer_model,
            )
            points.append(
                LearningCurvePoint(
                    budget=min(budget, len(dataset.train_texts)),
                    f1=f1,
                    cost_usd=estimate_annotation_cost(
                        strategy,
                        len(selected),
                        active_cost_scenario,
                    )[2],
                )
            )

        config = ExperimentConfig(
            strategy=strategy,
            task=NLPTask.CLASSIFICATION,
            budget=max(pt.budget for pt in points),
            model_type=downstream_model,
            dataset_name=dataset.name,
        )
        curves.append(LearningCurve(config=config, points=points))

    return curves


def run_text_classification_pilot_table(
    dataset: TextClassificationDataset,
    strategies: Sequence[AnnotationStrategy] | None = None,
    budgets: Sequence[int] = BUDGET_LEVELS,
    seeds: Sequence[int] = (42,),
    cost_per_sample: float = 0.10,
    cost_scenarios: Sequence[str] | None = None,
    downstream_model: str = DOWNSTREAM_MODEL_TFIDF_LOGREG,
    sentence_transformer_model: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL,
) -> pd.DataFrame:
    """Run pilot experiments and return one result row per condition."""
    rows: list[dict[str, object]] = []
    active_strategies = list(strategies or (
        AnnotationStrategy.RANDOM,
        AnnotationStrategy.UNCERTAINTY_AL,
        AnnotationStrategy.DIVERSITY_AL,
        AnnotationStrategy.HYBRID_AL,
    ))
    active_cost_scenarios = get_cost_scenarios(
        list(cost_scenarios) if cost_scenarios is not None else None
    )
    for seed in seeds:
        for strategy in active_strategies:
            for budget in budgets:
                selected = select_budget_indices(
                    dataset.train_texts,
                    dataset.train_labels,
                    budget=budget,
                    strategy=strategy,
                    seed=seed,
                )
                macro_f1, accuracy = _fit_and_score_metrics(
                    dataset,
                    selected,
                    seed=seed,
                    downstream_model=downstream_model,
                    sentence_transformer_model=sentence_transformer_model,
                )
                actual_budget = len(selected)
                for scenario in active_cost_scenarios:
                    annotation_cost, selection_cost, total_cost = estimate_annotation_cost(
                        strategy,
                        actual_budget,
                        scenario,
                    )
                    rows.append(
                        {
                            "dataset": dataset.name,
                            "strategy": strategy.value,
                            "budget": actual_budget,
                            "seed": seed,
                            "downstream_model": downstream_model,
                            "embedding_model": (
                                sentence_transformer_model
                                if downstream_model == DOWNSTREAM_MODEL_SENTENCE_TRANSFORMER_LOGREG
                                else ""
                            ),
                            "macro_f1": macro_f1,
                            "accuracy": accuracy,
                            "cost_scenario": scenario.name,
                            "cost_source": scenario.source_name,
                            "cost_source_url": scenario.source_url,
                            "cost_checked_at": scenario.checked_at,
                            "human_cost_per_label": scenario.human_cost_per_label,
                            "selection_cost_per_example": (
                                scenario.selection_cost_per_example.get(strategy, 0.0)
                            ),
                            "annotation_cost": annotation_cost,
                            "selection_cost": selection_cost,
                            "total_cost": total_cost,
                        }
                    )
    return pd.DataFrame(rows)


def select_budget_indices(
    texts: Sequence[str],
    labels: Sequence[str],
    budget: int,
    strategy: AnnotationStrategy,
    seed: int = 42,
) -> List[int]:
    """Select training indices for a simulated annotation budget."""
    if budget <= 0:
        return []

    n_samples = len(texts)
    capped_budget = min(budget, n_samples)
    if capped_budget == n_samples:
        return list(range(n_samples))

    rng = np.random.default_rng(seed)
    if strategy == AnnotationStrategy.RANDOM:
        return sorted(rng.choice(n_samples, size=capped_budget, replace=False).tolist())
    if strategy == AnnotationStrategy.DIVERSITY_AL:
        return _diversity_indices(texts, capped_budget, seed=seed)
    if strategy == AnnotationStrategy.UNCERTAINTY_AL:
        return _uncertainty_indices(texts, labels, capped_budget, seed=seed)
    if strategy == AnnotationStrategy.HYBRID_AL:
        seed_budget = max(1, capped_budget // 2)
        selected = _diversity_indices(texts, seed_budget, seed=seed)
        remaining_budget = capped_budget - len(selected)
        if remaining_budget <= 0:
            return selected
        uncertain = _uncertainty_indices(
            texts,
            labels,
            capped_budget,
            seed=seed,
            initial_indices=selected,
        )
        return uncertain[:capped_budget]

    return sorted(rng.choice(n_samples, size=capped_budget, replace=False).tolist())


def _fit_and_score(
    dataset: TextClassificationDataset,
    selected_indices: Sequence[int],
    seed: int,
    downstream_model: str = DOWNSTREAM_MODEL_TFIDF_LOGREG,
    sentence_transformer_model: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL,
) -> float:
    macro_f1, _ = _fit_and_score_metrics(
        dataset,
        selected_indices,
        seed,
        downstream_model=downstream_model,
        sentence_transformer_model=sentence_transformer_model,
    )
    return macro_f1


def _fit_and_score_metrics(
    dataset: TextClassificationDataset,
    selected_indices: Sequence[int],
    seed: int,
    downstream_model: str = DOWNSTREAM_MODEL_TFIDF_LOGREG,
    sentence_transformer_model: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL,
) -> tuple[float, float]:
    selected_texts = [dataset.train_texts[i] for i in selected_indices]
    selected_labels = [dataset.train_labels[i] for i in selected_indices]
    return fit_and_score_text_classifier(
        selected_texts,
        selected_labels,
        dataset.test_texts,
        dataset.test_labels,
        seed=seed,
        downstream_model=downstream_model,
        sentence_transformer_model=sentence_transformer_model,
    )


def fit_and_score_text_classifier(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    test_texts: Sequence[str],
    test_labels: Sequence[str],
    seed: int,
    downstream_model: str = DOWNSTREAM_MODEL_TFIDF_LOGREG,
    sentence_transformer_model: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL,
) -> tuple[float, float]:
    """Train the shared text classifier on supplied labels and score on gold test labels."""
    if not train_texts:
        raise ValueError("At least one training example is required.")
    if len(train_texts) != len(train_labels):
        raise ValueError("Training texts and labels must have the same length.")
    if len(test_texts) != len(test_labels):
        raise ValueError("Test texts and labels must have the same length.")

    if downstream_model not in SUPPORTED_DOWNSTREAM_MODELS:
        raise ValueError(
            f"Unsupported downstream model: {downstream_model}. "
            f"Choose one of {sorted(SUPPORTED_DOWNSTREAM_MODELS)}."
        )

    if len(set(train_labels)) < 2:
        model = DummyClassifier(strategy="most_frequent")
        model.fit(list(train_texts), list(train_labels))
        predictions = model.predict(list(test_texts))
    elif downstream_model == DOWNSTREAM_MODEL_TFIDF_LOGREG:
        predictions = _predict_tfidf_logreg(train_texts, train_labels, test_texts, seed)
    else:
        predictions = _predict_sentence_transformer_logreg(
            train_texts,
            train_labels,
            test_texts,
            seed,
            sentence_transformer_model,
        )

    return (
        float(f1_score(list(test_labels), predictions, average="macro")),
        float(accuracy_score(list(test_labels), predictions)),
    )


def _predict_tfidf_logreg(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    test_texts: Sequence[str],
    seed: int,
) -> np.ndarray:
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(min_df=1, ngram_range=(1, 2))),
            (
                "clf",
                LogisticRegression(max_iter=1000, random_state=seed),
            ),
        ]
    )
    model.fit(list(train_texts), list(train_labels))
    return model.predict(list(test_texts))


def _predict_sentence_transformer_logreg(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    test_texts: Sequence[str],
    seed: int,
    sentence_transformer_model: str,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "The sentence_transformer_logreg downstream model requires the optional "
            "`sentence-transformers` package. Install it before running this comparison."
        ) from exc

    encoder = SentenceTransformer(sentence_transformer_model)
    train_vectors = encoder.encode(list(train_texts), show_progress_bar=False)
    test_vectors = encoder.encode(list(test_texts), show_progress_bar=False)
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(train_vectors, list(train_labels))
    return model.predict(test_vectors)


def _diversity_indices(texts: Sequence[str], budget: int, seed: int) -> List[int]:
    vectors = TfidfVectorizer(min_df=1).fit_transform(texts)
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(len(texts)))]

    while len(selected) < budget:
        distances = cosine_distances(vectors, vectors[selected])
        min_distances = distances.min(axis=1)
        min_distances[selected] = -1.0
        jitter = rng.random(len(texts)) * 1e-12
        selected.append(int(np.argmax(min_distances + jitter)))

    return sorted(selected)


def _uncertainty_indices(
    texts: Sequence[str],
    labels: Sequence[str],
    budget: int,
    seed: int,
    initial_indices: Iterable[int] | None = None,
) -> List[int]:
    rng = np.random.default_rng(seed)
    selected = list(dict.fromkeys(initial_indices or []))
    seed_size = min(max(2, budget // 10), budget, len(texts))
    if len(selected) < seed_size:
        remaining_seed = [i for i in range(len(texts)) if i not in selected]
        selected.extend(
            rng.choice(
                remaining_seed,
                size=seed_size - len(selected),
                replace=False,
            ).tolist()
        )

    if len(selected) >= budget:
        return sorted(rng.choice(selected, size=budget, replace=False).tolist())

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(min_df=1, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
    train_texts = [texts[i] for i in selected]
    train_labels = [labels[i] for i in selected]
    if len(set(train_labels)) < 2:
        remaining = [i for i in range(len(texts)) if i not in selected]
        selected.extend(rng.choice(remaining, size=budget - len(selected), replace=False).tolist())
        return sorted(selected)

    model.fit(train_texts, train_labels)

    remaining = [i for i in range(len(texts)) if i not in selected]
    probabilities = model.predict_proba([texts[i] for i in remaining])
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1)
    tie_breakers = rng.random(len(remaining))
    ranked_remaining = [
        idx for _, _, idx in sorted(zip(entropy, tie_breakers, remaining), reverse=True)
    ]

    selected.extend(ranked_remaining[: budget - len(selected)])
    return sorted(selected)
