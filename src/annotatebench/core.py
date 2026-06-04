"""Core data structures and logic for AnnotateBench."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias


class AnnotationStrategy(str, Enum):
    """Annotation strategies evaluated in the benchmark."""

    RANDOM = "random"
    UNCERTAINTY_AL = "uncertainty_al"
    DIVERSITY_AL = "diversity_al"
    HYBRID_AL = "hybrid_al"
    LLM_ANNOTATOR = "llm_annotator"


class NLPTask(str, Enum):
    """NLP tasks in the benchmark."""

    CLASSIFICATION = "classification"
    NER = "ner"
    QA = "qa"
    SUMMARIZATION = "summarization"
    INSTRUCTION_TUNING = "instruction_tuning"


BudgetLevel: TypeAlias = int

BUDGET_LEVELS: list[BudgetLevel] = [50, 100, 250, 500, 1000]


@dataclass
class ExperimentConfig:
    """Configuration for a single benchmark experiment."""

    strategy: AnnotationStrategy
    task: NLPTask
    budget: int
    model_type: str
    dataset_name: str


@dataclass
class LearningCurvePoint:
    """A single point on a learning curve."""

    budget: int
    f1: float
    cost_usd: float


@dataclass
class LearningCurve:
    """A full learning curve for one experimental configuration."""

    config: ExperimentConfig
    points: list[LearningCurvePoint] = field(default_factory=list)


def fit_learning_curve(
    budgets: list[int], f1_scores: list[float]
) -> tuple[float, float]:
    """Fit a power-law learning curve y = a * x^b.

    Uses log-linear regression: log(y) = log(a) + b * log(x).
    Returns (a, b).
    """
    if len(budgets) != len(f1_scores):
        raise ValueError("budgets and f1_scores must have the same length.")
    if len(budgets) < 2:  # noqa: PLR2004
        raise ValueError("At least 2 data points required.")

    log_x = [math.log(b) for b in budgets]
    log_y = [math.log(max(f, 1e-12)) for f in f1_scores]

    n = len(log_x)
    mean_lx = sum(log_x) / n
    mean_ly = sum(log_y) / n

    num = sum((lx - mean_lx) * (ly - mean_ly) for lx, ly in zip(log_x, log_y))
    den = sum((lx - mean_lx) ** 2 for lx in log_x)

    b = num / den if den != 0 else 0.0
    log_a = mean_ly - b * mean_lx
    a = math.exp(log_a)
    return (a, b)
