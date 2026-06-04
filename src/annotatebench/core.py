from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class AnnotationStrategy(str, Enum):
    RANDOM = "random"
    UNCERTAINTY_AL = "uncertainty_al"
    DIVERSITY_AL = "diversity_al"
    HYBRID_AL = "hybrid_al"
    LLM_ANNOTATOR = "llm_annotator"


class NLPTask(str, Enum):
    CLASSIFICATION = "classification"
    NER = "ner"
    QA = "qa"
    SUMMARIZATION = "summarization"
    INSTRUCTION_TUNING = "instruction_tuning"


BUDGET_LEVELS: List[int] = [50, 100, 250, 500, 1000]


@dataclass
class ExperimentConfig:
    strategy: AnnotationStrategy
    task: NLPTask
    budget: int
    model_type: str
    dataset_name: str


@dataclass
class LearningCurvePoint:
    budget: int
    f1: float
    cost_usd: float

    @property
    def efficiency(self) -> float:
        return self.f1 / max(self.cost_usd, 1e-9)


@dataclass
class LearningCurve:
    config: ExperimentConfig
    points: List[LearningCurvePoint]

    def f1_at_budget(self, b: int) -> Optional[float]:
        for pt in self.points:
            if pt.budget == b:
                return pt.f1
        return None

    def max_f1(self) -> float:
        if not self.points:
            return 0.0
        return max(pt.f1 for pt in self.points)

    def total_cost(self) -> float:
        return sum(pt.cost_usd for pt in self.points)


def fit_learning_curve(
    budgets: List[int], f1_scores: List[float]
) -> Tuple[float, float]:
    """Fit y = a * x^b via log-linear regression.

    Returns (a, b).
    """
    if len(budgets) < 2:
        raise ValueError("Need at least 2 points.")

    log_x = [math.log(b) for b in budgets]
    log_y = [math.log(max(f, 1e-9)) for f in f1_scores]

    n = len(log_x)
    mean_lx = sum(log_x) / n
    mean_ly = sum(log_y) / n

    ss_xy = sum((x - mean_lx) * (y - mean_ly) for x, y in zip(log_x, log_y))
    ss_xx = sum((x - mean_lx) ** 2 for x in log_x)

    if ss_xx == 0:
        b_coef = 0.0
    else:
        b_coef = ss_xy / ss_xx

    log_a = mean_ly - b_coef * mean_lx
    a_coef = math.exp(log_a)

    return a_coef, b_coef


def predict_f1(a: float, b: float, budget: int) -> float:
    """Predict F1 using power law: a * budget^b."""
    return a * (budget ** b)
