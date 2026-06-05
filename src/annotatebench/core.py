from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


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

    def best_point(self) -> Optional[LearningCurvePoint]:
        """Return the point with the highest F1 score.

        If multiple points share the same F1, the lower-budget point is
        preferred because it reaches the same quality with fewer annotations.
        """
        if not self.points:
            return None
        return max(self.points, key=lambda pt: (pt.f1, -pt.budget))

    def total_cost(self) -> float:
        return sum(pt.cost_usd for pt in self.points)


@dataclass
class CostModel:
    """Model of API vs human annotation costs.

    Attributes:
        api_cost_per_sample: Cost in USD per sample when using an LLM API.
        human_cost_per_sample: Cost in USD per sample for human annotation.
        api_error_rate: Fraction of API annotations that are incorrect (0-1).
        human_error_rate: Fraction of human annotations that are incorrect (0-1).
        review_cost_per_sample: Optional cost to review a single annotation.
    """

    api_cost_per_sample: float
    human_cost_per_sample: float
    api_error_rate: float = 0.05
    human_error_rate: float = 0.02
    review_cost_per_sample: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.api_error_rate <= 1.0):
            raise ValueError("api_error_rate must be in [0, 1].")
        if not (0.0 <= self.human_error_rate <= 1.0):
            raise ValueError("human_error_rate must be in [0, 1].")

    def total_api_cost(self, n_samples: int) -> float:
        """Total API annotation cost for n_samples, including review."""
        base = self.api_cost_per_sample * n_samples
        review = self.review_cost_per_sample * round(self.api_error_rate * n_samples)
        return base + review

    def total_human_cost(self, n_samples: int) -> float:
        """Total human annotation cost for n_samples."""
        return self.human_cost_per_sample * n_samples

    def breakeven_samples(self) -> Optional[float]:
        """Number of samples at which API cost equals human cost.

        Returns None when costs are equal at all scales (parallel lines).
        """
        # total_api = api_cost_per_sample * n + review_cost_per_sample * api_error_rate * n
        # total_human = human_cost_per_sample * n
        effective_api = self.api_cost_per_sample + self.review_cost_per_sample * self.api_error_rate
        if abs(effective_api - self.human_cost_per_sample) < 1e-12:
            return None
        # They are proportional; breakeven is always at 0 if lines are different slopes
        # Return None — costs scale linearly so cheaper option stays cheaper throughout.
        return None

    def cheaper_option(self, n_samples: int) -> str:
        """Return 'api' or 'human' for the cheaper option at n_samples."""
        return "api" if self.total_api_cost(n_samples) <= self.total_human_cost(n_samples) else "human"

    def quality_adjusted_cost(self, n_samples: int, use_api: bool = True) -> float:
        """Cost per correctly-labelled sample."""
        if use_api:
            error_rate = self.api_error_rate
            total = self.total_api_cost(n_samples)
        else:
            error_rate = self.human_error_rate
            total = self.total_human_cost(n_samples)
        correct = n_samples * (1 - error_rate)
        return total / max(correct, 1e-9)


@dataclass
class UncertaintySamplingMetrics:
    """Metrics characterising an uncertainty-based active learning run.

    Attributes:
        entropy_scores: Per-sample entropy values from the model.
        selected_mask: Boolean mask indicating which samples were selected.
        oracle_labels: True labels for all samples (used to measure informativeness).
    """

    entropy_scores: List[float]
    selected_mask: List[bool]
    oracle_labels: List[int]

    def __post_init__(self) -> None:
        n = len(self.entropy_scores)
        if len(self.selected_mask) != n or len(self.oracle_labels) != n:
            raise ValueError("All lists must have the same length.")

    def mean_entropy_selected(self) -> float:
        """Average entropy among selected samples."""
        selected = [s for s, m in zip(self.entropy_scores, self.selected_mask) if m]
        return sum(selected) / max(len(selected), 1)

    def mean_entropy_unselected(self) -> float:
        """Average entropy among unselected samples."""
        unselected = [s for s, m in zip(self.entropy_scores, self.selected_mask) if not m]
        return sum(unselected) / max(len(unselected), 1)

    def informativeness_gain(self) -> float:
        """Difference in mean entropy: selected - unselected.

        Positive values indicate uncertainty sampling is selecting harder samples.
        """
        return self.mean_entropy_selected() - self.mean_entropy_unselected()

    def class_balance_selected(self) -> Dict[int, float]:
        """Fraction of each class label among selected samples."""
        selected_labels = [l for l, m in zip(self.oracle_labels, self.selected_mask) if m]
        total = max(len(selected_labels), 1)
        classes = set(self.oracle_labels)
        return {c: selected_labels.count(c) / total for c in sorted(classes)}

    def selection_rate(self) -> float:
        """Fraction of samples selected."""
        return sum(self.selected_mask) / max(len(self.selected_mask), 1)


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
