from __future__ import annotations

import random
from typing import Dict, List

from .core import (
    AnnotationStrategy,
    NLPTask,
    BUDGET_LEVELS,
    ExperimentConfig,
    LearningCurvePoint,
    LearningCurve,
)

# Illustrative reference F1 values at BUDGET_LEVELS = [50, 100, 250, 500, 1000].
# These are hand-specified curves for exercising the learning-curve utilities,
# not measured results from real LLM or human annotation runs.
STRATEGY_F1_CURVES: Dict[str, List[float]] = {
    AnnotationStrategy.RANDOM.value:         [0.42, 0.52, 0.62, 0.68, 0.72],
    AnnotationStrategy.UNCERTAINTY_AL.value: [0.50, 0.61, 0.72, 0.79, 0.84],
    AnnotationStrategy.DIVERSITY_AL.value:   [0.48, 0.59, 0.70, 0.77, 0.82],
    AnnotationStrategy.HYBRID_AL.value:      [0.52, 0.63, 0.74, 0.81, 0.86],
    AnnotationStrategy.LLM_ANNOTATOR.value:  [0.65, 0.72, 0.78, 0.82, 0.84],
}

# NER F1 curves: token-level annotation is more expensive; active learning
# helps more because entity mentions are rare and skewed.
_NER_F1_CURVES: Dict[str, List[float]] = {
    AnnotationStrategy.RANDOM.value:         [0.35, 0.44, 0.55, 0.62, 0.67],
    AnnotationStrategy.UNCERTAINTY_AL.value: [0.44, 0.56, 0.68, 0.76, 0.82],
    AnnotationStrategy.DIVERSITY_AL.value:   [0.41, 0.53, 0.65, 0.73, 0.79],
    AnnotationStrategy.HYBRID_AL.value:      [0.46, 0.58, 0.70, 0.78, 0.84],
    AnnotationStrategy.LLM_ANNOTATOR.value:  [0.58, 0.66, 0.73, 0.77, 0.80],
}

# QA (extractive/short-answer) reference curves: illustrative, not measured.
_QA_F1_CURVES: Dict[str, List[float]] = {
    AnnotationStrategy.RANDOM.value:         [0.38, 0.48, 0.59, 0.66, 0.71],
    AnnotationStrategy.UNCERTAINTY_AL.value: [0.47, 0.59, 0.71, 0.78, 0.83],
    AnnotationStrategy.DIVERSITY_AL.value:   [0.44, 0.56, 0.68, 0.75, 0.80],
    AnnotationStrategy.HYBRID_AL.value:      [0.49, 0.61, 0.73, 0.80, 0.85],
    AnnotationStrategy.LLM_ANNOTATOR.value:  [0.68, 0.75, 0.81, 0.85, 0.87],
}

# Summarization reference curves: illustrative, not measured. The lower
# plateau reflects high-variance, subjective summary quality judgments.
_SUMMARIZATION_F1_CURVES: Dict[str, List[float]] = {
    AnnotationStrategy.RANDOM.value:         [0.30, 0.38, 0.47, 0.53, 0.58],
    AnnotationStrategy.UNCERTAINTY_AL.value: [0.36, 0.46, 0.56, 0.63, 0.68],
    AnnotationStrategy.DIVERSITY_AL.value:   [0.35, 0.45, 0.55, 0.62, 0.67],
    AnnotationStrategy.HYBRID_AL.value:      [0.38, 0.48, 0.58, 0.65, 0.70],
    AnnotationStrategy.LLM_ANNOTATOR.value:  [0.52, 0.59, 0.65, 0.69, 0.71],
}

# Instruction-tuning rubric-score reference curves: illustrative, not measured.
_INSTRUCTION_TUNING_F1_CURVES: Dict[str, List[float]] = {
    AnnotationStrategy.RANDOM.value:         [0.33, 0.42, 0.52, 0.59, 0.64],
    AnnotationStrategy.UNCERTAINTY_AL.value: [0.41, 0.52, 0.63, 0.71, 0.77],
    AnnotationStrategy.DIVERSITY_AL.value:   [0.39, 0.50, 0.61, 0.69, 0.75],
    AnnotationStrategy.HYBRID_AL.value:      [0.43, 0.54, 0.65, 0.73, 0.79],
    AnnotationStrategy.LLM_ANNOTATOR.value:  [0.55, 0.63, 0.70, 0.74, 0.76],
}

# Cost per annotation batch (USD) at each budget level
_COST_PER_BUDGET: Dict[str, List[float]] = {
    AnnotationStrategy.RANDOM.value:         [5.0, 10.0, 25.0, 50.0, 100.0],
    AnnotationStrategy.UNCERTAINTY_AL.value: [6.0, 12.0, 28.0, 55.0, 108.0],
    AnnotationStrategy.DIVERSITY_AL.value:   [6.5, 13.0, 30.0, 58.0, 112.0],
    AnnotationStrategy.HYBRID_AL.value:      [7.0, 14.0, 32.0, 62.0, 118.0],
    AnnotationStrategy.LLM_ANNOTATOR.value:  [15.0, 28.0, 60.0, 110.0, 200.0],
}

# NER has higher per-sample cost (token labelling overhead ~2x)
_NER_COST_PER_BUDGET: Dict[str, List[float]] = {
    AnnotationStrategy.RANDOM.value:         [10.0, 20.0, 50.0, 100.0, 200.0],
    AnnotationStrategy.UNCERTAINTY_AL.value: [12.0, 24.0, 56.0, 110.0, 216.0],
    AnnotationStrategy.DIVERSITY_AL.value:   [13.0, 26.0, 60.0, 116.0, 224.0],
    AnnotationStrategy.HYBRID_AL.value:      [14.0, 28.0, 64.0, 124.0, 236.0],
    AnnotationStrategy.LLM_ANNOTATOR.value:  [20.0, 38.0, 80.0, 145.0, 260.0],
}

_QA_COST_PER_BUDGET: Dict[str, List[float]] = {
    AnnotationStrategy.RANDOM.value:         [6.5, 13.0, 32.5, 65.0, 130.0],
    AnnotationStrategy.UNCERTAINTY_AL.value: [7.8, 15.6, 36.4, 71.5, 140.4],
    AnnotationStrategy.DIVERSITY_AL.value:   [8.45, 16.9, 39.0, 75.4, 145.6],
    AnnotationStrategy.HYBRID_AL.value:      [9.1, 18.2, 41.6, 80.6, 153.4],
    AnnotationStrategy.LLM_ANNOTATOR.value:  [19.5, 36.4, 78.0, 143.0, 260.0],
}

_SUMMARIZATION_COST_PER_BUDGET: Dict[str, List[float]] = {
    AnnotationStrategy.RANDOM.value:         [15.0, 30.0, 75.0, 150.0, 300.0],
    AnnotationStrategy.UNCERTAINTY_AL.value: [18.0, 36.0, 84.0, 165.0, 324.0],
    AnnotationStrategy.DIVERSITY_AL.value:   [19.5, 39.0, 90.0, 174.0, 336.0],
    AnnotationStrategy.HYBRID_AL.value:      [21.0, 42.0, 96.0, 186.0, 354.0],
    AnnotationStrategy.LLM_ANNOTATOR.value:  [45.0, 84.0, 180.0, 330.0, 600.0],
}

_INSTRUCTION_TUNING_COST_PER_BUDGET: Dict[str, List[float]] = {
    AnnotationStrategy.RANDOM.value:         [10.0, 20.0, 50.0, 100.0, 200.0],
    AnnotationStrategy.UNCERTAINTY_AL.value: [12.0, 24.0, 56.0, 110.0, 216.0],
    AnnotationStrategy.DIVERSITY_AL.value:   [13.0, 26.0, 60.0, 116.0, 224.0],
    AnnotationStrategy.HYBRID_AL.value:      [14.0, 28.0, 64.0, 124.0, 236.0],
    AnnotationStrategy.LLM_ANNOTATOR.value:  [30.0, 56.0, 120.0, 220.0, 400.0],
}

_TASK_F1_CURVES: Dict[str, Dict[str, List[float]]] = {
    NLPTask.NER.value: _NER_F1_CURVES,
    NLPTask.QA.value: _QA_F1_CURVES,
    NLPTask.SUMMARIZATION.value: _SUMMARIZATION_F1_CURVES,
    NLPTask.INSTRUCTION_TUNING.value: _INSTRUCTION_TUNING_F1_CURVES,
}

_TASK_COST_CURVES: Dict[str, Dict[str, List[float]]] = {
    NLPTask.NER.value: _NER_COST_PER_BUDGET,
    NLPTask.QA.value: _QA_COST_PER_BUDGET,
    NLPTask.SUMMARIZATION.value: _SUMMARIZATION_COST_PER_BUDGET,
    NLPTask.INSTRUCTION_TUNING.value: _INSTRUCTION_TUNING_COST_PER_BUDGET,
}


def make_learning_curve(
    strategy: AnnotationStrategy,
    task: NLPTask,
    model_type: str = "bert-base",
    noise: float = 0.02,
    seed: int = 42,
) -> LearningCurve:
    """Create an illustrative LearningCurve with Gaussian noise on F1 scores."""
    rng = random.Random(seed)

    if task.value in _TASK_F1_CURVES:
        base_f1s = _TASK_F1_CURVES[task.value][strategy.value]
        base_costs = _TASK_COST_CURVES[task.value][strategy.value]
    else:
        base_f1s = STRATEGY_F1_CURVES[strategy.value]
        base_costs = _COST_PER_BUDGET[strategy.value]

    config = ExperimentConfig(
        strategy=strategy,
        task=task,
        budget=BUDGET_LEVELS[-1],
        model_type=model_type,
        dataset_name=f"{task.value}-dataset",
    )

    points: list[LearningCurvePoint] = []
    for budget, f1_base, cost in zip(BUDGET_LEVELS, base_f1s, base_costs):
        # Add Gaussian noise and clamp to [0, 1]
        noisy_f1 = f1_base + rng.gauss(0, noise)
        noisy_f1 = max(0.0, min(1.0, noisy_f1))
        points.append(LearningCurvePoint(budget=budget, f1=noisy_f1, cost_usd=cost))

    return LearningCurve(config=config, points=points)


def make_full_dataset(seed: int = 42) -> List[LearningCurve]:
    """Generate 5 strategies x 5 tasks = 25 learning curves."""
    rng = random.Random(seed)
    curves: List[LearningCurve] = []
    for strategy in AnnotationStrategy:
        for task in NLPTask:
            curve_seed = rng.randint(0, 99999)
            curves.append(make_learning_curve(strategy=strategy, task=task, seed=curve_seed))
    return curves
