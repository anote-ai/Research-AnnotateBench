# research-annotatebench

**AnnotateBench — How Much Labeled Data Do Annotation Strategies Need Across NLP Tasks?**

## Motivation

Choosing an annotation strategy is a high-stakes decision: active learning, LLM-assisted
annotation, and random sampling differ widely in cost-efficiency across NLP tasks and budget
levels. AnnotateBench is the first systematic benchmark that measures **5 annotation strategies
across 10 datasets, 5 budget levels, and 3 model types**, producing fitted learning curves and
cost-accuracy Pareto frontiers to guide practitioners.

## Benchmark Grid

| Strategy         | Classification | NER | QA  | Summarization | Instruction Tuning |
|------------------|:--------------:|:---:|:---:|:-------------:|:------------------:|
| Random           | 5 budgets      | 5   | 5   | 5             | 5                  |
| Uncertainty AL   | 5 budgets      | 5   | 5   | 5             | 5                  |
| Diversity AL     | 5 budgets      | 5   | 5   | 5             | 5                  |
| Hybrid AL        | 5 budgets      | 5   | 5   | 5             | 5                  |
| LLM Annotator    | 5 budgets      | 5   | 5   | 5             | 5                  |

Budget levels: **50 / 100 / 250 / 500 / 1000** labeled examples.

## Learning Curve Methodology

We fit a power-law model to each strategy's performance:

```
F1 = a * budget^b
```

Parameters (a, b) are estimated via log-linear regression. The area under the learning curve
(ALC) summarizes overall sample efficiency, and the Pareto frontier across (cost, F1) pairs
identifies strategies that are not dominated at any budget.

## Pareto Frontier

For each task, we compute the cost-F1 Pareto frontier across all strategies and budget levels.
A strategy appears on the frontier if no other strategy achieves the same or better F1 at
equal or lower annotation cost.

```
F1
1.0 |                                    * LLM_ANNOTATOR
    |                          * Hybrid AL
0.8 |               * Diversity AL
    |    * Random
0.6 |__________________________________
    0   $10   $50  $200  $1000     Cost
```

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```python
from annotatebench.core import (
    AnnotationStrategy, NLPTask, ExperimentConfig,
    LearningCurve, LearningCurvePoint, fit_learning_curve, BUDGET_LEVELS,
)
from annotatebench.evaluate import pareto_frontier, area_under_learning_curve, strategy_comparison

# Fit a learning curve
budgets = BUDGET_LEVELS
f1_scores = [0.52, 0.61, 0.70, 0.76, 0.81]
a, b = fit_learning_curve(budgets, f1_scores)
print(f"Power law: F1 = {a:.3f} * budget^{b:.3f}")

# Compute Pareto frontier
points = [(cost, f1) for cost, f1 in zip([5, 10, 25, 50, 100], f1_scores)]
frontier = pareto_frontier(points)
print("Pareto-optimal points:", frontier)
```

## Citation

```bibtex
@misc{anote2024annotatebench,
  title  = {AnnotateBench: How Much Labeled Data Do Annotation Strategies Need Across NLP Tasks?},
  author = {Anote AI Research},
  year   = {2024},
  url    = {https://github.com/anote-ai/research-annotatebench},
}
```
