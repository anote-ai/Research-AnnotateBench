# AnnotateBench Design Doc

## Current Two-Day Scope

Target venue: JDSE 2026.

The immediate deliverable is a reproducible pilot for annotation-efficiency benchmarking, not the full benchmark grid. The pilot uses public datasets with existing gold labels and simulates annotation budgets by revealing selected labels from the training split.

Priority order:

1. Financial PhraseBank, TREC, and Banking77 real-data pilots.
2. Learning-curve and Pareto frontier figures for each dataset.
3. Rough LaTeX paper draft.
4. Design doc updates that keep scope aligned.
5. Passing tests.

PyPI publishing is out of scope unless explicitly requested.

## Pilot Design

Datasets:

- Financial PhraseBank sentiment classification with negative, neutral, and positive labels.
- TREC coarse question classification.
- Banking77 intent classification.

Strategies:

- Random sampling.
- Uncertainty active learning.
- Diversity active learning.
- Hybrid active learning.

Budgets: 50, 100, 250, 500, and 1000 labeled examples, capped at the available training set size.

Model: TF-IDF features with logistic regression.

Primary metric: macro F1.

Secondary metric: accuracy.

Cost assumptions: low/base/high scenario analysis with human-label costs of $0.03, $0.10, and $0.30 per labeled example, plus small illustrative active-learning selection overheads.

## Outputs

- `results/pilot_results.csv`
- `results/pilot_results_trec.csv`
- `results/pilot_results_banking77.csv`
- `results/budget_recommendations.csv`
- `figures/learning_curve_*.png`
- `figures/pareto_*.png`
- `paper/main.tex`

## Deferred Work

- LLM annotator or simulated LLM annotator baseline.
- More seeds and confidence intervals in the paper tables.
- Externally calibrated annotation and LLM cost model.
- Reliability diagnostics such as ECE and LARI.
