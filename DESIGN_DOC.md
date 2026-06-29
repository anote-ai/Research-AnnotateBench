# Research Design Document: AnnotateBench

## Current Pilot Scope

Target venue: JDSE 2026.

The current deliverable is a reproducible pilot for annotation-efficiency benchmarking, not the full benchmark grid. The pilot uses public datasets with existing gold labels and simulates annotation budgets by revealing selected labels from the training split.

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

## Pilot Outputs

- `results/pilot_results.csv`
- `results/pilot_results_trec.csv`
- `results/pilot_results_banking77.csv`
- `results/budget_recommendations.csv`
- `figures/learning_curve_*.png`
- `figures/pareto_*.png`
- `paper/main.tex`

## Long-Term Vision

Build AnnotateBench into a benchmark for deciding when LLM-generated annotations can be trusted, when they should be reviewed by humans, and where they fail silently. The longer-term benchmark should provide calibration curves, failure taxonomies, and reliability-aware cost guidance for LLM annotation pipelines.

## Long-Term Research Questions

1. When is LLM annotation reliable enough to replace human annotation?
2. When does high accuracy hide poor calibration?
3. What human review rate preserves quality while reducing annotation cost?
4. How does annotation quality degrade as label schemas become more complex?
5. How do LLM-LLM agreement and human-LLM agreement differ across tasks?

## Proposed Contributions

| Contribution | Description |
|---|---|
| AnnotateBench dataset | A benchmark covering multiple NLP task types and domains with human gold labels |
| LARI metric | LLM Annotation Reliability Index: calibration-adjusted F1 that penalizes overconfidence |
| Failure taxonomy | Error categories for diagnosing LLM annotation failures |
| Review-rate optimizer | A thresholding method for setting human review rates under cost and quality constraints |
| Cross-annotator agreement | Comparison of LLM-LLM agreement and human-LLM agreement |

## LARI Definition

```text
LARI = F1 * Calibration_Score

where:
  Calibration_Score = 1 - ECE
  ECE = sum_b (|B_b| / n) * |accuracy(B_b) - confidence(B_b)|
```

Interpretation:

- LARI = 1.0: perfect accuracy and perfect calibration.
- LARI < 0.5: unreliable without human review.
- LARI >= 0.8: candidate threshold for auto-accepting annotations.

## Full Benchmark Roadmap

Task type coverage:

| Task Type | Complexity | Example |
|---|---|---|
| Sentiment analysis | Low | Positive/negative product review |
| Topic classification | Medium | News categorization |
| Named entity recognition | Medium | Person/organization/location extraction |
| Relation extraction | High | Typed entity relation extraction |
| Claim verification | High | Supported/refuted/insufficient evidence |
| Coreference resolution | Very high | Span linking |
| Instruction-following evaluation | High | Rubric-based response quality scoring |
| Medical coding | Expert-level | ICD-style code assignment |

Domain coverage should include general web text, medical literature, legal documents, financial reports, scientific papers, and social media.

## Full Experimental Plan

### Experiment 0: Baseline Reliability

Run a frontier LLM on a simple sentiment annotation task. Compute accuracy, macro F1, ECE, and LARI to establish a high-reliability baseline.

### Experiment 1: LARI by Task Type

Measure LARI across task types and models. The expected pattern is that reliability decreases as task complexity increases, with expert-level and long-context tasks requiring more human review.

### Experiment 2: Failure Taxonomy

Sample LLM annotation errors and code them into categories such as label ambiguity, domain knowledge gaps, context failures, schema confusion, and overconfident hallucination.

### Experiment 3: Calibration vs. Accuracy

Compare zero-shot and few-shot prompting to test whether prompting improves calibration, accuracy, or both.

### Experiment 4: Optimal Review Rate

Sweep human-review thresholds and measure the tradeoff between downstream model quality and annotation cost.

### Experiment 5: Cross-Model Agreement

Compare LLM-LLM agreement with human-LLM agreement to test whether agreement among models reflects true label quality or shared model bias.

## Deferred Work

- LLM annotator or simulated LLM annotator baseline.
- More seeds and confidence intervals in the paper tables.
- Externally calibrated annotation and LLM cost model.
- Reliability diagnostics such as ECE and LARI.
- Human annotation protocol for the larger benchmark.
- Review-rate optimizer implementation.
