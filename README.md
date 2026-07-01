# research-annotatebench

**Motivation:** Annotation budget is a primary constraint in NLP system development. This benchmark systematically compares annotation strategies across tasks and budgets using learning curves.

## Current Status

This repository is not yet the full LLM annotation reliability benchmark described in
`DESIGN_DOC.md`. The current code supports:

- illustrative synthetic learning curves for exercising cost and Pareto utilities;
- a real public-dataset text-classification pilot that simulates annotation by revealing gold
  labels under budgeted selection strategies;
- small ECE/LARI metric primitives and failure-taxonomy data structures for the future
  reliability track.

It does not yet contain real LLM annotation runs with confidence scores, human adjudication,
failure-taxonomy coding, or a review-rate optimizer. See `BLOG.md`, `PAPER_DRAFT.md`, and
`results/README.md` for the claim boundary.

## Benchmark Grid

The synthetic demo covers 5 strategies × 5 tasks × 5 budget levels = 25 learning curves, 125 data points. The real pilot path currently supports text classification train/test CSVs with gold labels.

| Strategy         | 50   | 100  | 250  | 500  | 1000 |
|------------------|------|------|------|------|------|
| Random           | 0.42 | 0.52 | 0.62 | 0.68 | 0.72 |
| Uncertainty AL   | 0.50 | 0.61 | 0.72 | 0.79 | 0.84 |
| Diversity AL     | 0.48 | 0.59 | 0.70 | 0.77 | 0.82 |
| Hybrid AL        | 0.52 | 0.63 | 0.74 | 0.81 | 0.86 |
| LLM Annotator    | 0.65 | 0.72 | 0.78 | 0.82 | 0.84 |

**Tasks:** Classification, NER, QA, Summarization, Instruction Tuning

## Learning Curve Methodology

F1 scores are modeled as power laws: `F1 = a * budget^b`, fit via log-linear regression. This captures diminishing returns as budget increases.

## Pareto Analysis

For each strategy, we plot cost vs. best F1 and compute the Pareto frontier — strategies that cannot be strictly dominated in both dimensions. This guides practitioners to cost-efficient annotation choices.

Synthetic demo finding: LLM Annotator dominates at low budgets (<100 samples); Hybrid AL leads at high budgets. This is an illustrative reference-curve pattern, not a measured empirical finding.

## Synthetic Demo Usage

```python
from annotatebench import AnnotationStrategy, NLPTask, make_learning_curve

curve = make_learning_curve(AnnotationStrategy.HYBRID_AL, NLPTask.CLASSIFICATION)
best = curve.best_point()

print(best.budget, best.f1)
```

Run a real text-classification pilot from CSV files:

```bash
./scripts/run_pilot.py \
  --train-csv data/financial_phrasebank_train.csv \
  --test-csv data/financial_phrasebank_test.csv \
  --text-column text \
  --label-column label \
  --dataset-name financial_phrasebank \
  --budgets 50,100,250,500,1000
```

The pilot trains TF-IDF + logistic-regression classifiers at each budget and compares random sampling, uncertainty active learning, diversity active learning, and hybrid active learning using the same learning-curve and Pareto utilities as the synthetic demo.

Run the Financial PhraseBank pilot from a local public PhraseBank file:

```bash
./scripts/run_pilot.py \
  --dataset financial_phrasebank \
  --financial-phrasebank-path data/raw/FinancialPhraseBank-v1.0.zip \
  --budgets 50,100,250,500,1000 \
  --seeds 0,1,2 \
  --cost-scenarios low,base,high \
  --output-csv results/pilot_results.csv
./scripts/make_figures.py --results results/pilot_results.csv --cost-scenario base
```

The Financial PhraseBank loader accepts CSV files with `text`/`label` or `sentence`/`label` columns, the original `sentence@label` text files, or a ZIP containing the original PhraseBank sentence files. The first paper draft is in `paper/main.tex`.

Run budget recommendations across available pilot CSVs:

```bash
./scripts/make_budget_recommendations.py \
  --results results/pilot_results.csv results/pilot_results_trec.csv results/pilot_results_banking77.csv \
  --output-csv results/budget_recommendations.csv \
  --cost-scenario base
```

## Venue

Rough **JDSE 2026** draft materials are in `paper/main.tex`; the project is not submission-ready
for the full `DESIGN_DOC.md` reliability claims.

## Citation

```bibtex
@article{anote2026annotatebench,
  title   = {AnnotateBench: A Learning Curve Benchmark for Annotation Strategy Selection},
  author  = {Anote AI},
  journal = {JDSE},
  year    = {2026},
}
```
