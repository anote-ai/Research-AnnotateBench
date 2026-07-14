# AnnotateBench Paper Draft Skeleton

Status: ten-dataset classification benchmark draft; seed-0 ten-dataset LLM annotator strategy
results added; full LLM reliability benchmark is not complete.

This skeleton maps the current repository to the larger research plan in `DESIGN_DOC.md`.
It deliberately separates implemented code, measured pilot outputs, illustrative demo curves,
and not-yet-implemented LLM annotation reliability experiments.

## Current Claim Boundary

AnnotateBench currently supports a learning-curve and cost-frontier framework for annotation
strategy selection. The current branch includes a real ten-dataset public text-classification
benchmark that simulates annotation by revealing existing gold labels.

The repository now contains an API-backed LLM annotator runner and a seed-0 ten-dataset
`llm_annotator` result grid. It does not yet contain a completed LLM annotation reliability
benchmark, human adjudication, calibrated review policies, or a coded failure-taxonomy study.

## Implemented

- Core learning-curve data structures and utilities in `src/annotatebench/core.py`.
- Cost and Pareto analysis utilities in `src/annotatebench/evaluate.py` and
  `src/annotatebench/costs.py`.
- Synthetic illustrative reference curves in `src/annotatebench/data.py`, including distinct
  QA, summarization, and instruction-tuning curves. These are not measured results.
- Real text-classification pilot machinery in `src/annotatebench/pilot.py`,
  `src/annotatebench/datasets.py`, `scripts/run_pilot.py`, and `scripts/run_benchmark.py`.
- Ten-dataset benchmark outputs in `results/benchmark_results.csv`,
  `results/budget_recommendations.csv`, `results/benchmark_best_*.csv`,
  `results/paper_core_summary.csv`, and `figures/*.png`.
- Seed-0 ten-dataset LLM annotator strategy outputs in
  `results/benchmark_results_with_llm_seed0_all_datasets.csv` and
  `results/llm_strategy_seed0_summary.csv`.
- Benchmark validation in `scripts/validate_benchmark_results.py`.
- Paper summary generation in `scripts/make_paper_summary_table.py`.
- Optional downstream-model comparison support for `sentence_transformer_logreg`; this is
  implemented as a runner option but is not yet a measured result.
- LLM annotator scripts in `scripts/run_llm_annotation.py` and
  `scripts/run_llm_strategy_benchmark.py`.
- ECE/LARI metric primitives in `src/annotatebench/metrics/lari.py`.
- Failure taxonomy structures in `src/annotatebench/taxonomy/failure_coder.py`.

## Partially Implemented

- LARI exists as a diagnostic metric and is populated for the seed-0 ten-dataset LLM annotator
  strategy grid and small row-level LLM annotation runs.
- The failure taxonomy exists as code-level categories and records, but no sample of real
  annotation failures has been coded by human reviewers.
- The paper draft in `paper/main.tex` reports the classification strategy benchmark, not the full
  reliability benchmark described in `DESIGN_DOC.md`.
- Stronger downstream learner support exists, but the sentence-transformer comparison still needs
  to be run and summarized before it can be claimed as evidence.

## Not Yet Implemented

- 10,000-task AnnotateBench reliability dataset across the eight task types in `DESIGN_DOC.md`.
- Multi-seed ten-dataset LLM annotator strategy run with versioned prompts, model names, measured
  costs, and confidence scores.
- Human annotation or adjudication protocol for gold labels.
- LARI heatmap by model and task type.
- Reliability diagrams and calibration study across prompting conditions.
- Failure-taxonomy coding study with inter-rater agreement.
- Review-rate optimizer that chooses human review thresholds from LARI/confidence features.
- Cross-model agreement analysis comparing LLM-LLM and human-LLM agreement.

## Paper Structure

1. Introduction: annotation strategy selection and LLM annotation reliability are related but
   separate problems; state which one this paper currently measures.
2. Related work: active learning, data-centric AI benchmarks, LLM annotation, calibration, and
   human-in-the-loop review.
3. Method: learning-curve benchmark framework, pilot selection strategies, cost scenarios, and
   LARI/ECE definitions.
4. Experiments: ten public-dataset text-classification results, plus the seed-0 LLM annotator
   strategy extension.
5. Reliability extension: row-level LLM annotation pilots, clearly separated from the full future
   reliability benchmark.
6. Limitations: gold-label reveal simulation, no completed LLM reliability run, no human failure
   coding yet, and no measured structured/generative task results yet.

## Next Required Evidence

Before making the `DESIGN_DOC.md` claims, the project needs at least one small measured LLM
annotation reliability pilot. A minimal pilot should produce rows with:

- dataset and task type;
- model and prompt version;
- gold label;
- LLM label;
- LLM confidence;
- correctness;
- F1, ECE, and LARI aggregates;
- optional failure category for incorrect examples.

## Draft Results Text

The seed-0 LLM annotator extension evaluates `gpt-4o-mini` as a budgeted labeling strategy on all
ten text-classification datasets. For budgets 50, 100, and 250, the runner selects training
examples with the random budget policy, asks the model for labels and confidence scores, trains the
same TF-IDF logistic-regression classifier used in the gold-label benchmark, and evaluates on the
gold test split. This makes the LLM results comparable to the gold-label strategies at the same
dataset, budget, and seed, while keeping the result scoped to one model and one seed.

The results show that high LLM label agreement does not always translate into equal downstream
performance, but the two are related. At the 250-label budget, Yelp Polarity is the strongest LLM
case: the LLM-trained classifier reaches 0.681 macro F1, slightly above the best matching
gold-label strategy at 0.639, with 0.972 LLM label accuracy. Financial PhraseBank is close to the
gold-label baseline, with 0.626 LLM macro F1 versus 0.642 for the best matching gold strategy and
0.932 label accuracy. SST-2 and Rotten Tomatoes also have high LLM label accuracy, but their
downstream gaps remain larger, suggesting that label correctness alone does not fully determine
few-shot classifier behavior.

The harder cases are mostly multi-class or schema-sensitive datasets. At budget 250, TREC reaches
0.342 macro F1 with LLM labels versus 0.496 for the best matching gold-label strategy. AG News has
0.776 LLM label accuracy but a large downstream gap, 0.262 versus 0.547 macro F1. Banking77,
Emotion, and 20 Newsgroups remain low in absolute downstream macro F1 under this simple
TF-IDF-based setup. These patterns support treating LLM annotation as another budgeted strategy to
measure, rather than assuming that model-generated labels are uniformly interchangeable with gold
labels.

As a row-level reliability pilot, we also repeated LLM annotation on held-out examples from
Financial PhraseBank, TREC, and TweetEval Sentiment using two temperatures, 0 and 0.7, with three
replicates per temperature. Financial PhraseBank is the high-agreement case: across 600
annotations from 100 unique test examples, `gpt-4o-mini` reaches 0.990 accuracy, 0.987 macro F1,
ECE 0.161, and LARI 0.828, with all six errors coming from positive examples predicted as neutral.
TREC is a harder multi-class schema case: across the same 600-annotation design, accuracy drops to
0.792 and macro F1 to 0.674, with ECE 0.084 and LARI 0.617. TweetEval Sentiment sits between these
two cases, with 0.782 accuracy, 0.766 macro F1, ECE 0.026, and LARI 0.745; its main error pattern
is sentiment softening into the neutral class. Pairwise temperature/replicate comparisons are not
significant after Bonferroni correction in all three pilots. These pilots are not a full LLM
reliability benchmark, but they show how the row-level outputs can diagnose label stability,
calibration, and task-specific failure patterns.
