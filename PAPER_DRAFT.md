# AnnotateBench Paper Draft Skeleton

Status: ten-dataset classification benchmark draft; LLM annotator strategy code path added; full LLM reliability benchmark is not complete.

This skeleton maps the current repository to the larger research plan in `DESIGN_DOC.md`.
It deliberately separates implemented code, measured pilot outputs, illustrative demo curves,
and not-yet-implemented LLM annotation reliability experiments.

## Current Claim Boundary

AnnotateBench currently supports a learning-curve and cost-frontier framework for annotation
strategy selection. The current branch includes a real ten-dataset public text-classification
benchmark that simulates annotation by revealing existing gold labels.

The repository now contains an API-backed LLM annotator runner that can produce benchmark-style
`llm_annotator` rows. It does not yet contain a completed LLM annotation reliability benchmark,
human adjudication, calibrated review policies, or a coded failure-taxonomy study.

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
- Benchmark validation in `scripts/validate_benchmark_results.py`.
- Paper summary generation in `scripts/make_paper_summary_table.py`.
- Optional downstream-model comparison support for `sentence_transformer_logreg`; this is
  implemented as a runner option but is not yet a measured result.
- LLM annotator scripts in `scripts/run_llm_annotation.py` and
  `scripts/run_llm_strategy_benchmark.py`.
- ECE/LARI metric primitives in `src/annotatebench/metrics/lari.py`.
- Failure taxonomy structures in `src/annotatebench/taxonomy/failure_coder.py`.

## Partially Implemented

- LARI exists as a diagnostic metric and is populated for small LLM annotation runs, but not yet
  for a complete ten-dataset LLM annotator grid.
- The failure taxonomy exists as code-level categories and records, but no sample of real
  annotation failures has been coded by human reviewers.
- The paper draft in `paper/main.tex` reports the classification strategy benchmark, not the full
  reliability benchmark described in `DESIGN_DOC.md`.
- Stronger downstream learner support exists, but the sentence-transformer comparison still needs
  to be run and summarized before it can be claimed as evidence.

## Not Yet Implemented

- 10,000-task AnnotateBench reliability dataset across the eight task types in `DESIGN_DOC.md`.
- Full ten-dataset LLM annotator strategy run with versioned prompts, model names, costs, and
  confidence scores.
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
4. Experiments: ten public-dataset text-classification results only.
5. Reliability extension: planned LLM annotation reliability protocol, clearly marked as future
   work until measured.
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
