# AnnotateBench Paper Draft Skeleton

Status: early draft, not submission-ready.

This skeleton maps the current repository to the larger research plan in `DESIGN_DOC.md`.
It deliberately separates implemented code, measured pilot outputs, illustrative demo curves,
and not-yet-implemented LLM annotation reliability experiments.

## Current Claim Boundary

AnnotateBench currently supports a learning-curve and cost-frontier framework for annotation
strategy selection. The current branch also includes a real public-dataset text-classification
pilot that simulates annotation by revealing existing gold labels.

The repository does not yet contain a completed LLM annotation reliability benchmark. It does
not yet include real LLM annotation runs with confidence scores, human adjudication, calibrated
review policies, or a coded failure-taxonomy study.

## Implemented

- Core learning-curve data structures and utilities in `src/annotatebench/core.py`.
- Cost and Pareto analysis utilities in `src/annotatebench/evaluate.py` and
  `src/annotatebench/costs.py`.
- Synthetic illustrative reference curves in `src/annotatebench/data.py`, including distinct
  QA, summarization, and instruction-tuning curves. These are not measured results.
- Real text-classification pilot machinery in `src/annotatebench/pilot.py`,
  `src/annotatebench/datasets.py`, and `scripts/run_pilot.py`.
- Pilot outputs in `results/pilot_results*.csv`, `results/budget_recommendations.csv`, and
  `figures/*.png`.
- ECE/LARI metric primitives in `src/annotatebench/metrics/lari.py`.
- Failure taxonomy structures in `src/annotatebench/taxonomy/failure_coder.py`.

## Partially Implemented

- LARI exists as a pure metric calculation, but no real LLM annotation reliability experiment
  has populated it with measured model confidences and human/gold correctness labels.
- The failure taxonomy exists as code-level categories and records, but no sample of real
  annotation failures has been coded by human reviewers.
- The paper draft in `paper/main.tex` reports the classification pilot, not the full reliability
  benchmark described in `DESIGN_DOC.md`.

## Not Yet Implemented

- 10,000-task AnnotateBench reliability dataset across the eight task types in `DESIGN_DOC.md`.
- Real LLM annotation calls with versioned prompts, model names, costs, and confidence scores.
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
4. Pilot experiments: public-dataset text-classification results only.
5. Reliability extension: planned LLM annotation reliability protocol, clearly marked as future
   work until measured.
6. Limitations: synthetic demo curves, gold-label reveal simulation, no completed LLM reliability
   run, no human failure coding yet.

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
