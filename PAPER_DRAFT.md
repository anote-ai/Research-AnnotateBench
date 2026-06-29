# AnnotateBench: A Cost-Aware Benchmark for Annotation Strategy Selection
### Paper draft skeleton — status: early draft, NOT submission-ready

> **Status note (read first):** This is a structural skeleton for a future paper, written to
> match what is actually implemented in this repository today. Numbers marked
> **(projected, pending full experiment run)** are illustrative reference values used to
> validate the codebase's math (curve fitting, Pareto frontier, cost-efficiency metrics) — they
> were NOT produced by running real models against real datasets, and must not be cited or
> quoted as empirical findings. Sections describing experiments not yet implemented in code are
> marked **(not yet implemented)**.

## Abstract (draft)

Choosing an annotation strategy — random sampling, active learning variants, or LLM-as-annotator
— is a high-stakes, under-studied decision in applied NLP. We introduce AnnotateBench, a
framework for comparing annotation strategies via fitted learning curves, Pareto-optimal
cost/F1 frontiers, and budget-to-target-quality estimates, across multiple NLP task types and
budget levels. *(Once real experiments are run, this abstract should report actual cross-task
findings, not the current framework description.)*

## 1. Introduction

- Motivation: annotation budget is a primary constraint in real-world NLP system development
  (restated from `README.md` / `DESIGN_DOC.md`).
- Gap: no standard, reproducible way to compare strategy cost-efficiency before committing
  budget.
- Contribution claimed: (a) a learning-curve-based comparison framework with Pareto frontier and
  budget-recommendation tooling (**implemented**), (b) reference implementations of cost models
  for API vs. human annotation (**implemented**), (c) a broader reliability/calibration study
  (LARI metric, failure taxonomy, review-rate optimizer — **not yet implemented**, see
  `DESIGN_DOC.md`).

## 2. Related Work (not yet drafted)
LLMAAA, PromptAnnotator, AnnoLLM, active learning surveys — placeholders carried over from
`DESIGN_DOC.md`'s "Related work audit" action item. **(not yet implemented — needs literature
pass)**

## 3. Method

### 3.1 Strategies
Five strategies are modeled: Random, Uncertainty AL, Diversity AL, Hybrid AL, LLM Annotator
(`src/annotatebench/core.py::AnnotationStrategy`).

### 3.2 Tasks
Five NLP task types: Classification, NER, QA, Summarization, Instruction Tuning
(`src/annotatebench/core.py::NLPTask`). Of these, Classification and NER currently have distinct
reference F1/cost curves (`src/annotatebench/data.py`); QA, Summarization, and Instruction
Tuning currently fall back to the Classification curves as a placeholder and need task-specific
modeling. **(partially implemented)**

### 3.3 Learning curve model
F1 is modeled as a power law in budget, `F1 = a * budget^b`, fit by log-linear regression
(`fit_learning_curve` in `core.py`). This is implemented and unit-tested
(`tests/test_core.py::test_fit_learning_curve_power_law`).

### 3.4 Cost model
`CostModel` (`core.py`) computes total API vs. human annotation cost, including an optional
human-review cost applied to a fraction of API-annotated samples equal to the API error rate.
Implemented and tested (`tests/test_cost_metrics.py`).

### 3.5 Evaluation metrics
Implemented in `src/annotatebench/evaluate.py`: Pareto frontier over (cost, F1) pairs, trapezoidal
area-under-learning-curve, cost-to-target-F1 interpolation, strategy comparison table, and
budget recommendation. Implemented and tested (`tests/test_evaluate.py`).

### 3.6 LARI / calibration track (not yet implemented)
The original design (`DESIGN_DOC.md`) additionally proposes the **LARI** metric
(`F1 x (1 - Expected Calibration Error)`), a 5-category failure taxonomy for LLM annotation
errors, and a review-rate optimizer trained on LLM confidence features. None of this is present
in `src/annotatebench/` today. This is the largest gap between the design doc's vision and the
current codebase and represents the highest-value next milestone for any paper claiming the
"AnnotateBench" name as originally scoped.

## 4. Experiments

### 4.1 Validating the framework (implemented)
We validate that the curve-fitting, Pareto-frontier, and cost-efficiency code is correct via
unit tests (19 tests across `tests/test_core.py`, `tests/test_cost_metrics.py`,
`tests/test_data.py`, `tests/test_evaluate.py`) and an end-to-end demo
(`scripts/run_demo.py`) that exercises the full pipeline on reference curves.

### 4.2 Strategy comparison **(projected, pending full experiment run)**
Using the reference learning curves shipped in `data.py` (not measured from live model runs):

| Strategy | Best F1 (Classification, budget=1000) |
|---|---|
| Random | 0.72 |
| Uncertainty AL | 0.84 |
| Diversity AL | 0.82 |
| Hybrid AL | 0.86 |
| LLM Annotator | 0.84 |

These values are the hand-specified reference curve endpoints in
`src/annotatebench/data.py::STRATEGY_F1_CURVES`, included here only to illustrate the kind of
table a completed study would produce. They must be replaced with measured results before any
submission.

### 4.3 Real experimental run **(not yet implemented)**
To produce real numbers we need to: (1) select concrete datasets per task type, (2) implement
actual annotator agents (active-learning loop against a real classifier, and LLM-annotator
calls against a real API), (3) run each (strategy, task, budget) cell with multiple seeds, and
(4) log real F1/cost into a `results/` directory consumed by `evaluate.py`. None of steps 1-4
exist yet.

## 5. Discussion (not yet drafted)
Pending real results.

## 6. Limitations
- All current numeric results are illustrative/synthetic, not measured.
- Only 2 of 5 task types have distinct cost/F1 curves; the rest are placeholders.
- No connection yet to the LARI/calibration research line from `DESIGN_DOC.md`.
- No real dataset, no real LLM API calls, no human annotation collected.

## 7. Reproducibility
`pip install -e .`, then `python scripts/run_demo.py` reproduces the illustrative tables in
Section 4.2 exactly (deterministic given `seed=42`). Unit tests: `pytest tests/`.
