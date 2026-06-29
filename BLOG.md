# How Much Labeled Data Do You Actually Need? Introducing AnnotateBench

*A practitioner's guide to picking an annotation strategy without guessing.*

## The problem

Every NLP project starts with the same question: "How many examples do we need to label, and
how should we label them?" Teams usually guess — pick a budget that feels reasonable, label
randomly or with whatever active-learning library is on hand, and hope the resulting model is
good enough. There is no standard way to compare "spend $50 on uncertainty sampling" against
"spend $50 on an LLM-as-annotator" before you've already spent the money.

AnnotateBench is our attempt to make that comparison concrete. It models five annotation
strategies — random sampling, uncertainty-based active learning, diversity-based active
learning, a hybrid of the two, and using an LLM directly as an annotator — across five budget
levels (50, 100, 250, 500, and 1,000 labeled examples) and multiple NLP task types
(classification, NER, QA, summarization, instruction tuning).

## What you get

For each (strategy, task, budget) combination, AnnotateBench fits a **learning curve**: how
F1 improves as you add more labeled data, modeled as a power law `F1 = a * budget^b`. From
the curves you can derive:

- **Pareto frontiers**: which strategies give you the best F1 for the money, at every budget.
- **Cost-to-target**: how much you'd need to spend with each strategy to hit a target F1
  (e.g., "what's the cheapest path to F1 = 0.85?").
- **Efficiency rankings**: F1 per dollar spent, so you can compare strategies on equal footing.

The headline pattern in the current model: LLM-as-annotator strategies front-load quality —
they look great at small budgets (50-100 examples) because the LLM already "knows" the task —
but active-learning strategies (especially the hybrid approach) catch up and overtake them once
the budget grows past a few hundred examples, because human-labeled data has lower error rates
and active learning targets the most informative examples.

## An important caveat: what's real today

AnnotateBench's current release is a **cost/strategy-comparison framework with illustrative
learning curves**, not yet a study with measured numbers from running real models against real
datasets. The F1-vs-budget curves shipped in the code are hand-specified, plausible reference
curves (with small synthetic noise added) used to validate the framework's math — Pareto
frontier computation, power-law curve fitting, cost-efficiency analysis — end to end. They are
**not yet outputs of actually running five annotation strategies against five live NLP
datasets.**

This distinction matters and we want to be upfront about it. The original research design for
this project (see `DESIGN_DOC.md`) is actually more ambitious than what's built: it proposes a
benchmark for measuring *when LLM annotations should be trusted* (a calibration-aware
reliability metric called LARI, a failure taxonomy for LLM annotation errors, and an optimal
human-review-rate policy) across 8 task types and 6 domains with real human-adjudicated gold
labels. None of that — LARI, the failure taxonomy, the review-rate optimizer, or the underlying
10,000-task dataset — has been implemented yet. What exists today is a smaller, different
(though related) framework focused purely on cost/budget learning curves for annotation
strategy selection, with placeholder numbers.

## What's next

Turning this into a real empirical study means: (1) wiring up the package to real datasets and
actually running the five strategies (this requires LLM API calls and/or training small
classifiers under each annotation budget), (2) replacing the illustrative curves with measured
results, and (3) building out the LARI/calibration line of work from the original design doc as
a separate, complementary contribution. See `PAPER_DRAFT.md` for the structured writeup and
`DESIGN_DOC.md` for the full original vision.

---

*This post accompanies the `research-annotatebench` repository. Code: `src/annotatebench/`.
Questions or want to contribute a real experimental run? Open an issue.*
