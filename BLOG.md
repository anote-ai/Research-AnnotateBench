# How Much Labeled Data Do Annotation Strategies Need?

AnnotateBench is currently a framework for comparing annotation strategies with learning
curves, cost estimates, and Pareto frontiers. It helps answer a practical planning question:
if a team can label 50, 100, 250, 500, or 1000 examples, which selection strategy gives the
best downstream model quality for the cost?

The repository has two tracks today:

- an illustrative synthetic demo covering random sampling, active-learning variants, and an
  LLM annotator placeholder across five NLP task types;
- a real text-classification pilot that reveals gold labels from public datasets and trains
  TF-IDF logistic-regression classifiers under budgeted selection strategies.

The current pilot is useful, but it is not the full AnnotateBench research vision from
`DESIGN_DOC.md`. The design doc describes a broader LLM annotation reliability benchmark:
10,000 annotation tasks, real LLM annotations, human gold labels, calibration metrics, a
failure taxonomy, and a human-review optimizer. That reliability study is not complete yet.

## What Is Real Today

The public-dataset pilot measures downstream classification performance after revealing gold
labels selected by random, uncertainty, diversity, and hybrid active-learning strategies. These
results are real outputs of the pilot scripts, but they do not measure whether LLM-generated
annotations are calibrated or trustworthy.

The synthetic curves in `src/annotatebench/data.py` are different. They are hand-specified
reference curves used to exercise the learning-curve and cost-analysis code. They should not
be cited as empirical findings.

## What Is Still Missing

The most important research gap is the LARI/calibration track. The code now includes small
metric primitives for ECE and LARI, plus a failure-taxonomy data structure, but the project
still needs real LLM annotation outputs, confidence scores, human gold labels, taxonomy coding,
and review-rate experiments before it can support the claims in `DESIGN_DOC.md`.

The practical next milestone is to run a small real reliability pilot: pick one task, collect
LLM predictions with confidences, compare against gold labels, compute F1/ECE/LARI, code a
sample of failures, and report exactly what is measured.
