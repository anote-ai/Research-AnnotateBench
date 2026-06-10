# AnnotateBench — Research Design Document

## Goal

Build the first systematic benchmark for comparing LLM annotation quality to human crowdworker annotation quality — with focus on calibration (not just accuracy) and across a diverse set of NLP annotation tasks.

## Objective

1. Construct a benchmark of 2,000+ annotation items across 8 task types (sentiment, NER, stance detection, text classification, summarization quality, factuality, toxicity, relevance)
2. Collect ground-truth human annotations from expert annotators and crowdworkers for all items
3. Evaluate 6+ LLMs on all tasks, measuring both accuracy AND calibration

## Background / Motivation

In 2023, Gilardi et al. published "ChatGPT outperforms crowd workers for text annotation tasks" — one of the most-cited NLP papers of the year. This sparked enormous practitioner interest in replacing crowdworkers with LLMs. But the paper studied a few tasks with one model and didn't measure calibration.

The AI annotation market is being disrupted right now. Practitioners lack a principled benchmark to determine which tasks are safe to delegate to LLMs vs. which require human review.

## Experimental Design

### Baseline Experiment

**Replicate Gilardi et al. on their original tasks using GPT-4o and two crowdworker platforms (MTurk, Prolific)**

- Metric: accuracy vs. expert gold labels; Cohen's κ vs. human majority vote
- Purpose: verify infrastructure matches published results; establish baseline agreement rates
- Expected result: GPT-4o matches Gilardi et al. accuracy (~75–80%); κ ≈ 0.65–0.75

### Test Experiment 1: Calibration vs. Accuracy Across 8 Task Types

For all 8 task types, collect LLM annotation with confidence scores and crowdworker annotation with individual disagreement rates. Measure for each model and task type: accuracy, Expected Calibration Error (ECE), overconfidence rate.

**Expected result:** LLMs are more accurate than crowdworkers on 5/8 task types but more overconfident on 7/8 — **"LLMs are more accurate but less honest about when they're uncertain"**

### Test Experiment 2: Task Difficulty and LLM Reliability

Stratify tasks by human inter-annotator agreement (IAA): easy (>0.8), medium (0.5–0.8), hard (<0.5). Measure LLM performance on each stratum.

**Expected result:** LLMs perform well on easy tasks but fail to capture genuine human disagreement on hard tasks

### Test Experiment 3: LLM Annotation Reliability Index (LARI)

Build a per-task LARI combining accuracy and calibration into a deployment recommendation:
- LARI > 0.8: safe to use LLM annotation autonomously
- LARI 0.5–0.8: use LLM with spot-check review
- LARI < 0.5: require human annotation

Validate: does LARI correctly predict annotation quality on held-out test sets?

## Expected Results

1. A benchmark of 2,000+ annotation items with expert gold labels across 8 task types
2. Accuracy and calibration tables for 6+ LLMs vs. crowdworkers
3. **Key finding:** "LLMs are 15% more accurate but 2x more overconfident than crowdworkers — calibration is the missing dimension"
4. The LLM Annotation Reliability Index: a practical per-task deployment recommendation tool

## Why This Matters / Why People Would Care

- **Data teams:** making LLM-vs-human annotation decisions now without principled tools; LARI gives them a concrete basis
- **Annotation platforms** (Scale AI, Labelbox): need to know which tasks to delegate to LLMs vs. humans
- **Researchers:** calibration in LLM annotation is almost entirely unexplored
- **AI ethics:** overconfident annotators produce training data that inherits that overconfidence

## Timeline

| Month | Milestone |
|---|---|
| 1–2 | Annotation item construction and expert gold labeling |
| 3 | Crowdworker data collection (MTurk + Prolific) |
| 4 | LLM annotation + calibration measurement |
| 5 | Analysis + LARI derivation |
| 6 | Submission to ACL 2026 |

## Related Issues

- Design doc GitHub issue: #20
- Target conferences: see issues labeled `conference-prep`
- Reproducibility package: see issues labeled `artifact-release`
