# Research Design Document: AnnotateBench

## Vision Statement

Build **AnnotateBench**: the first benchmark that systematically measures when LLM-generated annotations should be trusted, when they should be verified, and when they will silently fail — providing the community with calibration curves, failure taxonomies, and the **LARI** metric that becomes the standard reliability signal for LLM annotation pipelines.

---

## Problem Statement & Novelty

LLM annotation (using LLMs as data labelers) is now standard practice in NLP research and industry, yet practitioners have no principled way to answer:

1. **When is LLM annotation reliable enough to replace human annotation?** (The answer varies dramatically by task type, domain, and label schema.)
2. **When does high accuracy mask poor calibration?** (An LLM that is 85% accurate but 95% confident on errors is dangerous for downstream model training.)
3. **What is the optimal human-in-the-loop review rate?** (Reviewing 100% defeats the purpose; reviewing 0% is unsafe.)
4. **How does annotation quality degrade across label schema complexity?** (Binary → multiclass → span → relation extraction form a difficulty hierarchy.)

### Novel Contributions

| Contribution | Description |
|---|---|
| **AnnotateBench dataset** | 10,000 annotation tasks across 8 NLP task types, 6 domains, with human gold labels |
| **LARI metric** | LLM Annotation Reliability Index: calibration-adjusted F1 that penalizes overconfidence |
| **Failure taxonomy** | 5-category taxonomy of LLM annotation errors with diagnostic features |
| **Review rate optimizer** | Algorithm for setting human review thresholds given cost and quality targets |
| **Cross-annotator agreement** | First study comparing LLM–LLM agreement vs. human–LLM agreement across task types |

### LARI Definition

```
LARI = F1 × Calibration_Score

where:
  Calibration_Score = 1 - ECE  (Expected Calibration Error)
  ECE = Σ_b (|B_b| / n) × |accuracy(B_b) - confidence(B_b)|

Interpretation:
  LARI = 1.0: perfect accuracy + perfect calibration
  LARI < 0.5: unreliable for downstream use without human review
  LARI threshold for "auto-accept": 0.80 (proposed)
```

---

## Research Objectives

1. Establish **per-task-type LARI scores** for leading LLMs, creating the first reliability map for LLM annotation.
2. Identify **failure modes** where LLMs are confidently wrong (low calibration).
3. Develop an **optimal review rate algorithm** that minimizes human review cost while maintaining annotation quality above a threshold.
4. Quantify the **label schema complexity effect**: how does LARI degrade as annotation complexity increases?
5. Provide **cross-model reliability comparison**: which model is most reliable for which task types?

---

## Dataset Construction

### Task Type Coverage

| Task Type | Count | Complexity | Example |
|---|---|---|---|
| Sentiment analysis | 1,250 | Low (binary) | Positive/negative product review |
| Topic classification | 1,250 | Medium (5 classes) | News categorization |
| Named entity recognition | 1,250 | Medium (span) | Person/Org/Location extraction |
| Relation extraction | 1,250 | High (typed spans) | "X founded Y" extraction |
| Claim verification | 1,250 | High (3-way) | Supported/refuted/insufficient |
| Coreference resolution | 1,250 | Very high | Span linking |
| Instruction following eval | 1,250 | High (rubric) | LLM response quality scoring |
| Medical coding (ICD) | 1,250 | Expert-level | ICD-10 code assignment |

### Domain Coverage (6 domains)
General web text, medical literature, legal documents, financial reports, scientific papers, social media.

### Annotation Protocol

```
For each example:
1. Collect 3 human annotations (MTurk/Scale AI)
2. Adjudicate gold label (majority + expert review for disagreements)
3. Collect LLM annotations with log-probabilities for calibration
4. Record annotation time and confidence per annotator
5. Label disagreement type per error (taxonomy coding)
```

---

## Systems Under Evaluation

| System | Model | Prompting | Notes |
|---|---|---|---|
| GPT-4o zero-shot | OpenAI | Zero-shot CoT | Frontier baseline |
| GPT-4o few-shot | OpenAI | 5-shot | Calibration comparison |
| Claude Sonnet 4 zero-shot | Anthropic | Zero-shot CoT | Our primary |
| Claude Sonnet 4 few-shot | Anthropic | 5-shot | |
| Gemini 1.5 Pro | Google | Zero-shot | |
| Llama 3.1 70B | Meta | Zero-shot | Open-source |
| GPT-3.5-turbo | OpenAI | Zero-shot | Cost baseline |
| Human annotators | — | — | Gold standard |

---

## Experimental Design

### Baseline Experiment (Experiment 0)
**Protocol**: Run GPT-4o zero-shot on sentiment analysis (simplest task type). Compute F1, accuracy, ECE, LARI.

**Expected result**: F1 ≈ 0.91, ECE ≈ 0.04, LARI ≈ 0.88. This establishes that LLMs are highly reliable for simple sentiment annotation — the interesting question is where reliability drops.

---

### Experiment 1: LARI by Task Type
**Hypothesis**: LARI drops monotonically with task complexity; medical coding and coreference resolution have LARI < 0.55 for all models.

**Protocol**:
1. Run all LLM systems on all 8 task types (zero-shot).
2. Compute F1, ECE, LARI for each (model × task type) cell.
3. Visualize LARI heatmap.
4. Statistical test: Friedman test for model ranking consistency across task types.

**Expected results**:

| Task Type | GPT-4o LARI | Claude LARI | Llama 70B LARI |
|---|---|---|---|
| Sentiment | 0.88 | 0.86 | 0.79 |
| Topic classification | 0.82 | 0.80 | 0.71 |
| NER | 0.74 | 0.76 | 0.63 |
| Relation extraction | 0.68 | 0.69 | 0.54 |
| Claim verification | 0.65 | 0.67 | 0.52 |
| Coreference | 0.51 | 0.53 | 0.38 |
| Instruction eval | 0.72 | 0.75 | 0.60 |
| Medical coding | 0.44 | 0.46 | 0.31 |

- Key finding: LARI < 0.55 for medical coding across all models — human annotation required.
- Models rank consistently (Friedman p > 0.05): LARI is a stable discriminator.

---

### Experiment 2: Failure Taxonomy
**Hypothesis**: LLM annotation errors fall into 5 categories with distinct prevalence rates by task type.

**Protocol**:
1. Sample 500 LLM errors (stratified by task type and model).
2. Two annotators code each error into taxonomy categories.
3. Compute inter-rater agreement (Cohen's κ).
4. Analyze failure type prevalence by task type and model family.

**Taxonomy**:
```
Category A: Label ambiguity (correct given alternate interpretation)
Category B: Knowledge gap (requires domain expertise not in training)
Category C: Context window failure (answer in document but not attended to)
Category D: Schema confusion (misunderstands label definition)
Category E: Overconfident hallucination (fabricates with high confidence)
```

**Expected results**:
- Sentiment/topic: 60% Category A (ambiguity), 20% D (schema)
- Medical coding: 55% Category B (knowledge gap), 30% E (hallucination)
- Coreference: 50% Category C (context window), 30% A (ambiguity)
- Overconfident hallucination (E) is most dangerous: 85% confidence on wrong labels

---

### Experiment 3: Calibration vs. Accuracy
**Hypothesis**: Few-shot prompting improves calibration (lower ECE) by 40% vs. zero-shot, without necessarily improving F1.

**Protocol**:
1. Compare zero-shot vs. 5-shot prompting for all models on all task types.
2. Compute delta-F1 and delta-ECE for each condition.
3. Test whether few-shot examples improve calibration more than accuracy.

**Expected results**:
- Few-shot vs. zero-shot:
  - Mean delta-F1: +0.03 (modest accuracy improvement)
  - Mean delta-ECE: −0.08 (substantial calibration improvement)
  - LARI improvement: +0.09
- Key finding: few-shot examples primarily improve calibration, not accuracy — a novel finding for the annotation use case

---

### Experiment 4: Optimal Review Rate
**Hypothesis**: A LARI-threshold-based review policy achieves 90% of human annotation quality at <30% human review rate.

**Protocol**:
1. For each task type, train a review predictor: should this annotation be sent for human review?
2. Features: LLM confidence, task type, example length, few-shot LARI estimate.
3. Sweep review rate thresholds (0–100%).
4. Compute downstream model quality (trained on auto-accepted + reviewed labels) vs. all-human baseline.

**Expected results**:
- At 25% review rate: downstream model F1 = 97% of all-human baseline
- At 10% review rate: downstream model F1 = 89% of all-human baseline
- Optimal threshold: review when LARI < 0.70 (catches 80% of errors with 22% review rate)
- Cost savings vs. all-human: 78% reduction at 97% quality retention

```python
# Review rate optimizer
def optimal_review_threshold(target_quality_retention=0.95, lari_scores, human_labels):
    for threshold in np.arange(0.5, 1.0, 0.01):
        review_rate = np.mean(lari_scores < threshold)
        # simulate downstream quality
        quality = simulate_downstream_quality(lari_scores, human_labels, threshold)
        if quality >= target_quality_retention:
            return threshold, review_rate
```

---

### Experiment 5: Cross-Model Agreement Analysis
**Hypothesis**: LLM–LLM agreement is higher than human–LLM agreement due to shared training biases, creating a false sense of annotation reliability.

**Protocol**:
1. Compute pairwise agreement (Cohen's κ) for all (model × model) and (model × human) pairs.
2. Compare mean LLM–LLM κ vs. mean human–LLM κ.
3. Test whether LLM–LLM agreement predicts gold label accuracy.

**Expected results**:
- LLM–LLM mean κ ≈ 0.79
- Human–LLM mean κ ≈ 0.68
- LLM–LLM agreement does NOT predict gold label accuracy (r ≈ 0.12)
- Key finding: using multiple LLMs as "annotators" and taking majority vote does not improve reliability on hard tasks; it amplifies shared errors

---

## Expected Results Summary

| Finding | Result |
|---|---|
| LARI < 0.55 for medical coding | All models; human annotation required |
| LARI ≥ 0.80 for sentiment | All frontier models; safe to auto-annotate |
| Few-shot calibration improvement | −0.08 ECE, only +0.03 F1 |
| Optimal review rate | 22% review rate → 97% quality retention |
| LLM–LLM agreement inflation | +11 pp κ over human–LLM, but does not predict accuracy |

**Primary claim**: LARI, not F1, should be the standard reliability metric for LLM annotation pipelines; and per-task-type LARI thresholds enable principled human review budgeting.

---

## Why This Matters

**For researchers**: AnnotateBench provides the first standardized framework for evaluating LLM annotators — critical as the field increasingly uses LLM-labeled data for training.

**For practitioners**: LARI and the review rate optimizer directly reduce annotation costs while maintaining quality guarantees.

**For Anote products**: Anote's annotation platform can integrate LARI as a quality signal, directly improving product value and differentiation.

**Market**: The data annotation market is $1.7B and growing; AI-powered annotation tools are replacing human annotation workflows.

---

## Implementation Plan

```
research-annotatebench/
├── data/
│   ├── tasks/           # 10,000 annotation tasks
│   ├── gold_labels/     # Human adjudicated labels
│   └── llm_annotations/ # Per-model annotations + log-probs
├── metrics/
│   ├── lari.py          # LARI computation
│   ├── ece.py           # Expected Calibration Error
│   └── agreement.py     # Inter-annotator agreement
├── taxonomy/
│   └── failure_coder.py # Error taxonomy labeling tool
├── review_optimizer/
│   └── threshold_sweep.py
├── experiments/
│   ├── exp0_baseline.py
│   ├── exp1_lari_by_task.py
│   ├── exp2_taxonomy.py
│   ├── exp3_calibration.py
│   ├── exp4_review_rate.py
│   └── exp5_agreement.py
└── leaderboard/
```

---

## Timeline

| Phase | Duration | Deliverable |
|---|---|---|
| Dataset collection & human annotation | 8 weeks | 10,000 tasks with gold labels |
| LLM annotation collection | 3 weeks | All model annotations |
| Taxonomy coding | 3 weeks | Error taxonomy labels |
| Experiments | 5 weeks | All results |
| Paper writing | 4 weeks | ACL 2026 submission |

**Target venue**: ACL 2026 or EMNLP 2026

---

## Open Questions & Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Human annotation cost for 10K tasks | High | Phased approach; start with 3K |
| LARI threshold generalization | Medium | Validate on held-out domains |
| LLM log-probability access | Medium | Use models with exposed logprobs |
| Medical coding expert cost | High | Partner with medical school |

---

## Related Issues

- Product integration: Anote annotation platform
- Reproducibility package
- Statistical rigor
- Related work audit: LLMAAA, PromptAnnotator, AnnoLLM
