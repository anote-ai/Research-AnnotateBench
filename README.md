# research-annotatebench

**Motivation:** Annotation budget is a primary constraint in NLP system development. This benchmark systematically compares annotation strategies across public text-classification datasets and label budgets using learning curves.

## Current Status

This repository is not the full LLM annotation reliability benchmark described in
`DESIGN_DOC.md`. The current code supports:

- illustrative synthetic learning curves for exercising cost and Pareto utilities;
- a real public-dataset text-classification benchmark that simulates annotation by revealing gold
  labels under budgeted selection strategies across ten classification datasets;
- an API-backed `llm_annotator` strategy path that asks an OpenAI model to label selected
  training examples, trains the same downstream classifier on those labels, and writes
  benchmark-style rows;
- small ECE/LARI metric primitives and failure-taxonomy data structures for LLM annotation
  diagnostics.

It does not yet contain human adjudication, failure-taxonomy coding, a review-rate optimizer, or
the full cross-model reliability study. See `BLOG.md`, `PAPER_DRAFT.md`, and `results/README.md`
for the claim boundary.

## Benchmark Grid

The measured gold-label benchmark path covers 10 public text-classification datasets × 4 selection
strategies × 5 budget levels. The committed benchmark outputs cover all ten datasets with
three strategy-selection seeds. The LLM annotator strategy is implemented as the fifth strategy
and should be run separately with `OPENAI_API_KEY` before merging its rows into paper tables.

**Measured datasets:** Financial PhraseBank, TREC, Banking77, AG News, SST-2, 20 Newsgroups, Rotten Tomatoes, Yelp Polarity, TweetEval Sentiment, and Emotion.

**Future task extensions:** NER, QA, summarization, and relation extraction require task-specific loaders, annotation units, and metrics, so they are not claimed as measured results in the current classification benchmark.

## Measured Benchmark Methodology

The current gold-label benchmark trains TF-IDF + logistic-regression classifiers at each budget
and compares random sampling, uncertainty active learning, diversity active learning, and hybrid
active learning. Annotation is simulated by revealing existing gold labels from public datasets.
The `llm_annotator` strategy instead labels selected training examples with an OpenAI model and
trains the same downstream classifier on those model-generated labels.

## Pareto Analysis

For each strategy, we plot cost vs. best F1 and compute the Pareto frontier — strategies that cannot be strictly dominated in both dimensions. This guides practitioners to cost-efficient annotation choices.

## Cost Calibration

Cost scenarios are calibrated from dated public pricing sources in
`config/cost_sources.json`. Human annotation cost is estimated as:

```text
hourly_reward_usd * seconds_per_label / 3600 * (1 + platform_fee_rate) * redundancy
```

The default scenarios keep the historical `low`, `base`, and `high` names for compatibility:

- `low`: Amazon Mechanical Turk single-rater short-text estimate.
- `base`: Prolific recommended-pay single-rater short-text estimate.
- `high`: Prolific higher-pay, slower-task, two-rater estimate.

API annotation cost does not require an API key for calibration. Use the public model price and
estimated token counts:

```text
input_tokens * input_usd_per_million_tokens / 1_000_000
+ output_tokens * output_usd_per_million_tokens / 1_000_000
```

Managed annotation vendors with quote-based pricing should be documented qualitatively unless a
dated quote is available.

## Synthetic Demo for Development

The repository also keeps a small synthetic demo for exercising learning-curve, cost, and Pareto
utilities in tests. It covers 5 strategies × 6 tasks × 5 budget levels = 30 learning curves and
150 hand-specified data points. These values live in `src/annotatebench/data.py`; they are
illustrative reference curves, not experiment results, and should not be cited as measured
findings.

```python
from annotatebench import AnnotationStrategy, NLPTask, make_learning_curve

curve = make_learning_curve(AnnotationStrategy.HYBRID_AL, NLPTask.CLASSIFICATION)
best = curve.best_point()

print(best.budget, best.f1)
```

## Running the Measured Benchmark

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

Run the 10-dataset text-classification benchmark:

```bash
./scripts/run_benchmark.py \
  --financial-phrasebank-path data/raw/FinancialPhraseBank-v1.0.zip \
  --trec-train-path data/raw/trec_train.csv \
  --trec-test-path data/raw/trec_test.csv \
  --banking77-train-path data/raw/banking77_train.csv \
  --banking77-test-path data/raw/banking77_test.csv \
  --output-csv results/benchmark_results.csv
./scripts/make_figures.py --results results/benchmark_results.csv --cost-scenario base
```

The benchmark runner uses HuggingFace datasets for AG News, SST-2, 20 Newsgroups, Rotten Tomatoes, Yelp Polarity, TweetEval Sentiment, and Emotion. Large datasets are capped by default to keep active-learning experiments tractable. DBpedia-14 and IMDb remain available as optional heavier loaders, but are not part of the default 10-dataset run.

Run budget recommendations across available benchmark CSVs:

```bash
./scripts/make_budget_recommendations.py \
  --results results/benchmark_results.csv \
  --output-csv results/budget_recommendations.csv \
  --cost-scenario base
```

Summarize the API-backed LLM annotator strategy against the best gold-label strategy at matching
dataset, budget, seed, and cost scenario:

```bash
./scripts/make_llm_strategy_summary.py \
  --llm-results results/benchmark_results_with_llm_seed0_all_datasets.csv \
  --output-csv results/llm_strategy_seed0_summary.csv
```

Run a zero-shot LLM annotation experiment with the saved Financial PhraseBank prompt:

```bash
OPENAI_API_KEY=... python scripts/run_llm_annotation.py \
  --financial-phrasebank-path data/raw/FinancialPhraseBank-v1.0.zip \
  --model gpt-4o-mini \
  --split test \
  --limit 100 \
  --output-csv results/llm_annotations_financial_phrasebank.csv \
  --summary-csv results/llm_annotation_summary.csv
```

The script reads `prompts/financial_phrasebank_v1_zero_shot_json.md`, writes row-level
annotations with `prompt_version`, `predicted_label`, `confidence`, and `rationale`, then
summarizes accuracy, macro F1, ECE, and LARI. Use `--dry-run` to validate the local data path
without API calls.

Run the LLM annotator as a benchmark strategy. Start with dry runs or small budgets before
spending API credits:

```bash
python scripts/run_llm_strategy_benchmark.py \
  --datasets financial_phrasebank \
  --budgets 5 \
  --seeds 0 \
  --financial-phrasebank-path data/raw/FinancialPhraseBank-v1.0.zip \
  --dry-run \
  --output-csv results/benchmark_results_with_llm.csv
```

Then run the API-backed version from a terminal where `OPENAI_API_KEY` is set:

```bash
OPENAI_API_KEY=... python scripts/run_llm_strategy_benchmark.py \
  --financial-phrasebank-path data/raw/FinancialPhraseBank-v1.0.zip \
  --trec-train-path data/raw/trec_train.csv \
  --trec-test-path data/raw/trec_test.csv \
  --banking77-train-path data/raw/banking77_train.csv \
  --banking77-test-path data/raw/banking77_test.csv \
  --budgets 50,100,250 \
  --seeds 0,1,2 \
  --output-csv results/benchmark_results_with_llm.csv
```

For API cost estimates, pass dated token prices explicitly with
`--input-usd-per-million-tokens` and `--output-usd-per-million-tokens`; otherwise cost columns are
left as zero-estimate placeholders. The LLM strategy runner uses nested random budgets per
dataset/seed and resumes from a shared row-level CSV in `results/llm_annotations/` by default, so
interrupted runs do not re-call the API for examples already written to disk. Pass `--no-resume`
only when you intentionally want to refresh all annotations.

## Venue

**JDSE 2026** draft materials are in `paper/main.tex`. The ten-dataset classification
benchmark artifacts are present; the project is not yet submission-ready for the full
`DESIGN_DOC.md` LLM reliability claims.

## Citation

```bibtex
@article{anote2026annotatebench,
  title   = {AnnotateBench: How Much Labeled Data Do Annotation Strategies Need Across NLP Tasks?},
  author  = {Anote AI},
  journal = {JDSE},
  year    = {2026},
}
```
