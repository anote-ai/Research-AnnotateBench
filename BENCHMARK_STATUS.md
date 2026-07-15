# AnnotateBench Benchmark Status

## Completed measured gold-label benchmark

- Scope: 10 public text-classification datasets x 4 selection strategies x 5 label budgets x 3 seeds x 3 cost scenarios.
- Result grid: `results/benchmark_results.csv` contains 1,800 measured rows.
- Datasets: Financial PhraseBank, TREC, Banking77, AG News, SST-2, 20 Newsgroups, Rotten Tomatoes, Yelp Polarity, TweetEval Sentiment, and Emotion.
- Strategies: random, uncertainty active learning, diversity active learning, and hybrid active learning.
- Figures: `figures/learning_curve_*.png` and `figures/pareto_*.png` contain 10 learning curves and 10 Pareto frontiers for the final dataset set.
- Summaries:
  - `results/benchmark_best_overall.csv`
  - `results/benchmark_best_by_strategy.csv`
  - `results/budget_recommendations.csv`
  - `results/paper_core_summary.csv`

## Validation

Run:

```bash
.venv/bin/python scripts/validate_benchmark_results.py --results results/benchmark_results.csv
.venv/bin/python -m pytest -q
```

Current status:

- Benchmark validation passed.
- Test suite passed with 114 tests.

## Downstream model robustness status

Implemented:

- The default measured benchmark remains TF-IDF + logistic regression.
- `scripts/run_benchmark.py`, `scripts/run_pilot.py`, and
  `scripts/run_llm_strategy_benchmark.py` accept `--downstream-model`.
- `results/benchmark_results_sentence_transformer_logreg.csv` contains the measured ten-dataset
  sentence-transformer + logistic-regression robustness grid.
- `results/downstream_model_comparison_best.csv` and
  `results/downstream_model_comparison_summary.csv` compare the robustness grid against the
  primary TF-IDF baseline.

Claim boundary:

- The sentence-transformer results are exploratory robustness evidence. The primary paper table
  remains the TF-IDF + logistic-regression benchmark.

## LLM annotator strategy status

The current paper draft reports a measured annotation strategy and budget-selection benchmark for text classification. It does not claim the full LLM annotation reliability benchmark from `DESIGN_DOC.md`.

Implemented:

- `scripts/run_llm_annotation.py` supports row-level LLM annotations for the ten benchmark datasets.
- `scripts/run_llm_strategy_benchmark.py` emits benchmark-style `llm_annotator` rows by training the shared downstream classifier on LLM-generated labels.
- `results/benchmark_results_with_llm_seed0_all_datasets.csv` contains a ten-dataset seed-0 LLM annotator grid for budgets 50, 100, and 250.
- `results/benchmark_results_with_llm_seed1_2_all_datasets.csv` contains the matching seed-1 and seed-2 LLM annotator grid.
- `results/llm_strategy_seed0_summary.csv`, `results/llm_strategy_seed1_2_summary.csv`, and
  `results/llm_strategy_seed0_1_2_summary.csv` compare those LLM rows against the best matching
  gold-label strategy.
- `results/llm_api_cost_seed1_2_summary.csv` summarizes recorded seed-1 and seed-2 API token usage
  and estimated `gpt-4o-mini` cost as an auxiliary analysis. The seed-0 run predates the complete
  token-logging schema, so LLM rows are still excluded from the main human-label Pareto frontiers.
- `results/statistical_appendix.csv` and `results/statistical_significance.csv` contain combined
  aggregate diagnostics for the Financial PhraseBank, TREC, and TweetEval Sentiment row-level
  reliability pilots.

Still not claimed as complete:

- Measured API cost frontiers for the full seed-0/1/2 LLM annotator grid.
- Cross-model agreement.
- Failure-taxonomy coding of real LLM errors.
- Human adjudication or review-rate optimization.
