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

## Validation

Run:

```bash
.venv/bin/python scripts/validate_benchmark_results.py --results results/benchmark_results.csv
.venv/bin/python -m pytest -q
```

Current status:

- Benchmark validation passed.
- Test suite passed with 83 tests.

## LLM annotator strategy status

The current paper draft reports a measured annotation strategy and budget-selection benchmark for text classification. It does not claim the full LLM annotation reliability benchmark from `DESIGN_DOC.md`.

Implemented:

- `scripts/run_llm_annotation.py` supports row-level LLM annotations for the ten benchmark datasets.
- `scripts/run_llm_strategy_benchmark.py` emits benchmark-style `llm_annotator` rows by training the shared downstream classifier on LLM-generated labels.
- `results/benchmark_results_with_llm_seed0_all_datasets.csv` contains a ten-dataset seed-0 LLM annotator grid for budgets 50, 100, and 250.
- `results/llm_strategy_seed0_summary.csv` compares those LLM rows against the best matching gold-label strategy.

Still not claimed as complete:

- Multi-seed API-backed `llm_annotator` grid.
- Cross-model agreement.
- Failure-taxonomy coding of real LLM errors.
- Human adjudication or review-rate optimization.
