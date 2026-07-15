# AnnotateBench Artifact Guide

This document is the reproducibility entrypoint for reviewers. It describes the
current release artifact honestly: the repository supports the public
text-classification benchmark, a seed-0/1/2 LLM annotator extension, aggregate
row-level reliability pilot diagnostics, and statistical appendix generation,
but it is not yet a public human-subjects annotation release or the full
multi-task LLM reliability benchmark.

## Quick Validation

Run the smoke artifact without network access or API keys:

```bash
./scripts/run_all.sh
```

The script creates a tiny local fixture under `results/artifact_smoke/`, then
runs:

- gold-label text-classification pilot;
- compact benchmark summaries;
- paper summary table;
- learning-curve and Pareto figures;
- dry-run LLM row-level annotation with temperature/replicate fields;
- statistical appendix and significance CSVs from the row-level annotations.

Expected runtime is under five minutes on a laptop. The smoke outputs are for
pipeline validation only and must not be cited as benchmark findings.

## Full Benchmark Reproduction

Install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt pytest
```

Run the measured gold-label benchmark after placing local datasets in `data/raw/`:

```bash
./scripts/run_benchmark.py \
  --financial-phrasebank-path data/raw/FinancialPhraseBank-v1.0.zip \
  --trec-train-path data/raw/trec_train.csv \
  --trec-test-path data/raw/trec_test.csv \
  --banking77-train-path data/raw/banking77_train.csv \
  --banking77-test-path data/raw/banking77_test.csv \
  --output-csv results/benchmark_results.csv

./scripts/summarize_benchmark.py \
  --results results/benchmark_results.csv

./scripts/make_figures.py \
  --results results/benchmark_results.csv \
  --cost-scenario base

./scripts/make_paper_summary_table.py \
  --results results/benchmark_results.csv \
  --output-csv results/paper_core_summary.csv \
  --cost-scenario base
```

Run API-backed LLM annotation only from an environment where `OPENAI_API_KEY` is
set. Start with small budgets and dry runs:

```bash
python scripts/run_llm_strategy_benchmark.py \
  --datasets financial_phrasebank \
  --budgets 5 \
  --seeds 0 \
  --financial-phrasebank-path data/raw/FinancialPhraseBank-v1.0.zip \
  --temperatures 0,0.7 \
  --replicates 3 \
  --dry-run \
  --output-csv results/benchmark_results_with_llm.csv
```

Generate statistical appendix files from row-level annotations. The checked-in combined appendix
summarizes Financial PhraseBank, TREC, and TweetEval Sentiment pilots; local row-level CSVs are
ignored by git because they can include raw API responses.

```bash
python scripts/make_statistical_appendix.py \
  --annotations results/annotation_runs/financial_phrasebank_rowlevel.csv \
    results/annotation_runs/trec_rowlevel_complete.csv \
    results/annotation_runs/tweet_eval_sentiment_rowlevel.csv \
  --output-csv results/statistical_appendix.csv \
  --significance-csv results/statistical_significance.csv
```

## Artifact Contents

- `README.md`: installation, benchmark commands, and claim boundaries.
- `results/README.md`: result schemas and row-level annotation schema.
- `scripts/run_all.sh`: no-network smoke reproduction.
- `scripts/run_benchmark.py`: measured gold-label benchmark runner.
- `scripts/run_llm_annotation.py`: row-level LLM annotation runner.
- `scripts/run_llm_strategy_benchmark.py`: LLM-as-annotator strategy runner.
- `scripts/make_statistical_appendix.py`: confidence intervals, agreement, and
  significance tables from row-level annotations.
- `scripts/validate_benchmark_results.py`: benchmark grid sanity checks.
- `results/statistical_appendix.csv` and `results/statistical_significance.csv`:
  aggregate diagnostics for the current three-dataset row-level reliability pilots.

## Internal Handoff vs. Public Release

For an internal handoff to Natan, the current artifact package is technically ready once the
tracked paper, aggregate result files, and documentation updates are reviewed. The repository
license, public dataset release, and human-subjects documentation can remain deferred until the
team decides to publish or submit the work externally.

## Public Release Blockers Before Closing #14

These items require external decisions or measured data before a public release or formal
submission, and are not completed by the current artifact scaffold:

- choose and add the repository license;
- tag the exact paper-submission commit;
- publish any released dataset on HuggingFace with a dataset card;
- verify licenses and consent for any public human annotations;
- attach IRB approval or exemption documentation if human-subjects annotations
  are released;
- document annotator compensation and demographics for any human study;
- add a Docker image if the venue requires containerized artifacts.

Until those blockers are resolved, PRs should say `Partially addresses #14`
rather than `Fixes #14`.
