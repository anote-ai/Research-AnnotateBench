# IEEE BigData 2026 camera-ready working package

This directory is an isolated copy of the August submission source. The
original submission package is intentionally unchanged. It is also separate
from `paper/` and the JDSE materials.

See `REVIEW_RESPONSE_MATRIX.md` for completed changes and release checks. The
result-generating revision cited by the manuscript is `a4d29d0`.

## Reproduce the corrected LLM row

From the repository root, with `OPENAI_API_KEY` set:

```bash
python scripts/repair_fpb_seed1_annotation.py

python scripts/recompute_llm_dataset_seed.py \
  --dataset financial_phrasebank \
  --seed 1 \
  --source-results results/benchmark_results_with_llm_seed1_2_all_datasets.csv \
  --annotation-cache results/camera_ready/llm_annotations/financial_phrasebank_llm_annotator_seed1.csv \
  --output-results results/camera_ready/benchmark_results_with_llm_seed1_2.csv \
  --financial-phrasebank-path data/raw/FinancialPhraseBank-v1.0.zip

python scripts/make_matched_gold_summary.py \
  --gold-results results/benchmark_results.csv \
  --llm-results results/benchmark_results_with_llm_seed0_all_datasets.csv \
    results/camera_ready/benchmark_results_with_llm_seed1_2.csv \
  --annotation-dir results/llm_annotations results/camera_ready/llm_annotations \
  --output-per-seed results/camera_ready/llm_matched_gold_per_seed.csv \
  --output-summary results/camera_ready/llm_matched_gold_summary.csv
```

Before any future release, confirm that no temporary marker has been introduced:

```bash
rg -n 'pending|RESULTS-COMMIT-PENDING' \
  release/ieee_bigdata_2026_camera_ready/main.tex
```

## Build

From this directory:

```bash
LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Clean auxiliary files without deleting the PDF:

```bash
latexmk -c
```

The bundled `IEEEtran.cls` and `IEEEtran.bst` came from the official CTAN
IEEEtran distribution. The paper uses US letter size, two columns, and the
conference mode of IEEEtran.

## Result validation

From the repository root:

```bash
.venv/bin/python scripts/validate_benchmark_results.py \
  --results results/benchmark_results.csv

.venv/bin/python scripts/make_ieee_robustness_summary.py \
  --baseline-results results/benchmark_results.csv \
  --embedding-results \
    results/benchmark_results_sentence_transformer_logreg_5seed.csv \
  --output-csv results/ieee_downstream_model_robustness.csv

.venv/bin/python scripts/make_ieee_paired_strategy_stats.py \
  --results results/benchmark_results.csv \
  --output-csv results/ieee_paired_strategy_statistics.csv

.venv/bin/python scripts/run_ieee_cumulative_trajectories.py \
  --financial-phrasebank-path data/raw/FinancialPhraseBank-v1.0.zip \
  --trec-train-path data/raw/trec_train.csv \
  --trec-test-path data/raw/trec_test.csv \
  --download-data

.venv/bin/python scripts/run_ieee_pool_scalability.py

.venv/bin/python -m pytest -q
```

The robustness summarizer fails unless both downstream-model grids contain
exactly seeds 0 through 4. The cumulative runner verifies that each larger
budget contains the preceding selection before saving results. Never replace a
partial result file in place.

## Submission scope

- Track: Big Data Benchmarks
- Submission type: Regular Full Paper
- Maximum length: 10 pages including references
- Review model: single-blind
- Repository URL is visible in the manuscript
- Do not submit the same or substantially overlapping paper to ARR or another
  archival venue while it is under review

See `SUBMISSION_CHECKLIST.md` for the author-controlled fields and final gate.
