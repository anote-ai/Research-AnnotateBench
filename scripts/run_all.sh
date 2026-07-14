#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="results/artifact_smoke"
FIG_DIR="figures/artifact_smoke"
DATA_DIR="$OUT_DIR/data"
mkdir -p "$OUT_DIR" "$FIG_DIR" "$DATA_DIR"

TRAIN_CSV="$DATA_DIR/mini_train.csv"
TEST_CSV="$DATA_DIR/mini_test.csv"

cat > "$TRAIN_CSV" <<'CSV'
text,label
profits increased,positive
shares rose,positive
earnings beat estimates,positive
revenue growth accelerated,positive
the outlook improved,positive
losses widened,negative
sales fell,negative
margins declined,negative
the warning hurt shares,negative
demand weakened,negative
CSV

cat > "$TEST_CSV" <<'CSV'
text,label
profits beat expectations,positive
shares gained,positive
losses hurt outlook,negative
sales declined,negative
CSV

python scripts/run_pilot.py \
  --dataset csv \
  --train-csv "$TRAIN_CSV" \
  --test-csv "$TEST_CSV" \
  --dataset-name artifact_smoke \
  --budgets 10 \
  --seeds 0 \
  --cost-scenarios base \
  --output-csv "$OUT_DIR/benchmark_results.csv"

python scripts/summarize_benchmark.py \
  --results "$OUT_DIR/benchmark_results.csv" \
  --cost-scenario base \
  --best-overall-csv "$OUT_DIR/benchmark_best_overall.csv" \
  --best-by-strategy-csv "$OUT_DIR/benchmark_best_by_strategy.csv"

python scripts/make_paper_summary_table.py \
  --results "$OUT_DIR/benchmark_results.csv" \
  --output-csv "$OUT_DIR/paper_core_summary.csv" \
  --cost-scenario base

python scripts/make_figures.py \
  --results "$OUT_DIR/benchmark_results.csv" \
  --figures-dir "$FIG_DIR" \
  --cost-scenario base

python scripts/run_llm_annotation.py \
  --dataset financial_phrasebank \
  --financial-phrasebank-path "$TRAIN_CSV" \
  --split test \
  --limit 2 \
  --temperatures 0,0.7 \
  --replicates 2 \
  --dry-run \
  --output-csv "$OUT_DIR/llm_annotations.csv" \
  --summary-csv "$OUT_DIR/llm_annotation_summary.csv"

python scripts/make_statistical_appendix.py \
  --annotations "$OUT_DIR/llm_annotations.csv" \
  --output-csv "$OUT_DIR/statistical_appendix.csv" \
  --significance-csv "$OUT_DIR/statistical_significance.csv" \
  --n-resamples 100

echo "Artifact smoke reproduction complete."
echo "Outputs: $OUT_DIR"
echo "Figures: $FIG_DIR"
