# results/

This directory contains current benchmark outputs and should also be the landing spot for future
measured experiments.

Do not commit fabricated "real" results. Synthetic reference curves belong in code comments and
tests, not in result CSVs.

## Current Benchmark Results

The main measured gold-label output is `benchmark_results.csv`, a ten-dataset public text-classification
benchmark. It simulates annotation by revealing gold labels from public datasets, trains a
downstream classifier, and reports selection-strategy learning curves. These are measured
classification benchmark outputs, but they are not LLM annotation reliability results.

The row-level `benchmark_results.csv` currently contains the full measured gold-label grid:
10 datasets x 4 selection strategies x 5 label budgets x 3 seeds x 3 cost scenarios, for
1,800 rows. The summary files `benchmark_best_overall.csv`, `benchmark_best_by_strategy.csv`,
`budget_recommendations.csv`, and `paper_core_summary.csv` are derived from the same ten-dataset
benchmark outputs.

`benchmark_results_with_llm.csv` is the intended output for the API-backed `llm_annotator`
strategy. It is generated separately by `scripts/run_llm_strategy_benchmark.py` after
`OPENAI_API_KEY` is set in the terminal.

`benchmark_results_with_llm_seed0_all_datasets.csv` contains the ten-dataset seed-0 LLM annotator
grid for budgets 50, 100, and 250. `benchmark_results_with_llm_seed1_2_all_datasets.csv` contains
the matching seed-1 and seed-2 grid. `llm_strategy_seed0_summary.csv`,
`llm_strategy_seed1_2_summary.csv`, and `llm_strategy_seed0_1_2_summary.csv` compare those rows
against the best matching gold-label strategy from `benchmark_results.csv`.

`llm_api_cost_seed1_2_summary.csv` is an auxiliary API-cost summary derived from recorded row-level
token usage in local `results/llm_annotations/*seed1.csv` and `*seed2.csv` logs. It uses the
published `gpt-4o-mini` prices checked on 2024-07-18 ($0.15 per 1M input tokens and $0.60 per
1M output tokens). The seed-0 LLM strategy run predates the complete token-logging schema, so this
cost summary is not used to place LLM rows on the main human-label Pareto frontiers.

`statistical_appendix.csv` and `statistical_significance.csv` are the combined aggregate outputs
for the current row-level LLM reliability pilots on Financial PhraseBank, TREC, and TweetEval
Sentiment. The per-dataset appendix/significance files are retained for traceability.

`benchmark_results_sentence_transformer_logreg.csv` and the
`downstream_model_comparison_*.csv` files contain the measured sentence-transformer + logistic
regression robustness check. Treat these as sensitivity-analysis artifacts; the primary benchmark
table uses `benchmark_results.csv`.

The older `pilot_results*.csv` files are retained as per-dataset pilot outputs for the original
three-dataset run. Prefer `benchmark_results.csv` for paper tables and figures.

Current benchmark columns:

| column | description |
|---|---|
| dataset | public dataset name |
| strategy | selection strategy |
| budget | number of revealed labels |
| seed | strategy-selection seed |
| downstream_model | optional downstream classifier identifier in newly generated comparison CSVs |
| embedding_model | optional embedding model name when `downstream_model` uses sentence-transformer embeddings |
| macro_f1 | measured downstream macro F1 |
| accuracy | measured downstream accuracy |
| cost_scenario | calibrated low/base/high cost scenario |
| cost_source | public pricing source used for the scenario |
| cost_source_url | URL for the public pricing source |
| cost_checked_at | date when the pricing source was checked |
| human_cost_per_label | calibrated cost per revealed label |
| selection_cost_per_example | strategy overhead assumption |
| annotation_cost | estimated label cost |
| selection_cost | estimated selection overhead |
| total_cost | estimated total cost |

LLM strategy outputs use the same core columns and may additionally include `llm_label_accuracy`,
`llm_label_macro_f1`, `llm_ece`, `llm_lari`, `model_name`, and `prompt_version`. Current
seed-0/1/2 LLM strategy rows use zero-estimate cost placeholders in the benchmark-style result
files. Use `llm_api_cost_seed1_2_summary.csv` for the auxiliary recorded-token API cost analysis;
do not include the LLM rows in the main human-label Pareto frontier.

Paper core summary columns:

| column | description |
|---|---|
| dataset | public dataset name |
| best_strategy | strategy with the highest mean macro F1 under the selected cost scenario |
| best_budget | label budget for that strategy-budget point |
| mean_macro_f1 | mean macro F1 across strategy-selection seeds |
| std_macro_f1 | sample standard deviation of macro F1 across seeds |
| mean_accuracy | mean accuracy across seeds |
| mean_total_cost | mean estimated cost under the selected cost scenario |
| n_seeds | number of unique seeds represented |
| cost_scenario | cost scenario used for the table |

LLM summary columns:

| column | description |
|---|---|
| macro_f1 | downstream classifier macro F1 trained on LLM-generated labels |
| best_gold_macro_f1 | best downstream macro F1 among gold-label strategies at the same dataset, budget, seed, and cost scenario |
| macro_f1_gap_vs_best_gold | `macro_f1 - best_gold_macro_f1` |
| best_gold_strategy | gold-label strategy that achieved `best_gold_macro_f1` |
| llm_label_accuracy | agreement between LLM labels and gold labels on the selected training examples |
| llm_lari | calibration-adjusted LLM label quality diagnostic |

LLM API cost summary columns:

| column | description |
|---|---|
| dataset | public dataset name |
| seed | LLM annotator strategy seed with recorded token usage |
| model_name | API model used for annotation |
| rows_total | row-level annotation records in the local log |
| rows_with_recorded_tokens | records with nonzero API token usage |
| input_tokens | total recorded API input tokens |
| output_tokens | total recorded API output tokens |
| total_tokens | total recorded API tokens |
| input_usd_per_million_tokens | dated input-token price used for the estimate |
| output_usd_per_million_tokens | dated output-token price used for the estimate |
| api_cost_usd | estimated API cost from recorded usage and dated token prices |
| price_checked_at | date associated with the token-pricing assumption |
| price_source_url | public source for the token prices |

## Reliability Results Schema

For the LLM annotation reliability track in `DESIGN_DOC.md`, measured row-level CSVs should use a
schema that can support ECE, LARI, failure coding, and review-rate experiments. The current release
includes aggregate pilot outputs; local row-level runs are ignored by git because they may include
raw API responses.

Minimum aggregate schema for measured experiment summaries:

| column | type | description |
|---|---|---|
| budget | int | number of labeled or reviewed examples |
| seed | int | random seed for the run |
| f1 | float | measured F1 on a held-out evaluation set |
| input_tokens | int | total API input tokens returned in response usage |
| output_tokens | int | total API output tokens returned in response usage |
| total_tokens | int | total API tokens returned in response usage |
| cost_usd | float | measured or audited cost for the run |
| model_type | str | downstream model or LLM annotator family |
| dataset_name | str | concrete dataset used |
| notes | str | optional anomalies, partial runs, or caveats |

Recommended row-level schema:

| column | type | description |
|---|---|---|
| run_id | str | stable identifier for the annotation run |
| dataset_name | str | concrete dataset used |
| split | str | dataset split annotated, such as `train` or `test` |
| seed | int | random seed used for dataset loading or selection |
| task_type | str | task type from the reliability benchmark |
| example_id | str | stable example identifier |
| difficulty_bucket | str | optional easy/medium/hard or derived difficulty bucket |
| annotator_type | str | `llm` or `human` |
| annotator_id | str | stable annotator identity, including model/temperature/replicate for LLMs |
| model_name | str | LLM annotator name/version |
| temperature | float | LLM sampling temperature |
| prompt_version | str | prompt template identifier |
| replicate_id | int | repeated annotation index for the same model and temperature |
| gold_label | str | human/gold label |
| predicted_label | str | LLM annotation |
| confidence | float | model confidence in [0, 1] |
| correct | bool | whether prediction matches gold label |
| input_tokens | int | API input tokens for the example |
| output_tokens | int | API output tokens for the example |
| total_tokens | int | API total tokens for the example |
| cost_usd | float | measured API or annotation cost |
| failure_category | str | optional taxonomy category for incorrect examples |
| notes | str | optional anomalies or run details |

Aggregated files may also include `budget`, `seed`, `f1`, `ece`, `lari`, `review_rate`,
`model_type`, and `notes`, but they should be derived from row-level measured outputs rather
than hand-entered as projected results.

Use `scripts/make_statistical_appendix.py` on row-level annotation files to produce bootstrap
confidence intervals, agreement rows when multiple annotators are present, and paired significance
tests for annotator comparisons.
