# results/

This directory contains current pilot outputs and should also be the landing spot for future
measured experiments.

Do not commit fabricated "real" results. Synthetic reference curves belong in code comments and
tests, not in result CSVs.

## Current Pilot Results

The existing `pilot_results*.csv` files come from the public-dataset text-classification pilot.
They simulate annotation by revealing gold labels from public datasets, train a downstream
classifier, and report selection-strategy learning curves. These are measured pilot outputs,
but they are not LLM annotation reliability results.

Current pilot columns:

| column | description |
|---|---|
| dataset | public dataset name |
| strategy | selection strategy |
| budget | number of revealed labels |
| seed | strategy-selection seed |
| macro_f1 | measured downstream macro F1 |
| accuracy | measured downstream accuracy |
| cost_scenario | low/base/high cost scenario |
| human_cost_per_label | scenario cost per revealed label |
| selection_cost_per_example | strategy overhead assumption |
| annotation_cost | estimated label cost |
| selection_cost | estimated selection overhead |
| total_cost | estimated total cost |

## Future Reliability Results Schema

For the LLM annotation reliability track in `DESIGN_DOC.md`, write measured CSVs here using a
schema that can support ECE, LARI, failure coding, and review-rate experiments.

Minimum aggregate schema for measured experiment summaries:

| column | type | description |
|---|---|---|
| budget | int | number of labeled or reviewed examples |
| seed | int | random seed for the run |
| f1 | float | measured F1 on a held-out evaluation set |
| cost_usd | float | measured or audited cost for the run |
| model_type | str | downstream model or LLM annotator family |
| dataset_name | str | concrete dataset used |
| notes | str | optional anomalies, partial runs, or caveats |

Recommended row-level schema:

| column | type | description |
|---|---|---|
| dataset_name | str | concrete dataset used |
| task_type | str | task type from the reliability benchmark |
| example_id | str | stable example identifier |
| model_name | str | LLM annotator name/version |
| prompt_version | str | prompt template identifier |
| gold_label | str | human/gold label |
| predicted_label | str | LLM annotation |
| confidence | float | model confidence in [0, 1] |
| correct | bool | whether prediction matches gold label |
| cost_usd | float | measured API or annotation cost |
| failure_category | str | optional taxonomy category for incorrect examples |
| notes | str | optional anomalies or run details |

Aggregated files may also include `budget`, `seed`, `f1`, `ece`, `lari`, `review_rate`,
`model_type`, and `notes`, but they should be derived from row-level measured outputs rather
than hand-entered as projected results.
