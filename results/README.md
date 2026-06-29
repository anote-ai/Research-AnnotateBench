# results/

This directory is the intended landing spot for **real, measured** experiment output — it is
empty of data as of this writing. No real annotation-strategy experiments have been run yet;
see `PAPER_DRAFT.md` Section 4.3 and `BLOG.md` for the current status.

The reference/illustrative curves currently used by `src/annotatebench/data.py` to validate the
framework's math (Pareto frontier, curve fitting, cost-efficiency) live in code, not here, and
are explicitly labeled as non-measured.

## Expected schema for real results

When a real experimental run is performed, results should be written here as CSV files named:

```
results/<task>_<strategy>.csv
```

e.g. `results/classification_uncertainty_al.csv`.

Each file should contain one row per (budget, seed):

| column | type | description |
|---|---|---|
| budget | int | number of labeled examples used (one of BUDGET_LEVELS) |
| seed | int | random seed for this run |
| f1 | float | measured F1 on a held-out evaluation set |
| cost_usd | float | actual measured cost (API spend and/or human annotator pay) |
| model_type | str | model trained/evaluated (e.g. "bert-base", "gpt-4o") |
| dataset_name | str | concrete dataset used for this task type |
| notes | str | optional: anomalies, partial runs, etc. |

A loader for this schema (e.g. `load_results_csv(path) -> LearningCurve`, averaging across
seeds per budget) does not exist yet and is a good first contribution — see `PAPER_DRAFT.md`
Section 4.3 for the full list of steps needed to go from this framework to real numbers.

Do not commit fabricated numbers into this directory under any circumstances; an empty
directory with this README is preferable to invented data.
