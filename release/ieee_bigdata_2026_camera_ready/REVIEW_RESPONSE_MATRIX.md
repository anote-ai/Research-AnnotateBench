# IEEE BigData 2026 Camera-Ready Response Matrix

This matrix maps every actionable review point to the camera-ready manuscript or artifact. Reviewer 4 raised no requested change.

| Reviewer | Comment | Resolution | Evidence |
|---|---|---|---|
| 1, 3 | Table VI mixes annotation source, example selection, and best-result selection | Added a gold classifier on the identical LLM-selected examples and retained best selector only as a separate retrospective reference | `main.tex`, Table VI; `results/camera_ready/llm_matched_gold_*.csv` |
| 1, 3 | “Class-covering” uncertainty seed conflicts with code | Replaced with the exact random seed size, budget accounting, single-batch entropy selection, and random fallback | `main.tex`, Section III-B |
| 1 | Table V changes only the downstream representation | Renamed as downstream-model sensitivity and stated that acquisition remains in TF-IDF space | `main.tex`, Section V-D and Table V caption |
| 1, 3 | Three-seed LLM spread is hidden | Table VI reports paired-gap mean and sample SD | `main.tex`, Table VI |
| 1, 3 | Five-seed intervals use a normal critical value | Recomputed with Student-t, df=4; top-two overlap is now 7/10 | `results/camera_ready/paper_core_summary_t.csv`; `top_two_interval_overlap.csv` |
| 1, 3 | Cost conclusions may depend on modeled overhead | Audited 1x, 2x, 5x, and 10x overhead; all ten frontiers are stable | `results/camera_ready/selection_overhead_sensitivity.csv` |
| 1 | Add modest LLM review cost | Added 10% review at $0.0952 per item as a cost-only sensitivity | `main.tex`, Fig. 3 and Section V-E |
| 1 | Print full base human price | Changed $0.095 to $0.0952 and stated that tables use unrounded inputs | `main.tex`, Section III-C |
| 1 | Report LLM settings and off-schema handling | Added response model snapshot, prompt ID, temperature, JSON mode, and deterministic invalid-output fallback | `main.tex`, Section III-B; `results/camera_ready/llm_off_schema_audit.csv` |
| 1 | Target costs and best configurations are retrospective | Explicitly described them as evaluated-grid summaries, not deployable rules without validation/test feedback | `main.tex`, Section V-A |
| 1 | Scope the sub-0.5-second claim | Limited it to selector wall time for AG News, budget 100, and the reported machine | `main.tex`, Section V-C |
| 2 | Abstract is dense | Separated the benchmark contribution from secondary experiments | `main.tex`, Abstract |
| 2 | Sharpen distinction from adjacent benchmarks | Stated the joint acquisition/source/cost/downstream-utility capability | `main.tex`, Section II-A |
| 2 | Add a metrics summary | Added a compact definitions table | `main.tex`, metrics table |
| 2 | Actionable Insights duplicates conclusion | Folded the five principles into the conclusion | `main.tex`, Conclusion |
| 2, 3 | Limited scale, simple models, missing recent selectors | Expanded limitations without adding unrequested experiments | `main.tex`, Limitations |

## Completed release gates

- Replaced the one selected FPB seed-1 dry-run row with a production response from `gpt-4o-mini-2024-07-18`.
- Regenerated the FPB LLM rows, matched-gold summaries, and Table VI.
- Cited result-generating commit `a4d29d0` in the manuscript.
- Verified the analysis regressions and compiled the six-page camera-ready PDF.
