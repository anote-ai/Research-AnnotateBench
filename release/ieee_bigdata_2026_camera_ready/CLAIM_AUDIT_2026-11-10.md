# Camera-ready claim audit — November 10, 2026

Scope: current camera-ready source against local code, raw annotation records,
and result CSVs. This is not an independent rerun of all classifiers or API calls.

## Corrected in this audit

- Public availability: initially blocked by repository permissions. After write
  access was restored on November 10, analysis revision `a4d29d0` was pushed on
  `feature/benchmark-robustness-scalability`; remote branch verification succeeded.
  The manuscript links that branch explicitly rather than the older default branch.
- Gold labels: uncertainty/hybrid use initial labels during acquisition, rather
  than revealing every label only after selection.
- Invalid outputs: annotation-field fallback occurs after JSON decoding;
  decoding errors propagate. Two invalid-label records are documented in
  available main-extension seed-1/2 caches. Seed-0 row responses are unavailable.
- Matched gold: budget-250 random gold results are reused after deterministic
  index matching, rather than claiming a new matched-gold training run.
- High human-cost input is $0.4284, not $0.428.
- Existing paired-bootstrap CSV is exactly reproduced with 20,000 resamples
  (not 10,000); five intervals exclude zero (not four). Holm conclusions remain.
- Learning curves show sample-SD error bars, not error bands or t intervals.
- Downstream-model table entries maximize over the evaluated strategy/budget grid.
- Historical batch API-cost summaries are retained after FPB label repair;
  a newly measured corrected-batch cost is not claimed.
- Separate reliability pilots describe annotation errors, not causes of training
  gaps. Their repeated items, different FPB prompt, and different treatment of
  schema-external predictions are disclosed.
- Conclusion describes evaluation practices rather than validated deployment rules.

## Freshly checked

- Gold grid: 3,000 rows / 1,000 unique model conditions.
- Available LLM caches pass the selected-ID/dry-run audit; paired summaries
  reproduce from seed-level inputs. Eight datasets have negative mean gaps.
- Main best-score/t-interval numbers and LLM table values match generated CSVs.
- Downstream summary recomputed: eight improved best scores; five matching
  best-strategy identities; rank correlations range from -0.4 to 1.0.
- Cumulative grid has 300 rows; winners agree in 11 of 15 dataset/budget pairs.
- Runtime grid has 36 rows; reported endpoint means agree with raw CSVs.
- All overhead-frontier flags are stable; cost-validation mean APE is 2.4414%.
- Reliability accuracy/F1 recomputed from 600 rows / 100 unique items per dataset;
  supplied Bonferroni tables contain no significant comparisons.

## Outstanding; do not mark complete

- Publish any subsequent PDF eXpress or coauthor changes before final submission.
- IEEE PDF eXpress validation, coauthor approval, eCF, final portal upload,
  registration/payment and receipt upload.
- Full pytest has not passed in the current environment; previous targeted
  regression checks are not a substitute for a full-suite pass.
- No fresh end-to-end classifier rerun or independent dataset-version recovery
  was performed in this audit. Seed-0 matching relies on deterministic reconstruction.
