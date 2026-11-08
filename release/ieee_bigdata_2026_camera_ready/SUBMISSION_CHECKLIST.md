# IEEE BigData 2026 submission checklist

## Official constraints

- [x] Regular Full Paper source uses `\documentclass[conference]{IEEEtran}`
- [x] Target topic is **Big Data Benchmarks**
- [x] Single-blind manuscript retains author names
- [x] Abstract is one paragraph, self-contained, and below 250 words
- [x] Five IEEE keywords are present
- [x] Final PDF is no more than 10 pages including references
- [x] PDF is US letter, two-column, searchable, and has all fonts embedded
- [x] No undefined references, missing citations, serious overfull boxes,
      passwords, attachments, crop marks, or non-template page numbers
- [x] Uploaded PDF downloaded from CyberChair and matched to the local final
      PDF by SHA-256

## Evidence and reproducibility

- [x] Gold-label grid distinguishes 1,000 model conditions from 3,000
      cost-scenario records
- [x] Main results are based on ten datasets, four strategies, five budgets,
      and five seeds
- [x] Primary budget conditions are described as independent, and the new
      three-dataset cumulative experiment is reported separately
- [x] Confidence intervals are not presented as formal significance tests;
      paired bootstrap/effect-size/sign-flip results are explicitly exploratory
- [x] Human cost assumptions and LLM token-cost scope are disclosed
- [x] Diversity-selection scaling limitation is disclosed
- [x] Dataset citations are included for all ten datasets
- [x] No unsupported preregistration or predeclaration claim remains
- [x] `scripts/validate_benchmark_results.py` passes
- [x] Full project test suite passes
- [x] Sentence-transformer seeds 3 and 4 complete for all ten datasets
- [x] Five-seed downstream-model summary passes its completeness check
- [x] Three-dataset cumulative grid contains 300 complete, unique records
- [x] AG News pool-scaling grid contains 36 complete, unique records through
      10,000 candidate examples
- [x] Novelty-positioning table directly distinguishes DataPerf, DataComp,
      OpenDataVal, and AnnotateBench
- [x] Every number in the abstract, tables, and captions matches a generated
      result CSV in the repository
- [x] Final source bundle includes only the files needed to build the paper

## Author confirmation required

Do not infer or silently fill any of these fields.

- [x] Title approved by Aitong Zhang, Natan Vidra, and Spurthi Setty
- [x] Author order approved
- [x] Exact affiliation(s) approved
- [x] Email addresses approved
- [ ] ORCID identifiers confirmed or intentionally omitted
- [x] Corresponding/submitting author confirmed: Aitong Zhang
- [ ] Presenting author confirmed
- [ ] Registration and travel responsibility confirmed
- [ ] Funding acknowledgements and grant numbers confirmed
- [ ] Conflicts of interest entered in the submission system
- [x] All authors approve the final PDF
- [x] All authors confirm that there is no simultaneous archival submission
- [ ] All authors agree to the in-person presentation arrangement in Phoenix

## Dates and submission gate

- Internal upload target: **August 20, 2026, Tokyo time**
- Official deadline: **August 21, 2026**
- Notification: **October 24, 2026**
- Camera-ready: **November 14, 2026**
- Conference: **December 14--17, 2026, Phoenix**

The final CyberChair submission must not be executed until every item in
“Author confirmation required” is resolved.

Official links:

- https://bigdataieee.org/BigData2026/calls/papers/
- https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/
- https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/meet-ieee-xplore-requirements/
