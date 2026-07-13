# AnnotateBench Submission Checklist

Use this checklist for the final submission package.

## Required Items

- Final research paper source: `paper/main.tex`
- Research code repository link: <https://github.com/anote-ai/research-annotatebench>
- Research blog post: `BLOG.md`
- Benchmark status summary: `BENCHMARK_STATUS.md`
- Result inventory: `results/README.md`
- Primary gold-label benchmark results: `results/benchmark_results.csv`
- Paper-ready result table: `results/paper_core_summary.csv`
- Budget recommendations: `results/budget_recommendations.csv`
- Seed-0 LLM annotator summary: `results/llm_strategy_seed0_summary.csv`

## Final Verification

- Run `.venv/bin/python scripts/validate_benchmark_results.py --results results/benchmark_results.csv`.
- Run `.venv/bin/python -m pytest -q`.
- Compile the paper from the `paper/` directory when LaTeX is available:
  `cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`.
- Confirm README links point to the paper, blog, benchmark status, and result inventory.
- Confirm the final pushed commit is the one used for the repository link.

## Short Submission Blurb

AnnotateBench is a pilot benchmark for annotation budget planning in NLP text classification.
It compares random sampling, active-learning strategies, and an API-backed LLM annotator path
through learning curves, cost-performance frontiers, and budget recommendations across public
classification datasets. The current submission reports a measured ten-dataset gold-label
simulation benchmark and a seed-0 LLM annotator extension, while explicitly leaving full LLM
annotation reliability, human adjudication, and non-classification tasks to future work.
