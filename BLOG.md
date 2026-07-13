# How Much Labeled Data Do Annotation Strategies Need?

Most NLP teams eventually face the same practical question: if we can afford to label only a
small batch of examples, which examples should we choose?

AnnotateBench is a benchmark for that decision. Instead of reporting one model score after one
fixed training set, it compares annotation strategies across label budgets. The current pilot
asks how downstream classification quality changes when a team can label 50, 100, 250, 500, or
1000 examples.

The core idea is simple: annotation should be evaluated as a budget-allocation problem. A strategy
is useful only if it improves model quality enough to justify the labeling and selection cost.

## What AnnotateBench Measures

The current repository contains a ten-dataset text-classification benchmark. It uses public
datasets with existing gold labels and simulates annotation by revealing labels from the training
set under different selection policies. That lets us study annotation strategy without collecting
new human labels or making unsupported claims about human annotation behavior.

The measured gold-label benchmark covers:

- Financial PhraseBank, TREC, Banking77, AG News, SST-2, 20 Newsgroups, Rotten Tomatoes,
  Yelp Polarity, TweetEval Sentiment, and Emotion;
- five label budgets: 50, 100, 250, 500, and 1000 examples;
- four gold-label selection strategies: random sampling, uncertainty active learning, diversity
  active learning, and a hybrid active-learning strategy;
- three strategy-selection seeds and three calibrated cost scenarios;
- a shared TF-IDF logistic-regression downstream classifier.

The repository also includes an API-backed `llm_annotator` path. In the current pilot, that path
labels selected training examples with `gpt-4o-mini`, trains the same downstream classifier on
those model-generated labels, and compares the result against the best gold-label strategy at the
same dataset, budget, and seed.

## Why This Is Useful

Annotation planning is usually discussed in vague terms: label more data, use active learning,
try LLM labels, or collect a small seed set. AnnotateBench makes those choices concrete. For each
dataset and budget, it records the downstream macro F1, estimated annotation cost, and whether a
strategy-budget point lies on the cost-performance Pareto frontier.

That framing helps answer questions like:

- Is active learning better than random sampling for this dataset?
- Does the gain appear at 100 labels, 250 labels, or only near 1000?
- Which strategy reaches a target macro F1 at the lowest estimated cost?
- When do LLM-generated labels help downstream training, and when do they underperform gold
  labels?

## Early Results

The gold-label benchmark produces 1,800 measured rows: 10 datasets x 4 strategies x 5 budgets x
3 seeds x 3 cost scenarios.

The main takeaway is that there is no universal winner. Random sampling is competitive or best on
some topic and intent datasets, while active-learning variants help on others. In the paper-ready
summary table, uncertainty sampling is strongest for Financial PhraseBank and SST-2, random
sampling is strongest for Banking77 and 20 Newsgroups, and the hybrid strategy is strongest for
AG News, Emotion, Rotten Tomatoes, TREC, TweetEval Sentiment, and Yelp Polarity.

The budget recommendations are also dataset-dependent. Financial PhraseBank reaches macro F1
0.75 with uncertainty sampling at 500 labels, but does not reach 0.80 in the current grid. Yelp
Polarity reaches 0.80 with diversity sampling at 1000 labels. Harder datasets such as Emotion,
TweetEval Sentiment, and 20 Newsgroups remain weak under the simple TF-IDF classifier and current
1000-label cap, which is an important reminder: annotation strategy cannot compensate for every
task/model mismatch.

The seed-0 LLM annotator run adds another useful signal. At 250 labels, LLM-generated labels are
strong on some sentiment-style tasks: Financial PhraseBank reaches 0.626 macro F1 downstream
against 0.642 for the best matching gold-label strategy, and Yelp Polarity reaches 0.681 against
0.639. On high-cardinality or schema-sensitive tasks such as Banking77, Emotion, and 20 Newsgroups,
the LLM annotator is much less effective at this budget.

## What Is Real Today

The current empirical claims are intentionally narrow:

- The gold-label benchmark is a real measured text-classification benchmark over ten public
  datasets.
- The learning curves, Pareto plots, budget recommendations, and paper summary tables are derived
  from those measured results.
- The LLM annotator path has a measured seed-0 ten-dataset run for budgets 50, 100, and 250.
- The code includes ECE, LARI, and failure-taxonomy primitives for future LLM annotation
  diagnostics.

The synthetic curves in `src/annotatebench/data.py` are different. They are hand-specified
reference curves used to test the learning-curve and cost-analysis utilities. They should not be
cited as empirical findings.

## What Is Still Missing

AnnotateBench is not yet the full LLM annotation reliability benchmark described in
`DESIGN_DOC.md`.

The project does not yet include human adjudication, multi-model agreement, expanded multi-seed
LLM annotator runs, failure-taxonomy coding of real LLM errors, measured human annotation
invoices, or a review-rate optimizer. The current benchmark is also classification-only. NER, QA,
summarization, and relation extraction need task-specific data structures, annotation units, and
metrics before they can be claimed.

The next research step is to expand the LLM annotator grid, report ECE and LARI across more runs,
code a sample of LLM failures, and compare against at least one stronger downstream learner.

## Links

- Code repository: <https://github.com/anote-ai/research-annotatebench>
- Paper source: [`paper/main.tex`](paper/main.tex)
- Benchmark status: [`BENCHMARK_STATUS.md`](BENCHMARK_STATUS.md)
- Result files: [`results/README.md`](results/README.md)
