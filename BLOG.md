# How Many Labels Do You Actually Need?

If a team only has budget to label 250 examples, what should it do? Pick examples at random, use
active learning, ask an LLM to label them, or keep labeling until the budget runs out?

AnnotateBench turns that planning choice into a benchmark question: given a fixed label budget,
which annotation strategy produces the best downstream classifier, and where does the
cost-performance frontier change?

## Key Takeaways

- No annotation strategy wins across every dataset.
- LLM-generated labels can be useful, but high label accuracy does not always translate into the
  same downstream classifier performance as gold labels.
- Cost frontiers are more useful than raw accuracy when the real decision is how to spend a
  labeling budget.
- This release is a text-classification annotation-strategy benchmark, not the full multi-task LLM
  annotation reliability benchmark described in `DESIGN_DOC.md`.

## The Problem

Annotation strategy is often chosen before a team has much evidence. Random sampling is simple.
Uncertainty sampling sounds efficient. Diversity sampling can avoid labeling the same kind of
example repeatedly. LLM labels are tempting because they are fast and cheap to request.

The hard part is that "best" depends on the dataset, the model family, the budget, and the cost
assumptions. A strategy that looks good at 50 labels may not be the best use of 500 labels. A
strategy that works for binary sentiment may not work for a many-class intent dataset.

AnnotateBench is built around that practical question: before spending the full annotation budget,
can we estimate which strategy is likely to buy the most downstream performance?

## What AnnotateBench Measures

The main benchmark uses ten public text-classification datasets: Financial PhraseBank, TREC,
Banking77, AG News, SST-2, 20 Newsgroups, Rotten Tomatoes, Yelp Polarity, TweetEval Sentiment, and
Emotion. For each dataset, AnnotateBench simulates annotation by revealing existing gold labels at
budgets 50, 100, 250, 500, and 1000.

The primary grid compares four strategies: random sampling, uncertainty active learning, diversity
active learning, and a hybrid active-learning strategy. Each condition trains the same TF-IDF +
logistic-regression classifier and records macro F1, accuracy, and cost estimates. The full
gold-label grid contains 1,800 measured rows.

![Pareto frontier for TREC](figures/pareto_trec.png)

The headline result is not that one strategy always wins. Strategy choice is dataset-dependent.
Financial PhraseBank and SST-2 favor uncertainty sampling in the primary table. AG News, Emotion,
Rotten Tomatoes, TREC, TweetEval Sentiment, and Yelp Polarity favor the hybrid strategy. Banking77
and 20 Newsgroups are strongest with random sampling under the primary TF-IDF setup.

That variation is the point: annotation plans should be evaluated against the dataset, budget,
model family, and cost assumptions instead of chosen by habit.

## What the LLM Extension Shows

AnnotateBench also includes a seed-0/1/2 LLM annotator extension. For all ten datasets,
`gpt-4o-mini` labels randomly selected training examples at budgets 50, 100, and 250. The benchmark
then trains the same downstream classifier on those model-generated labels and evaluates on the
gold test set.

At budget 250, averaged across the three seeds, LLM label quality and downstream utility are
related but not interchangeable. Yelp Polarity is the strongest case: the LLM-trained classifier
reaches 0.675 macro F1, roughly matching the best gold-label strategy at 0.672, with 0.972 label
accuracy on the selected examples. Financial PhraseBank is close to the gold-label strategy, with
0.589 versus 0.634 macro F1. Other datasets show larger gaps even when label accuracy is high.

![Learning curve for Financial PhraseBank](figures/learning_curve_financial_phrasebank.png)

The practical takeaway is that LLM annotations should be evaluated as a budgeted strategy, not
treated as automatic replacements for human or gold labels. A team should measure whether model
labels improve its downstream system, not only whether a sample of labels looks accurate.

These LLM rows currently use zero-estimate API cost placeholders, so they should not be read as
part of the human-label Pareto frontier yet.

## Reliability Pilots

The release includes small row-level reliability pilots for Financial PhraseBank, TREC, and
TweetEval Sentiment. Each pilot annotates 100 held-out test examples with `gpt-4o-mini` at
temperatures 0 and 0.7, with three replicates per temperature, for 600 annotations per dataset.

Financial PhraseBank is the high-agreement case: 0.990 accuracy, 0.987 macro F1, ECE 0.161, and
LARI 0.828. TREC is harder: 0.792 accuracy, 0.562 macro F1, ECE 0.084, and LARI 0.514 when
schema-external predictions are counted. TweetEval Sentiment sits between them with 0.782
accuracy, 0.766 macro F1, ECE 0.026, and LARI 0.745.

These pilots are diagnostic evidence, not a completed reliability benchmark. They show how the
row-level schema can support calibration, agreement, and failure-pattern analysis, but they do not
replace a larger multi-model, human-reviewed study.

## What This Does Not Claim

This release does not show that LLM annotation is solved. It does not include human adjudication,
coded failure categories, a review-rate optimizer, or a completed cross-model reliability study. It
also does not claim measured results for NER, QA, summarization, relation extraction, or other
non-classification tasks.

The safe conclusion is narrower and more useful: annotation strategy selection is an empirical
budget-allocation problem. Learning curves and Pareto frontiers make that problem visible before a
team spends its labeling budget.

## Where To Go Next

- Use `README.md` for setup, benchmark commands, and reproducibility instructions.
- Use `results/README.md` for result schemas and claim boundaries.
- Use `PAPER_DRAFT.md` for the research framing and planned paper structure.
- Use `DESIGN_DOC.md` for the broader benchmark vision that is not fully implemented yet.
