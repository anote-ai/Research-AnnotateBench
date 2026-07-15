# How Much Labeled Data Do Text-Classification Annotation Strategies Need?

Annotation strategy is usually chosen before a team has much evidence. A project might label the
next random 250 examples, use uncertainty sampling, ask an LLM to label a batch, or keep labeling
until the budget runs out. AnnotateBench turns that planning decision into a benchmark question:
given a fixed label budget, which strategy produces the best downstream classifier, and where does
the cost-performance frontier change?

The current release is deliberately scoped. It is a text-classification annotation-strategy
benchmark, not the full multi-task LLM annotation reliability benchmark described in
`DESIGN_DOC.md`.

## What Is Measured

The main benchmark uses ten public text-classification datasets: Financial PhraseBank, TREC,
Banking77, AG News, SST-2, 20 Newsgroups, Rotten Tomatoes, Yelp Polarity, TweetEval Sentiment, and
Emotion. For each dataset, AnnotateBench simulates annotation by revealing existing gold labels at
budgets 50, 100, 250, 500, and 1000.

The primary grid compares four strategies: random sampling, uncertainty active learning, diversity
active learning, and a hybrid active-learning strategy. Each condition trains the same TF-IDF +
logistic-regression classifier and records macro F1, accuracy, and cost estimates. The full
gold-label grid contains 1,800 measured rows.

The headline result is not that one strategy always wins. Strategy choice is dataset-dependent.
Financial PhraseBank and SST-2 favor uncertainty sampling in the primary table. AG News, Emotion,
Rotten Tomatoes, TREC, TweetEval Sentiment, and Yelp Polarity favor the hybrid strategy. Banking77
and 20 Newsgroups are strongest with random sampling under the primary TF-IDF setup. That variation
is the point: annotation plans should be evaluated against the dataset, budget, model family, and
cost assumptions instead of chosen by habit.

## What The LLM Extension Shows

AnnotateBench also includes a seed-0/1/2 LLM annotator extension. For all ten datasets, `gpt-4o-mini`
labels randomly selected training examples at budgets 50, 100, and 250. The benchmark then trains
the same downstream classifier on those model-generated labels and evaluates on the gold test set.

At budget 250, averaged across the three seeds, LLM label quality and downstream utility are
related but not interchangeable. Yelp Polarity is the strongest case: the LLM-trained classifier
reaches 0.675 macro F1, roughly matching the best gold-label strategy at 0.672, with 0.972 label
accuracy on the selected examples. Financial PhraseBank is close to the gold-label strategy, with
0.589 versus 0.634 macro F1. Other datasets show larger gaps even when label accuracy is high,
which suggests that LLM annotations need to be evaluated as a budgeted strategy rather than treated
as automatic replacements for human or gold labels.

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

## What Comes Next

The next step is not to claim that LLM annotation is solved. It is to make the benchmark more
complete: add measured API costs, code real failure categories, compare more models, and test
review policies that decide when human adjudication is worth the cost.

For now, the safe conclusion is narrower and more useful: annotation strategy selection is an
empirical budget-allocation problem. Learning curves and Pareto frontiers make that problem visible
before a team spends its labeling budget.
