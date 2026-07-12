from __future__ import annotations

import collections
import re
import string
from typing import Sequence

from sklearn.metrics import f1_score


EntitySpan = tuple[str, int, int]


def bio_tags_to_spans(labels: Sequence[str]) -> set[EntitySpan]:
    """Convert BIO labels into inclusive-exclusive entity spans."""
    spans: set[EntitySpan] = set()
    entity_type: str | None = None
    start: int | None = None

    for idx, label in enumerate(list(labels) + ["O"]):
        prefix, current_type = _split_bio_label(label)
        starts_new = prefix == "B" or (
            prefix == "I" and (entity_type is None or current_type != entity_type)
        )
        closes_current = prefix == "O" or starts_new

        if closes_current and entity_type is not None and start is not None:
            spans.add((entity_type, start, idx))
            entity_type = None
            start = None
        if starts_new:
            entity_type = current_type
            start = idx

    return spans


def entity_level_f1(
    gold_sequences: Sequence[Sequence[str]],
    predicted_sequences: Sequence[Sequence[str]],
) -> float:
    if len(gold_sequences) != len(predicted_sequences):
        raise ValueError("Gold and predicted NER sequence counts must match.")

    true_positive = 0
    gold_total = 0
    predicted_total = 0
    for gold, predicted in zip(gold_sequences, predicted_sequences):
        if len(gold) != len(predicted):
            raise ValueError("Gold and predicted NER token counts must match.")
        gold_spans = bio_tags_to_spans(gold)
        predicted_spans = bio_tags_to_spans(predicted)
        true_positive += len(gold_spans & predicted_spans)
        gold_total += len(gold_spans)
        predicted_total += len(predicted_spans)

    if gold_total == 0 and predicted_total == 0:
        return 1.0
    precision = true_positive / max(predicted_total, 1)
    recall = true_positive / max(gold_total, 1)
    return _f1(precision, recall)


def qa_exact_match(prediction: str, gold_answers: Sequence[str]) -> float:
    return float(
        any(_normalize_answer(prediction) == _normalize_answer(answer) for answer in gold_answers)
    )


def qa_token_f1(prediction: str, gold_answers: Sequence[str]) -> float:
    if not gold_answers:
        raise ValueError("QA metrics require at least one gold answer.")
    return max(_single_qa_token_f1(prediction, answer) for answer in gold_answers)


def qa_scores(predictions: Sequence[str], gold_answers: Sequence[Sequence[str]]) -> dict[str, float]:
    if len(predictions) != len(gold_answers):
        raise ValueError("QA prediction and gold counts must match.")
    if not predictions:
        return {"exact_match": 0.0, "token_f1": 0.0}
    return {
        "exact_match": sum(
            qa_exact_match(pred, answers) for pred, answers in zip(predictions, gold_answers)
        )
        / len(predictions),
        "token_f1": sum(
            qa_token_f1(pred, answers) for pred, answers in zip(predictions, gold_answers)
        )
        / len(predictions),
    }


def rouge_l_f1(prediction: str, reference: str) -> float:
    predicted_tokens = _tokenize_for_rouge(prediction)
    reference_tokens = _tokenize_for_rouge(reference)
    if not predicted_tokens and not reference_tokens:
        return 1.0
    if not predicted_tokens or not reference_tokens:
        return 0.0
    lcs = _lcs_length(predicted_tokens, reference_tokens)
    return _f1(lcs / len(predicted_tokens), lcs / len(reference_tokens))


def mean_rouge_l_f1(predictions: Sequence[str], references: Sequence[str]) -> float:
    if len(predictions) != len(references):
        raise ValueError("Summarization prediction and reference counts must match.")
    if not predictions:
        return 0.0
    return sum(rouge_l_f1(pred, ref) for pred, ref in zip(predictions, references)) / len(predictions)


def relation_macro_f1(gold_labels: Sequence[str], predicted_labels: Sequence[str]) -> float:
    if len(gold_labels) != len(predicted_labels):
        raise ValueError("Relation gold and predicted label counts must match.")
    if not gold_labels:
        return 0.0
    labels = sorted(set(gold_labels) | set(predicted_labels))
    return float(
        f1_score(gold_labels, predicted_labels, labels=labels, average="macro", zero_division=0)
    )


def _split_bio_label(label: str) -> tuple[str, str | None]:
    if label == "O":
        return "O", None
    if "-" not in label:
        raise ValueError(f"Invalid BIO label: {label}")
    prefix, entity_type = label.split("-", 1)
    if prefix not in {"B", "I"} or not entity_type:
        raise ValueError(f"Invalid BIO label: {label}")
    return prefix, entity_type


def _normalize_answer(text: str) -> str:
    lowered = text.lower()
    without_punctuation = "".join(ch for ch in lowered if ch not in string.punctuation)
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def _tokenize_for_rouge(text: str) -> list[str]:
    without_punctuation = "".join(ch.lower() if ch not in string.punctuation else " " for ch in text)
    return without_punctuation.split()


def _single_qa_token_f1(prediction: str, answer: str) -> float:
    predicted_tokens = _normalize_answer(prediction).split()
    answer_tokens = _normalize_answer(answer).split()
    if not predicted_tokens and not answer_tokens:
        return 1.0
    if not predicted_tokens or not answer_tokens:
        return 0.0
    common = collections.Counter(predicted_tokens) & collections.Counter(answer_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    return _f1(overlap / len(predicted_tokens), overlap / len(answer_tokens))


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for idx, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[idx - 1] + 1)
            else:
                current.append(max(previous[idx], current[-1]))
        previous = current
    return previous[-1]


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
