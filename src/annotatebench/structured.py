from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class NERExample:
    tokens: List[str]
    labels: List[str]

    def __post_init__(self) -> None:
        if len(self.tokens) != len(self.labels):
            raise ValueError("NER tokens and labels must have the same length.")


@dataclass(frozen=True)
class NERDataset:
    name: str
    train_examples: List[NERExample]
    test_examples: List[NERExample]
    label_names: List[str] | None = None


@dataclass(frozen=True)
class QAExample:
    question: str
    context: str
    answers: List[str]

    def __post_init__(self) -> None:
        if not self.answers:
            raise ValueError("QA examples must include at least one gold answer.")


@dataclass(frozen=True)
class QADataset:
    name: str
    train_examples: List[QAExample]
    test_examples: List[QAExample]


@dataclass(frozen=True)
class SummarizationExample:
    document: str
    summary: str


@dataclass(frozen=True)
class SummarizationDataset:
    name: str
    train_examples: List[SummarizationExample]
    test_examples: List[SummarizationExample]


@dataclass(frozen=True)
class RelationExtractionExample:
    context: str
    head_entity: str
    tail_entity: str
    relation: str


@dataclass(frozen=True)
class RelationExtractionDataset:
    name: str
    train_examples: List[RelationExtractionExample]
    test_examples: List[RelationExtractionExample]
    label_names: List[str] | None = None
