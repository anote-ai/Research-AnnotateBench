from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class FailureCategory(str, Enum):
    LABEL_AMBIGUITY = "label_ambiguity"
    KNOWLEDGE_GAP = "knowledge_gap"
    CONTEXT_WINDOW_FAILURE = "context_window_failure"
    SCHEMA_CONFUSION = "schema_confusion"
    OVERCONFIDENT_HALLUCINATION = "overconfident_hallucination"


@dataclass(frozen=True)
class FailureCode:
    example_id: str
    category: FailureCategory
    annotator_id: str | None = None
    model_name: str | None = None
    confidence: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1] when provided.")


def summarize_failure_codes(codes: Sequence[FailureCode]) -> dict[FailureCategory, int]:
    """Count coded failures by taxonomy category."""
    summary = {category: 0 for category in FailureCategory}
    for code in codes:
        summary[code.category] += 1
    return summary
