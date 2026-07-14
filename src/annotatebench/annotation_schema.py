from __future__ import annotations

from typing import Mapping


ROW_LEVEL_FIELDNAMES = [
    "run_id",
    "dataset_name",
    "split",
    "seed",
    "task_type",
    "example_id",
    "difficulty_bucket",
    "annotator_type",
    "annotator_id",
    "model_name",
    "temperature",
    "prompt_version",
    "replicate_id",
    "gold_label",
    "predicted_label",
    "confidence",
    "correct",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cost_usd",
    "failure_category",
    "rationale",
    "notes",
    "raw_response",
]


def annotation_cache_key(row: Mapping[str, object]) -> str:
    return "|".join(
        str(row.get(column, ""))
        for column in ["example_id", "model_name", "temperature", "prompt_version", "replicate_id"]
    )


def annotator_id(model_name: str, temperature: float, replicate_id: int) -> str:
    return f"{model_name}_temp{format_temperature(temperature)}_rep{replicate_id}"


def format_temperature(temperature: float) -> str:
    return f"{temperature:g}"
