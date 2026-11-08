#!/usr/bin/env python3
"""Replace the single FPB seed-1 dry-run row with one production API response."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from annotatebench.annotation_schema import ROW_LEVEL_FIELDNAMES, annotator_id
from annotatebench.costs import LlmTokenPricing
from annotatebench.datasets import load_benchmark_dataset
from annotatebench.llm import (
    build_user_prompt,
    call_chat_completion,
    dataset_label_names,
    load_prompt,
    normalize_annotation,
)


TARGET_EXAMPLE_ID = "financial_phrasebank_train_1139"
MODEL_SNAPSHOT = "gpt-4o-mini-2024-07-18"
PROMPT_VERSION = "text_classification_v1_zero_shot_json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-cache",
        default="results/llm_annotations/financial_phrasebank_llm_annotator_seed1.csv",
    )
    parser.add_argument(
        "--output-cache",
        default="results/camera_ready/llm_annotations/financial_phrasebank_llm_annotator_seed1.csv",
    )
    parser.add_argument(
        "--financial-phrasebank-path",
        default="data/raw/FinancialPhraseBank-v1.0.zip",
    )
    parser.add_argument("--prompt-dir", default="prompts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required; no fallback model is permitted.")
    source = Path(args.source_cache)
    output = Path(args.output_cache)
    if source.resolve() == output.resolve():
        raise SystemExit("Output must differ from the original cache so the audit record is preserved.")
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    target_rows = [row for row in rows if row["example_id"] == TARGET_EXAMPLE_ID]
    if len(target_rows) != 1 or target_rows[0].get("notes") != "dry_run":
        raise SystemExit("Expected exactly one dry-run target row; refusing to make a broader change.")

    dataset = load_benchmark_dataset(
        "financial_phrasebank",
        financial_phrasebank_path=args.financial_phrasebank_path,
    )
    index = int(TARGET_EXAMPLE_ID.rsplit("_", 1)[-1])
    label_names = dataset_label_names(dataset)
    system_prompt, user_template = load_prompt(args.prompt_dir, PROMPT_VERSION)
    response = call_chat_completion(
        api_url="https://api.openai.com/v1/chat/completions",
        api_key=os.environ["OPENAI_API_KEY"],
        model=MODEL_SNAPSHOT,
        system_prompt=system_prompt,
        user_prompt=build_user_prompt(
            user_template,
            text=dataset.train_texts[index],
            dataset_name=dataset.name,
            label_names=label_names,
        ),
        temperature=0.0,
        max_retries=0,
    )
    pricing = LlmTokenPricing(
        source_name="OpenAI API usage",
        source_url="https://platform.openai.com/docs/pricing",
        checked_at="2026-07-22",
        model=MODEL_SNAPSHOT,
        input_usd_per_million_tokens=0.15,
        output_usd_per_million_tokens=0.60,
    )
    parsed = normalize_annotation(response, label_names, pricing)
    response_model = parsed.raw_response.get("model")
    if response_model != MODEL_SNAPSHOT:
        raise SystemExit(f"Unexpected response model {response_model!r}; cache was not written.")

    replacement = dict(target_rows[0])
    replacement.update(
        {
            "model_name": MODEL_SNAPSHOT,
            "run_id": f"financial_phrasebank_train_seed1_{MODEL_SNAPSHOT}_{PROMPT_VERSION}",
            "annotator_id": annotator_id(MODEL_SNAPSHOT, 0.0, 0),
            "predicted_label": parsed.predicted_label,
            "confidence": parsed.confidence,
            "correct": dataset.train_labels[index] == parsed.predicted_label,
            "input_tokens": parsed.input_tokens,
            "output_tokens": parsed.output_tokens,
            "total_tokens": parsed.total_tokens,
            "cost_usd": parsed.cost_usd,
            "rationale": parsed.rationale,
            "notes": "",
            "raw_response": json.dumps(parsed.raw_response, ensure_ascii=True),
        }
    )
    corrected = [replacement if row["example_id"] == TARGET_EXAMPLE_ID else row for row in rows]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_LEVEL_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(corrected)
    print(f"Wrote corrected cache to {output}")


if __name__ == "__main__":
    main()
