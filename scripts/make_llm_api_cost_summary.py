#!/usr/bin/env python3
"""Summarize recorded LLM annotation token usage into API cost estimates."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_PRICE_SOURCE = "https://developers.openai.com/api/docs/models/gpt-4o-mini"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-dir",
        default="results/llm_annotations",
        help="Directory containing row-level LLM annotation CSV logs.",
    )
    parser.add_argument("--seeds", default="1,2", help="Comma-separated seeds to summarize.")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--input-usd-per-million-tokens", type=float, default=0.15)
    parser.add_argument("--output-usd-per-million-tokens", type=float, default=0.60)
    parser.add_argument("--price-checked-at", default="2026-07-22")
    parser.add_argument("--price-source-url", default=DEFAULT_PRICE_SOURCE)
    parser.add_argument("--output-csv", default="results/llm_api_cost_seed1_2_summary.csv")
    return parser.parse_args()


def parse_seeds(raw: str) -> set[int]:
    return {int(value.strip()) for value in raw.split(",") if value.strip()}


def cost_usd(input_tokens: int, output_tokens: int, input_price: float, output_price: float) -> float:
    return input_tokens * input_price / 1_000_000 + output_tokens * output_price / 1_000_000


def token_counts(row: dict[str, str]) -> tuple[int, int, int]:
    input_tokens = int(float(row.get("input_tokens") or 0))
    output_tokens = int(float(row.get("output_tokens") or 0))
    total_tokens = int(float(row.get("total_tokens") or 0))
    if total_tokens > 0:
        return input_tokens, output_tokens, total_tokens
    raw_response = row.get("raw_response") or ""
    if raw_response:
        try:
            usage = json.loads(raw_response).get("usage", {})
            input_tokens = int(usage.get("prompt_tokens") or 0)
            output_tokens = int(usage.get("completion_tokens") or 0)
            total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    return input_tokens, output_tokens, total_tokens


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    annotation_dir = Path(args.annotation_dir)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    totals: dict[tuple[str, int], dict[str, object]] = defaultdict(
        lambda: {
            "rows_total": 0,
            "rows_with_recorded_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    )

    for csv_path in sorted(annotation_dir.glob("*_llm_annotator_seed*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    seed = int(row["seed"])
                except (KeyError, TypeError, ValueError):
                    continue
                if seed not in seeds:
                    continue
                dataset = row.get("dataset_name", "")
                if not dataset:
                    continue
                key = (dataset, seed)
                input_tokens, output_tokens, total_tokens = token_counts(row)
                totals[key]["rows_total"] = int(totals[key]["rows_total"]) + 1
                if total_tokens > 0:
                    totals[key]["rows_with_recorded_tokens"] = int(totals[key]["rows_with_recorded_tokens"]) + 1
                totals[key]["input_tokens"] = int(totals[key]["input_tokens"]) + input_tokens
                totals[key]["output_tokens"] = int(totals[key]["output_tokens"]) + output_tokens
                totals[key]["total_tokens"] = int(totals[key]["total_tokens"]) + total_tokens

    fieldnames = [
        "dataset",
        "seed",
        "model_name",
        "rows_total",
        "rows_with_recorded_tokens",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "input_usd_per_million_tokens",
        "output_usd_per_million_tokens",
        "api_cost_usd",
        "cost_estimation_method",
        "cost_sample_size",
        "estimated_cost_lower_95",
        "estimated_cost_upper_95",
        "price_checked_at",
        "price_source_url",
    ]
    rows: list[dict[str, object]] = []
    for (dataset, seed), values in sorted(totals.items()):
        input_tokens = int(values["input_tokens"])
        output_tokens = int(values["output_tokens"])
        measured_cost = cost_usd(
            input_tokens,
            output_tokens,
            args.input_usd_per_million_tokens,
            args.output_usd_per_million_tokens,
        )
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "model_name": args.model,
                "rows_total": values["rows_total"],
                "rows_with_recorded_tokens": values["rows_with_recorded_tokens"],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": values["total_tokens"],
                "input_usd_per_million_tokens": args.input_usd_per_million_tokens,
                "output_usd_per_million_tokens": args.output_usd_per_million_tokens,
                "api_cost_usd": measured_cost,
                "cost_estimation_method": "measured",
                "cost_sample_size": values["rows_with_recorded_tokens"],
                "estimated_cost_lower_95": measured_cost,
                "estimated_cost_upper_95": measured_cost,
                "price_checked_at": args.price_checked_at,
                "price_source_url": args.price_source_url,
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    total_input = sum(int(row["input_tokens"]) for row in rows)
    total_output = sum(int(row["output_tokens"]) for row in rows)
    total_cost = cost_usd(
        total_input,
        total_output,
        args.input_usd_per_million_tokens,
        args.output_usd_per_million_tokens,
    )
    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Recorded input tokens: {total_input}")
    print(f"Recorded output tokens: {total_output}")
    print(f"Estimated API cost: ${total_cost:.6f}")


if __name__ == "__main__":
    main()
