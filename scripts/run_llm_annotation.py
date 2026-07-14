#!/usr/bin/env python3
"""Run an LLM annotation experiment with a versioned prompt."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from annotatebench.annotation_schema import (
    ROW_LEVEL_FIELDNAMES,
    annotator_id,
    format_temperature,
)
from annotatebench.datasets import BENCHMARK_DATASETS, load_benchmark_dataset
from annotatebench.llm import (
    DEFAULT_API_URL,
    build_user_prompt,
    call_chat_completion,
    dataset_label_names,
    default_prompt_version,
    dry_run_annotation,
    load_prompt,
    make_pricing,
    normalize_annotation,
)
from annotatebench.metrics.lari import expected_calibration_error, lari_score
from sklearn.metrics import accuracy_score, f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=BENCHMARK_DATASETS, default="financial_phrasebank")
    parser.add_argument("--financial-phrasebank-path")
    parser.add_argument("--trec-train-path")
    parser.add_argument("--trec-test-path")
    parser.add_argument("--banking77-train-path")
    parser.add_argument("--banking77-test-path")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--max-train-examples", type=int, default=1200)
    parser.add_argument("--max-test-examples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--temperatures", default="0")
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--run-id")
    parser.add_argument("--difficulty-csv")
    parser.add_argument("--api-url", default=os.environ.get("OPENAI_API_URL", DEFAULT_API_URL))
    parser.add_argument("--prompt-version")
    parser.add_argument("--prompt-dir", default="prompts")
    parser.add_argument("--output-csv", default="results/llm_annotations_financial_phrasebank.csv")
    parser.add_argument("--summary-csv", default="results/llm_annotation_summary.csv")
    parser.add_argument("--input-usd-per-million-tokens", type=float, default=0.0)
    parser.add_argument("--output-usd-per-million-tokens", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true", help="Exercise the pipeline without calling an API.")
    return parser.parse_args()


def parse_csv_floats(raw: str) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("At least one temperature is required.")
    return values


def load_difficulty_buckets(path: str | None) -> dict[str, str]:
    if path is None:
        return {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"example_id", "difficulty_bucket"}
    missing = required.difference(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"Missing difficulty columns: {sorted(missing)}")
    return {str(row["example_id"]): str(row["difficulty_bucket"]) for row in rows}


def selected_examples(args: argparse.Namespace) -> tuple[str, list[str], list[str], list[str]]:
    dataset = load_benchmark_dataset(
        args.dataset,
        seed=args.seed,
        max_train_examples=args.max_train_examples,
        max_test_examples=args.max_test_examples,
        financial_phrasebank_path=args.financial_phrasebank_path,
        trec_train_path=args.trec_train_path,
        trec_test_path=args.trec_test_path,
        banking77_train_path=args.banking77_train_path,
        banking77_test_path=args.banking77_test_path,
        download=args.download_data,
    )
    if args.split == "train":
        texts, labels = dataset.train_texts, dataset.train_labels
    else:
        texts, labels = dataset.test_texts, dataset.test_labels
    end = None if args.limit is None else args.offset + args.limit
    return dataset.name, texts[args.offset:end], labels[args.offset:end], dataset_label_names(dataset)


def write_summary(
    *,
    path: Path,
    dataset_name: str,
    split: str,
    model: str,
    prompt_version: str,
    label_names: list[str],
    gold_labels: list[str],
    predicted_labels: list[str],
    confidences: list[float],
    input_tokens: list[int],
    output_tokens: list[int],
    total_tokens: list[int],
    costs_usd: list[float],
) -> None:
    if not gold_labels:
        raise ValueError("Cannot summarize an empty annotation run.")
    correctness = [gold == predicted for gold, predicted in zip(gold_labels, predicted_labels)]
    metric_labels = label_names + sorted(set(predicted_labels).difference(label_names))
    macro_f1 = f1_score(
        gold_labels,
        predicted_labels,
        labels=metric_labels,
        average="macro",
        zero_division=0,
    )
    ece = expected_calibration_error(confidences, correctness)
    confusion = {
        gold: {
            predicted: sum(
                1
                for gold_label, predicted_label in zip(gold_labels, predicted_labels)
                if gold_label == gold and predicted_label == predicted
            )
            for predicted in metric_labels
        }
        for gold in metric_labels
    }
    row = {
        "dataset_name": dataset_name,
        "split": split,
        "model_name": model,
        "prompt_version": prompt_version,
        "n_examples": len(gold_labels),
        "accuracy": accuracy_score(gold_labels, predicted_labels),
        "macro_f1": macro_f1,
        "ece": ece,
        "lari": lari_score(macro_f1, confidences, correctness),
        "error_count": correctness.count(False),
        "input_tokens": sum(input_tokens),
        "output_tokens": sum(output_tokens),
        "total_tokens": sum(total_tokens),
        "cost_usd": sum(costs_usd),
        "confusion_json": json.dumps(confusion, sort_keys=True),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required unless --dry-run is set.")
    if args.replicates <= 0:
        raise SystemExit("--replicates must be positive.")

    temperatures = parse_csv_floats(args.temperatures)
    prompt_version = args.prompt_version or default_prompt_version(args.dataset)
    system_prompt, user_template = load_prompt(args.prompt_dir, prompt_version)
    dataset_name, texts, gold_labels, label_names = selected_examples(args)
    if not texts:
        raise SystemExit("No examples selected.")
    run_id = args.run_id or f"{dataset_name}_{args.split}_seed{args.seed}_{args.model}_{prompt_version}"
    difficulty_buckets = load_difficulty_buckets(args.difficulty_csv)
    pricing = make_pricing(
        model=args.model,
        input_usd_per_million_tokens=args.input_usd_per_million_tokens,
        output_usd_per_million_tokens=args.output_usd_per_million_tokens,
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predicted_labels: list[str] = []
    confidences: list[float] = []
    input_tokens: list[int] = []
    output_tokens: list[int] = []
    total_tokens: list[int] = []
    costs_usd: list[float] = []
    completed_gold_labels: list[str] = []
    stopped_early = False

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_LEVEL_FIELDNAMES)
        writer.writeheader()
        for local_idx, (text, gold_label) in enumerate(zip(texts, gold_labels)):
            example_id = f"{dataset_name}_{args.split}_{args.offset + local_idx}"
            for temperature in temperatures:
                for replicate_id in range(args.replicates):
                    user_prompt = build_user_prompt(
                        user_template,
                        text=text,
                        dataset_name=dataset_name,
                        label_names=label_names,
                    )
                    try:
                        annotation = (
                            dry_run_annotation(label_names)
                            if args.dry_run
                            else call_chat_completion(
                                api_url=args.api_url,
                                api_key=os.environ["OPENAI_API_KEY"],
                                model=args.model,
                                system_prompt=system_prompt,
                                user_prompt=user_prompt,
                                temperature=temperature,
                                max_retries=args.max_retries,
                                retry_sleep_seconds=args.retry_sleep_seconds,
                            )
                        )
                        parsed = normalize_annotation(annotation, label_names, pricing)
                        notes = "dry_run" if args.dry_run else ""
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
                        stopped_early = True
                        print(f"Stopping before {example_id}: {exc}", file=sys.stderr)
                        break

                    completed_gold_labels.append(gold_label)
                    predicted_labels.append(parsed.predicted_label)
                    confidences.append(parsed.confidence)
                    input_tokens.append(parsed.input_tokens)
                    output_tokens.append(parsed.output_tokens)
                    total_tokens.append(parsed.total_tokens)
                    if parsed.cost_usd is not None:
                        costs_usd.append(parsed.cost_usd)
                    writer.writerow(
                        {
                            "run_id": run_id,
                            "dataset_name": dataset_name,
                            "split": args.split,
                            "seed": args.seed,
                            "task_type": "classification",
                            "example_id": example_id,
                            "difficulty_bucket": difficulty_buckets.get(example_id, ""),
                            "annotator_type": "llm",
                            "annotator_id": annotator_id(args.model, temperature, replicate_id),
                            "model_name": args.model,
                            "temperature": format_temperature(temperature),
                            "prompt_version": prompt_version,
                            "replicate_id": replicate_id,
                            "gold_label": gold_label,
                            "predicted_label": parsed.predicted_label,
                            "confidence": parsed.confidence,
                            "correct": gold_label == parsed.predicted_label,
                            "input_tokens": parsed.input_tokens,
                            "output_tokens": parsed.output_tokens,
                            "total_tokens": parsed.total_tokens,
                            "cost_usd": "" if parsed.cost_usd is None else parsed.cost_usd,
                            "failure_category": "",
                            "rationale": parsed.rationale,
                            "notes": notes,
                            "raw_response": json.dumps(parsed.raw_response, ensure_ascii=True),
                        }
                    )
                if stopped_early:
                    break
            if stopped_early:
                break

    if not completed_gold_labels:
        raise SystemExit("No completed annotations; summary metrics were not written.")

    summary_path = Path(args.summary_csv)
    write_summary(
        path=summary_path,
        dataset_name=dataset_name,
        split=args.split,
        model=args.model,
        prompt_version=prompt_version,
        label_names=label_names,
        gold_labels=completed_gold_labels,
        predicted_labels=predicted_labels,
        confidences=confidences,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        costs_usd=costs_usd,
    )
    print(f"Saved row-level annotations: {output_path}")
    print(f"Saved summary metrics: {summary_path}")
    print(
        "Completed annotations: "
        f"{len(completed_gold_labels)} / {len(gold_labels) * len(temperatures) * args.replicates}"
    )
    if stopped_early:
        print("Run stopped early; outputs contain only completed measured annotations.")


if __name__ == "__main__":
    main()
