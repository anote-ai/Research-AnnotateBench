#!/usr/bin/env python3
"""Run the LLM annotator strategy and emit benchmark-style rows."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from annotatebench.core import AnnotationStrategy
from annotatebench.costs import get_cost_scenarios
from annotatebench.datasets import BENCHMARK_DATASETS, load_benchmark_dataset
from annotatebench.llm import (
    DEFAULT_API_URL,
    GENERIC_PROMPT_VERSION,
    build_user_prompt,
    call_chat_completion,
    dataset_label_names,
    dry_run_annotation,
    load_prompt,
    make_pricing,
    normalize_annotation,
)
from annotatebench.metrics.lari import expected_calibration_error, lari_score
from annotatebench.pilot import fit_and_score_text_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(BENCHMARK_DATASETS))
    parser.add_argument("--budgets", default="50,100,250")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--cost-scenarios", default="low,base,high")
    parser.add_argument("--max-train-examples", type=int, default=1200)
    parser.add_argument("--max-test-examples", type=int, default=1000)
    parser.add_argument("--financial-phrasebank-path")
    parser.add_argument("--trec-train-path")
    parser.add_argument("--trec-test-path")
    parser.add_argument("--banking77-train-path")
    parser.add_argument("--banking77-test-path")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--api-url", default=os.environ.get("OPENAI_API_URL", DEFAULT_API_URL))
    parser.add_argument("--prompt-version", default=GENERIC_PROMPT_VERSION)
    parser.add_argument("--prompt-dir", default="prompts")
    parser.add_argument("--annotation-dir", default="results/llm_annotations")
    parser.add_argument("--output-csv", default="results/benchmark_results_with_llm.csv")
    parser.add_argument("--input-usd-per-million-tokens", type=float, default=0.0)
    parser.add_argument("--output-usd-per-million-tokens", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=2.0)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing row-level annotation CSVs and call the API for every selected example.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Exercise the pipeline without calling an API.")
    return parser.parse_args()


def parse_csv_values(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def main() -> None:
    args = parse_args()
    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required unless --dry-run is set.")

    dataset_names = parse_csv_values(args.datasets)
    budgets = [int(value) for value in parse_csv_values(args.budgets)]
    seeds = [int(value) for value in parse_csv_values(args.seeds)]
    cost_scenarios = get_cost_scenarios(parse_csv_values(args.cost_scenarios))
    system_prompt, user_template = load_prompt(args.prompt_dir, args.prompt_version)
    pricing = make_pricing(
        model=args.model,
        input_usd_per_million_tokens=args.input_usd_per_million_tokens,
        output_usd_per_million_tokens=args.output_usd_per_million_tokens,
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_dir = Path(args.annotation_dir)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict[str, object]] = []
    for dataset_name in dataset_names:
        print(f"Loading {dataset_name}...")
        dataset = load_benchmark_dataset(
            dataset_name,
            max_train_examples=args.max_train_examples,
            max_test_examples=args.max_test_examples,
            financial_phrasebank_path=args.financial_phrasebank_path,
            trec_train_path=args.trec_train_path,
            trec_test_path=args.trec_test_path,
            banking77_train_path=args.banking77_train_path,
            banking77_test_path=args.banking77_test_path,
            download=args.download_data,
        )
        label_names = dataset_label_names(dataset)
        active_budgets = [budget for budget in budgets if budget <= len(dataset.train_texts)]
        if not active_budgets:
            active_budgets = [len(dataset.train_texts)]

        for seed in seeds:
            max_budget = max(active_budgets)
            selected_pool = nested_random_indices(len(dataset.train_texts), max_budget, seed)
            cache_path = annotation_dir / f"{dataset.name}_llm_annotator_seed{seed}.csv"
            for budget in active_budgets:
                selected = sorted(selected_pool[:budget])
                print(f"Running llm_annotator: dataset={dataset.name}, seed={seed}, budget={len(selected)}")
                annotations = annotate_selected_examples(
                    cache_path=cache_path,
                    dataset_name=dataset.name,
                    selected_indices=selected,
                    texts=dataset.train_texts,
                    gold_labels=dataset.train_labels,
                    label_names=label_names,
                    model=args.model,
                    prompt_version=args.prompt_version,
                    system_prompt=system_prompt,
                    user_template=user_template,
                    api_url=args.api_url,
                    dry_run=args.dry_run,
                    pricing=pricing,
                    max_retries=args.max_retries,
                    retry_sleep_seconds=args.retry_sleep_seconds,
                    resume=not args.no_resume,
                )
                predicted_labels = [row["predicted_label"] for row in annotations]
                selected_texts = [dataset.train_texts[i] for i in selected]
                selected_gold_labels = [dataset.train_labels[i] for i in selected]
                macro_f1, accuracy = fit_and_score_text_classifier(
                    selected_texts,
                    predicted_labels,
                    dataset.test_texts,
                    dataset.test_labels,
                    seed=seed,
                )
                llm_correctness = [
                    gold == predicted for gold, predicted in zip(selected_gold_labels, predicted_labels)
                ]
                llm_macro_f1 = f1_score(
                    selected_gold_labels,
                    predicted_labels,
                    labels=label_names,
                    average="macro",
                )
                confidences = [float(row["confidence"]) for row in annotations]
                annotation_cost = sum(float(row["cost_usd"] or 0.0) for row in annotations)
                for scenario in cost_scenarios:
                    result_rows.append(
                        {
                            "dataset": dataset.name,
                            "strategy": AnnotationStrategy.LLM_ANNOTATOR.value,
                            "budget": len(selected),
                            "seed": seed,
                            "macro_f1": macro_f1,
                            "accuracy": accuracy,
                            "cost_scenario": scenario.name,
                            "cost_source": "OpenAI API usage" if pricing is not None else "not estimated",
                            "cost_source_url": "",
                            "cost_checked_at": "",
                            "human_cost_per_label": scenario.human_cost_per_label,
                            "selection_cost_per_example": 0.0,
                            "annotation_cost": annotation_cost,
                            "selection_cost": 0.0,
                            "total_cost": annotation_cost,
                            "llm_label_accuracy": accuracy_score(selected_gold_labels, predicted_labels),
                            "llm_label_macro_f1": llm_macro_f1,
                            "llm_ece": expected_calibration_error(confidences, llm_correctness),
                            "llm_lari": lari_score(llm_macro_f1, confidences, llm_correctness),
                            "model_name": args.model,
                            "prompt_version": args.prompt_version,
                        }
                    )
                pd.DataFrame(result_rows).to_csv(output_path, index=False)
                print(f"Saved partial LLM strategy results: {len(result_rows)} rows to {output_path}")

    pd.DataFrame(result_rows).to_csv(output_path, index=False)
    print(f"Wrote {len(result_rows)} rows to {output_path}")


def nested_random_indices(n_samples: int, max_budget: int, seed: int) -> list[int]:
    capped_budget = min(max_budget, n_samples)
    rng = np.random.default_rng(seed)
    return rng.choice(n_samples, size=capped_budget, replace=False).tolist()


def annotate_selected_examples(
    *,
    cache_path: Path,
    dataset_name: str,
    selected_indices: list[int],
    texts: list[str],
    gold_labels: list[str],
    label_names: list[str],
    model: str,
    prompt_version: str,
    system_prompt: str,
    user_template: str,
    api_url: str,
    dry_run: bool,
    pricing: object,
    max_retries: int,
    retry_sleep_seconds: float,
    resume: bool,
) -> list[dict[str, object]]:
    cached_by_id = read_annotation_cache(cache_path) if resume else {}
    rows_by_id: dict[str, dict[str, object]] = dict(cached_by_id)
    for index in selected_indices:
        example_id = f"{dataset_name}_train_{index}"
        if example_id in rows_by_id:
            continue

        user_prompt = build_user_prompt(
            user_template,
            text=texts[index],
            dataset_name=dataset_name,
            label_names=label_names,
        )
        annotation = (
            dry_run_annotation(label_names)
            if dry_run
            else call_chat_completion(
                api_url=api_url,
                api_key=os.environ["OPENAI_API_KEY"],
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_retries=max_retries,
                retry_sleep_seconds=retry_sleep_seconds,
            )
        )
        try:
            parsed = normalize_annotation(annotation, label_names, pricing)
            predicted_label = parsed.predicted_label
            confidence = parsed.confidence
            cost_usd = parsed.cost_usd
            rationale = parsed.rationale
            raw_response = parsed.raw_response
            notes = "dry_run" if dry_run else ""
        except (TypeError, ValueError, KeyError) as exc:
            predicted_label = label_names[0]
            confidence = 0.0
            cost_usd = None
            rationale = ""
            raw_response = annotation.get("_raw_response", annotation)
            notes = f"invalid_annotation: {exc}"

        rows_by_id[example_id] = {
            "dataset_name": dataset_name,
            "task_type": "classification",
            "example_id": example_id,
            "model_name": model,
            "prompt_version": prompt_version,
            "gold_label": gold_labels[index],
            "predicted_label": predicted_label,
            "confidence": confidence,
            "correct": gold_labels[index] == predicted_label and not str(notes).startswith("invalid_annotation"),
            "cost_usd": "" if cost_usd is None else cost_usd,
            "failure_category": "",
            "rationale": rationale,
            "notes": notes,
            "raw_response": json.dumps(raw_response, ensure_ascii=True),
        }
        write_annotation_rows(
            cache_path,
            [rows_by_id[f"{dataset_name}_train_{selected_index}"] for selected_index in selected_indices if f"{dataset_name}_train_{selected_index}" in rows_by_id],
        )

    rows = [rows_by_id[f"{dataset_name}_train_{index}"] for index in selected_indices]
    write_annotation_rows(cache_path, rows)
    return rows


def read_annotation_cache(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {str(row["example_id"]): row for row in rows if row.get("example_id")}


def write_annotation_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write empty annotation rows.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
