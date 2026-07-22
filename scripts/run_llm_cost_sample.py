#!/usr/bin/env python3
"""Run only the LLM calls listed in a cost-sample manifest."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from annotatebench.datasets import load_benchmark_dataset
from annotatebench.llm import DEFAULT_API_URL, GENERIC_PROMPT_VERSION, dataset_label_names, load_prompt, make_pricing
from run_llm_strategy_benchmark import annotate_selected_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="results/llm_cost_seed0_sample_manifest.csv")
    parser.add_argument("--annotation-dir", default="results/llm_cost_samples")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--prompt-version", default=GENERIC_PROMPT_VERSION)
    parser.add_argument("--prompt-dir", default="prompts")
    parser.add_argument("--api-url", default=os.environ.get("OPENAI_API_URL", DEFAULT_API_URL))
    parser.add_argument("--input-usd-per-million-tokens", type=float, required=True)
    parser.add_argument("--output-usd-per-million-tokens", type=float, required=True)
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--financial-phrasebank-path")
    parser.add_argument("--trec-train-path")
    parser.add_argument("--trec-test-path")
    parser.add_argument("--banking77-train-path")
    parser.add_argument("--banking77-test-path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required.")
    if args.input_usd_per_million_tokens <= 0 or args.output_usd_per_million_tokens <= 0:
        raise SystemExit("Token prices must be positive; zero-cost placeholders are invalid.")
    manifest = pd.read_csv(args.manifest)
    system_prompt, user_template = load_prompt(args.prompt_dir, args.prompt_version)
    pricing = make_pricing(
        model=args.model,
        input_usd_per_million_tokens=args.input_usd_per_million_tokens,
        output_usd_per_million_tokens=args.output_usd_per_million_tokens,
    )
    output_dir = Path(args.annotation_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name, group in manifest.groupby("dataset", sort=True):
        dataset = load_benchmark_dataset(
            str(dataset_name),
            download=args.download_data,
            financial_phrasebank_path=args.financial_phrasebank_path,
            trec_train_path=args.trec_train_path,
            trec_test_path=args.trec_test_path,
            banking77_train_path=args.banking77_train_path,
            banking77_test_path=args.banking77_test_path,
        )
        selected = sorted(int(value) for value in group["dataset_index"])
        output = output_dir / f"{dataset.name}_llm_cost_sample_seed0.csv"
        annotate_selected_examples(
            cache_path=output,
            run_id=f"{dataset.name}_cost_sample_seed0_{args.model}_{args.prompt_version}",
            dataset_name=dataset.name,
            split="train",
            seed=0,
            selected_indices=selected,
            texts=dataset.train_texts,
            gold_labels=dataset.train_labels,
            label_names=dataset_label_names(dataset),
            model=args.model,
            temperature=0.0,
            replicate_id=0,
            prompt_version=args.prompt_version,
            system_prompt=system_prompt,
            user_template=user_template,
            api_url=args.api_url,
            dry_run=False,
            pricing=pricing,
            difficulty_buckets={},
            max_retries=3,
            retry_sleep_seconds=2.0,
            resume=True,
        )
        print(f"Wrote cost sample to {output}")


if __name__ == "__main__":
    main()
