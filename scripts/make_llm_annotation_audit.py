#!/usr/bin/env python3
"""Extract recorded off-schema LLM annotations for camera-ready auditing."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", default="results/llm_annotations")
    parser.add_argument("--output-csv", default="results/camera_ready/llm_off_schema_audit.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = [pd.read_csv(path) for path in sorted(Path(args.annotation_dir).glob("*.csv"))]
    if not frames:
        raise SystemExit("No annotation caches found.")
    output = pd.concat(frames, ignore_index=True)
    output = output[output["notes"].fillna("").str.startswith("invalid_annotation:")].copy()
    columns = [
        "dataset_name",
        "seed",
        "example_id",
        "gold_label",
        "predicted_label",
        "confidence",
        "model_name",
        "temperature",
        "prompt_version",
        "notes",
    ]
    path = Path(args.output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    output[columns].sort_values(["dataset_name", "seed", "example_id"]).to_csv(path, index=False)
    print(f"Wrote {len(output)} off-schema rows to {path}")


if __name__ == "__main__":
    main()
