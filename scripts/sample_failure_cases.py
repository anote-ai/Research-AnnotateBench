#!/usr/bin/env python3
"""Sample incorrect row-level annotation cases for qualitative review."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_GROUP_COLUMNS = ["dataset_name", "model_name", "prompt_version"]
KEEP_COLUMNS = [
    "source_file",
    "dataset_name",
    "dataset",
    "task_type",
    "example_id",
    "model_name",
    "prompt_version",
    "gold_label",
    "predicted_label",
    "confidence",
    "correct",
    "failure_category",
    "rationale",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", nargs="+", required=True)
    parser.add_argument("--n-per-group", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group-columns", default=",".join(DEFAULT_GROUP_COLUMNS))
    parser.add_argument("--output-csv", default="results/failure_case_samples.csv")
    return parser.parse_args()


def sample_failure_cases(
    annotation_paths: list[str | Path],
    *,
    n_per_group: int = 5,
    seed: int = 0,
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    frames = []
    for annotation_path in annotation_paths:
        path = Path(annotation_path)
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    failures = _filter_failures(df)
    if failures.empty:
        return failures[[column for column in KEEP_COLUMNS if column in failures.columns]]

    requested_groups = group_columns or DEFAULT_GROUP_COLUMNS
    actual_groups = [column for column in requested_groups if column in failures.columns]
    if actual_groups:
        samples = []
        for _, group in failures.groupby(actual_groups, sort=True):
            samples.append(group.sample(min(len(group), n_per_group), random_state=seed))
        sampled = pd.concat(samples, ignore_index=True)
    else:
        sampled = failures.sample(min(len(failures), n_per_group), random_state=seed)

    sort_columns = [column for column in actual_groups + ["confidence"] if column in sampled.columns]
    if sort_columns:
        ascending = [True] * len(sort_columns)
        if sort_columns[-1] == "confidence":
            ascending[-1] = False
        sampled = sampled.sort_values(sort_columns, ascending=ascending)

    return sampled[[column for column in KEEP_COLUMNS if column in sampled.columns]].reset_index(drop=True)


def _filter_failures(df: pd.DataFrame) -> pd.DataFrame:
    if "correct" in df.columns:
        correct = df["correct"].astype(str).str.lower().isin(["true", "1", "yes"])
        return df[~correct].copy()
    if {"gold_label", "predicted_label"}.issubset(df.columns):
        return df[df["gold_label"].astype(str) != df["predicted_label"].astype(str)].copy()
    raise ValueError("Need either a correct column or gold_label/predicted_label columns.")


def main() -> None:
    args = parse_args()
    group_columns = [column.strip() for column in args.group_columns.split(",") if column.strip()]
    samples = sample_failure_cases(
        args.annotations,
        n_per_group=args.n_per_group,
        seed=args.seed,
        group_columns=group_columns,
    )
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(output_path, index=False)
    print(f"Wrote {len(samples)} sampled failure cases to {output_path}")


if __name__ == "__main__":
    main()
