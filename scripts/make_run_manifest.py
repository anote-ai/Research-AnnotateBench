#!/usr/bin/env python3
"""Create a manifest summarizing result CSV provenance and coverage."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SUMMARY_COLUMNS = [
    "dataset",
    "dataset_name",
    "strategy",
    "budget",
    "seed",
    "cost_scenario",
    "downstream_model",
    "embedding_model",
    "model_name",
    "prompt_version",
]
METRIC_COLUMNS = ["macro_f1", "accuracy", "llm_label_accuracy", "llm_label_macro_f1", "confidence"]
PLURAL_NAMES = {"strategy": "strategies"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--output-csv", default="results/run_manifest.csv")
    return parser.parse_args()


def make_run_manifest(result_paths: list[str | Path]) -> pd.DataFrame:
    generated_at = datetime.now(timezone.utc).isoformat()
    git_commit = _git_commit()
    rows = []
    for result_path in result_paths:
        path = Path(result_path)
        df = pd.read_csv(path)
        stat = path.stat()
        row: dict[str, object] = {
            "file": str(path),
            "generated_at_utc": generated_at,
            "git_commit": git_commit,
            "size_bytes": stat.st_size,
            "modified_at_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "sha256_12": _sha256_12(path),
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": "|".join(df.columns),
        }
        for column in SUMMARY_COLUMNS:
            if column in df.columns:
                plural = PLURAL_NAMES.get(column, f"{column}s")
                row[plural] = _summarize_values(df[column])
                row[f"n_{plural}"] = int(df[column].nunique(dropna=True))
        for column in METRIC_COLUMNS:
            if column in df.columns:
                values = pd.to_numeric(df[column], errors="coerce")
                row[f"{column}_min"] = values.min()
                row[f"{column}_max"] = values.max()
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_values(series: pd.Series, limit: int = 24) -> str:
    values = sorted(str(value) for value in series.dropna().unique())
    if len(values) > limit:
        return "|".join(values[:limit]) + f"|...(+{len(values) - limit})"
    return "|".join(values)


def _sha256_12(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def main() -> None:
    args = parse_args()
    manifest = make_run_manifest(args.results)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    print(f"Wrote {len(manifest)} manifest rows to {output_path}")


if __name__ == "__main__":
    main()
