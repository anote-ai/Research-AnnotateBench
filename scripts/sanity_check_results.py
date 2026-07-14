#!/usr/bin/env python3
"""Run lightweight sanity checks over benchmark result CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CORE_COLUMNS = ["dataset", "strategy", "budget", "seed", "macro_f1"]
KEY_CANDIDATES = ["dataset", "strategy", "budget", "seed", "cost_scenario", "downstream_model"]
METRIC_COLUMNS = [
    "macro_f1",
    "accuracy",
    "llm_label_accuracy",
    "llm_label_macro_f1",
    "llm_ece",
    "llm_lari",
]
COST_COLUMNS = ["annotation_cost", "selection_cost", "total_cost", "cost_usd"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--cost-scenario")
    parser.add_argument("--output-csv", default="results/sanity_check_report.csv")
    return parser.parse_args()


def sanity_check_results(
    result_paths: list[str | Path],
    *,
    baseline_path: str | Path | None = None,
    cost_scenario: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    frames: list[tuple[Path, pd.DataFrame]] = []
    for result_path in result_paths:
        path = Path(result_path)
        df = pd.read_csv(path)
        if cost_scenario and "cost_scenario" in df.columns:
            df = df[df["cost_scenario"] == cost_scenario].copy()
        frames.append((path, df))
        rows.extend(_check_frame(path, df))

    if baseline_path is not None:
        baseline = pd.read_csv(baseline_path)
        if cost_scenario and "cost_scenario" in baseline.columns:
            baseline = baseline[baseline["cost_scenario"] == cost_scenario].copy()
        for path, df in frames:
            rows.extend(_compare_to_baseline(path, df, Path(baseline_path), baseline))

    return pd.DataFrame(rows, columns=["level", "file", "check", "detail"])


def _check_frame(path: Path, df: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    _add(rows, "info", path, "shape", f"{len(df)} rows x {len(df.columns)} columns")
    if df.empty:
        _add(rows, "fail", path, "empty", "No rows after filtering.")
        return rows

    missing_core = [column for column in CORE_COLUMNS if column not in df.columns]
    if missing_core:
        _add(rows, "warning", path, "core_columns", f"Missing core columns: {missing_core}")

    key_columns = [column for column in KEY_CANDIDATES if column in df.columns]
    if key_columns:
        duplicate_count = int(df.duplicated(key_columns).sum())
        _add(
            rows,
            "fail" if duplicate_count else "info",
            path,
            "duplicate_keys",
            f"{duplicate_count} duplicates on {key_columns}",
        )
        missing_key_counts = df[key_columns].isna().sum()
        missing = {column: int(count) for column, count in missing_key_counts.items() if count}
        if missing:
            _add(rows, "fail", path, "missing_key_values", str(missing))

    if "budget" in df.columns:
        bad_budgets = int((pd.to_numeric(df["budget"], errors="coerce") <= 0).sum())
        if bad_budgets:
            _add(rows, "fail", path, "budget_range", f"{bad_budgets} rows have non-positive budget.")

    for column in [column for column in METRIC_COLUMNS if column in df.columns]:
        values = pd.to_numeric(df[column], errors="coerce")
        null_count = int(values.isna().sum())
        out_of_range = int(((values < 0) | (values > 1)).sum())
        if null_count or out_of_range:
            _add(
                rows,
                "fail",
                path,
                f"{column}_range",
                f"{null_count} null/non-numeric, {out_of_range} outside [0, 1]",
            )
        else:
            _add(rows, "info", path, f"{column}_range", f"{values.min():.4f}..{values.max():.4f}")

    for column in [column for column in COST_COLUMNS if column in df.columns]:
        values = pd.to_numeric(df[column], errors="coerce")
        negative = int((values < 0).sum())
        if negative:
            _add(rows, "fail", path, f"{column}_range", f"{negative} negative values")

    if {"dataset", "macro_f1"}.issubset(df.columns):
        for dataset, dataset_df in df.groupby("dataset"):
            if dataset_df["macro_f1"].nunique(dropna=True) == 1 and len(dataset_df) > 1:
                _add(
                    rows,
                    "warning",
                    path,
                    "constant_macro_f1",
                    f"{dataset} has one macro_f1 value across {len(dataset_df)} rows.",
                )

    return rows


def _compare_to_baseline(
    path: Path,
    df: pd.DataFrame,
    baseline_path: Path,
    baseline: pd.DataFrame,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    key_columns = [column for column in KEY_CANDIDATES if column in df.columns and column in baseline.columns]
    if not key_columns:
        _add(rows, "warning", path, "baseline_keys", f"No shared key columns with {baseline_path}.")
        return rows

    current_keys = df[key_columns].drop_duplicates()
    baseline_keys = baseline[key_columns].drop_duplicates()
    merged = current_keys.merge(baseline_keys, on=key_columns, how="outer", indicator=True)
    only_current = int((merged["_merge"] == "left_only").sum())
    only_baseline = int((merged["_merge"] == "right_only").sum())
    _add(
        rows,
        "warning" if only_current or only_baseline else "info",
        path,
        "baseline_key_coverage",
        f"{only_current} keys only in current, {only_baseline} only in {baseline_path} using {key_columns}",
    )
    return rows


def _add(rows: list[dict[str, str]], level: str, path: Path, check: str, detail: str) -> None:
    rows.append({"level": level, "file": str(path), "check": check, "detail": detail})


def main() -> None:
    args = parse_args()
    report = sanity_check_results(
        args.results,
        baseline_path=args.baseline,
        cost_scenario=args.cost_scenario,
    )
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    print(f"Wrote {len(report)} checks to {output_path}")
    failures = report[report["level"] == "fail"]
    warnings = report[report["level"] == "warning"]
    print(f"Failures: {len(failures)}")
    print(f"Warnings: {len(warnings)}")
    if len(failures):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
