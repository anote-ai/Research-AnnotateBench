#!/usr/bin/env python3
"""Compare benchmark results across downstream models."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_BASELINE_MODEL = "tfidf_logreg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--cost-scenario", default="base")
    parser.add_argument(
        "--summary-csv",
        default="results/downstream_model_comparison_summary.csv",
    )
    parser.add_argument(
        "--best-csv",
        default="results/downstream_model_comparison_best.csv",
    )
    parser.add_argument("--baseline-model", default=DEFAULT_BASELINE_MODEL)
    return parser.parse_args()


def make_downstream_model_comparison(
    result_paths: list[str | Path],
    summary_csv: str | Path,
    best_csv: str | Path,
    cost_scenario: str = "base",
    baseline_model: str = DEFAULT_BASELINE_MODEL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for path in result_paths:
        frame = pd.read_csv(path)
        if "downstream_model" not in frame.columns:
            frame = frame.copy()
            frame["downstream_model"] = baseline_model
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    required = {
        "dataset",
        "downstream_model",
        "strategy",
        "budget",
        "seed",
        "macro_f1",
        "accuracy",
        "total_cost",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")
    if "cost_scenario" in df.columns:
        df = df[df["cost_scenario"] == cost_scenario].copy()
    if df.empty:
        raise ValueError(f"No rows found for cost scenario: {cost_scenario}")

    summary = (
        df.groupby(["dataset", "downstream_model", "strategy", "budget"], as_index=False)
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            mean_accuracy=("accuracy", "mean"),
            mean_total_cost=("total_cost", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(
            ["dataset", "downstream_model", "mean_macro_f1", "mean_total_cost"],
            ascending=[True, True, False, True],
        )
    )
    summary["std_macro_f1"] = summary["std_macro_f1"].fillna(0.0)
    summary["cost_scenario"] = cost_scenario

    best = (
        summary.sort_values(
            ["dataset", "downstream_model", "mean_macro_f1", "mean_total_cost"],
            ascending=[True, True, False, True],
        )
        .groupby(["dataset", "downstream_model"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best = _add_baseline_deltas(best, baseline_model)

    for frame in (summary, best):
        frame["mean_macro_f1"] = frame["mean_macro_f1"].round(4)
        frame["std_macro_f1"] = frame["std_macro_f1"].round(4)
        frame["mean_accuracy"] = frame["mean_accuracy"].round(4)
        frame["mean_total_cost"] = frame["mean_total_cost"].round(2)
        if "baseline_best_macro_f1" in frame.columns:
            frame["baseline_best_macro_f1"] = frame["baseline_best_macro_f1"].round(4)
        if "delta_macro_f1_vs_baseline" in frame.columns:
            frame["delta_macro_f1_vs_baseline"] = frame["delta_macro_f1_vs_baseline"].round(4)

    summary_path = Path(summary_csv)
    best_path = Path(best_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    best.to_csv(best_path, index=False)
    return summary, best


def _add_baseline_deltas(best: pd.DataFrame, baseline_model: str) -> pd.DataFrame:
    baseline = best[best["downstream_model"] == baseline_model][
        ["dataset", "mean_macro_f1"]
    ].rename(columns={"mean_macro_f1": "baseline_best_macro_f1"})
    result = best.merge(baseline, on="dataset", how="left")
    result["delta_macro_f1_vs_baseline"] = (
        result["mean_macro_f1"] - result["baseline_best_macro_f1"]
    )
    return result


def main() -> None:
    args = parse_args()
    summary, best = make_downstream_model_comparison(
        args.results,
        args.summary_csv,
        args.best_csv,
        args.cost_scenario,
        args.baseline_model,
    )
    print(f"Wrote {len(summary)} rows to {args.summary_csv}")
    print(f"Wrote {len(best)} rows to {args.best_csv}")


if __name__ == "__main__":
    main()
