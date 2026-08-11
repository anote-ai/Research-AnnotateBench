#!/usr/bin/env python3
"""Compare the two best gold-label configurations with paired seed statistics."""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/benchmark_results.csv")
    parser.add_argument(
        "--output-csv",
        default="results/paired_strategy_statistics.csv",
    )
    parser.add_argument("--cost-scenario", default="base")
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def exact_sign_flip_p_value(differences: np.ndarray) -> float:
    observed = abs(float(np.mean(differences)))
    null = [
        abs(float(np.mean(differences * np.asarray(signs))))
        for signs in product((-1.0, 1.0), repeat=len(differences))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def holm_adjust(p_values: pd.Series) -> pd.Series:
    order = np.argsort(p_values.to_numpy())
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values.iloc[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return pd.Series(adjusted, index=p_values.index)


def make_statistics(
    results: pd.DataFrame,
    *,
    cost_scenario: str = "base",
    bootstrap_resamples: int = 20_000,
    seed: int = 2026,
) -> pd.DataFrame:
    required = {
        "dataset",
        "strategy",
        "budget",
        "seed",
        "macro_f1",
        "cost_scenario",
        "total_cost",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")
    frame = results[results["cost_scenario"] == cost_scenario].copy()
    if frame.empty:
        raise ValueError(f"No rows for cost scenario: {cost_scenario}")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for dataset, group in frame.groupby("dataset", sort=True):
        means = (
            group.groupby(["strategy", "budget"], as_index=False)
            .agg(mean_macro_f1=("macro_f1", "mean"), mean_cost=("total_cost", "mean"))
            .sort_values(
                ["mean_macro_f1", "mean_cost", "strategy"],
                ascending=[False, True, True],
            )
            .head(2)
            .reset_index(drop=True)
        )
        if len(means) != 2:
            raise ValueError(f"Need at least two configurations for {dataset}")
        top, runner = means.iloc[0], means.iloc[1]
        top_scores = group[
            (group["strategy"] == top["strategy"])
            & (group["budget"] == top["budget"])
        ].set_index("seed")["macro_f1"]
        runner_scores = group[
            (group["strategy"] == runner["strategy"])
            & (group["budget"] == runner["budget"])
        ].set_index("seed")["macro_f1"]
        paired = pd.concat([top_scores, runner_scores], axis=1, join="inner").dropna()
        paired.columns = ["top", "runner"]
        differences = (paired["top"] - paired["runner"]).to_numpy(dtype=float)
        if len(differences) != 5:
            raise ValueError(f"Expected five paired seeds for {dataset}")
        draws = differences[
            rng.integers(0, len(differences), size=(bootstrap_resamples, len(differences)))
        ].mean(axis=1)
        standard_deviation = float(np.std(differences, ddof=1))
        rows.append(
            {
                "dataset": dataset,
                "top_strategy": top["strategy"],
                "top_budget": int(top["budget"]),
                "runner_strategy": runner["strategy"],
                "runner_budget": int(runner["budget"]),
                "paired_mean_difference": float(np.mean(differences)),
                "bootstrap_ci_lower_95": float(np.quantile(draws, 0.025)),
                "bootstrap_ci_upper_95": float(np.quantile(draws, 0.975)),
                "paired_effect_size_dz": (
                    float(np.mean(differences) / standard_deviation)
                    if standard_deviation > 0
                    else np.nan
                ),
                "exact_sign_flip_p": exact_sign_flip_p_value(differences),
                "n_paired_seeds": len(differences),
            }
        )
    result = pd.DataFrame(rows)
    result["holm_adjusted_p"] = holm_adjust(result["exact_sign_flip_p"])
    numeric = [
        "paired_mean_difference",
        "bootstrap_ci_lower_95",
        "bootstrap_ci_upper_95",
        "paired_effect_size_dz",
        "exact_sign_flip_p",
        "holm_adjusted_p",
    ]
    result[numeric] = result[numeric].round(4)
    return result


def main() -> None:
    args = parse_args()
    result = make_statistics(
        pd.read_csv(args.results),
        cost_scenario=args.cost_scenario,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Wrote {len(result)} rows to {output}")
    print(
        "Holm-significant comparisons: "
        f"{int((result['holm_adjusted_p'] < 0.05).sum())}/{len(result)}"
    )


if __name__ == "__main__":
    main()
