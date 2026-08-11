#!/usr/bin/env python3
"""Summarize cross-model robustness for the benchmark report."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


BASELINE_MODEL = "tfidf_logreg"
EMBEDDING_MODEL = "sentence_transformer_logreg"
REQUIRED_SEEDS = {0, 1, 2, 3, 4}
REQUIRED_STRATEGIES = {
    "random",
    "uncertainty_al",
    "diversity_al",
    "hybrid_al",
}
REQUIRED_BUDGETS = {50, 100, 250, 500, 1000}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--embedding-results", nargs="+", required=True)
    parser.add_argument(
        "--output-csv",
        default="results/downstream_model_robustness_summary.csv",
    )
    parser.add_argument(
        "--condition-output-csv",
        default="results/downstream_model_condition_summary_5seed.csv",
    )
    parser.add_argument(
        "--merged-embedding-output-csv",
        default="results/benchmark_results_sentence_transformer_logreg_5seed.csv",
    )
    parser.add_argument("--cost-scenario", default="base")
    return parser.parse_args()


def _load_results(paths: list[str], model: str, cost_scenario: str) -> pd.DataFrame:
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    required = {"dataset", "strategy", "budget", "seed", "macro_f1", "cost_scenario"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")
    frame = frame[frame["cost_scenario"] == cost_scenario].copy()
    if frame.empty:
        raise ValueError(f"No rows found for cost scenario: {cost_scenario}")
    frame["downstream_model"] = model
    frame = frame.drop_duplicates(
        ["dataset", "strategy", "budget", "seed", "cost_scenario"],
        keep="last",
    )
    return frame


def _validate_grid(frame: pd.DataFrame, model: str) -> None:
    seeds = set(frame["seed"].astype(int).unique())
    if seeds != REQUIRED_SEEDS:
        raise ValueError(
            f"{model} must contain exactly seeds {sorted(REQUIRED_SEEDS)}; "
            f"found {sorted(seeds)}"
        )
    strategies = set(frame["strategy"].unique())
    if strategies != REQUIRED_STRATEGIES:
        raise ValueError(
            f"{model} must contain strategies {sorted(REQUIRED_STRATEGIES)}; "
            f"found {sorted(strategies)}"
        )
    budgets = set(frame["budget"].astype(int).unique())
    if budgets != REQUIRED_BUDGETS:
        raise ValueError(
            f"{model} must contain budgets {sorted(REQUIRED_BUDGETS)}; "
            f"found {sorted(budgets)}"
        )
    condition_seed_counts = frame.groupby(
        ["dataset", "strategy", "budget"]
    )["seed"].nunique()
    incomplete = condition_seed_counts[
        condition_seed_counts != len(REQUIRED_SEEDS)
    ]
    if not incomplete.empty:
        preview = ", ".join(
            f"{dataset}/{strategy}/{budget}={count}"
            for (dataset, strategy, budget), count in incomplete.head(5).items()
        )
        raise ValueError(
            f"{model} has incomplete five-seed conditions: {preview}"
        )


def make_summary(
    baseline_paths: list[str],
    embedding_paths: list[str],
    *,
    cost_scenario: str = "base",
) -> pd.DataFrame:
    baseline = _load_results(baseline_paths, BASELINE_MODEL, cost_scenario)
    embedding = _load_results(embedding_paths, EMBEDDING_MODEL, cost_scenario)

    for name, frame in ((BASELINE_MODEL, baseline), (EMBEDDING_MODEL, embedding)):
        _validate_grid(frame, name)
    if set(baseline["dataset"].unique()) != set(embedding["dataset"].unique()):
        missing_embedding = sorted(
            set(baseline["dataset"].unique()) - set(embedding["dataset"].unique())
        )
        missing_baseline = sorted(
            set(embedding["dataset"].unique()) - set(baseline["dataset"].unique())
        )
        raise ValueError(
            "Downstream-model dataset grids differ; "
            f"missing from embedding={missing_embedding}, "
            f"missing from baseline={missing_baseline}"
        )

    combined = pd.concat([baseline, embedding], ignore_index=True)
    condition_means = (
        combined.groupby(
            ["dataset", "downstream_model", "strategy", "budget"],
            as_index=False,
        )
        .agg(mean_macro_f1=("macro_f1", "mean"))
    )
    strategy_best = (
        condition_means.sort_values(
            ["dataset", "downstream_model", "strategy", "mean_macro_f1", "budget"],
            ascending=[True, True, True, False, True],
        )
        .groupby(["dataset", "downstream_model", "strategy"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    rows: list[dict[str, object]] = []
    for dataset, group in strategy_best.groupby("dataset", sort=True):
        by_model = {
            model: model_group.set_index("strategy")
            for model, model_group in group.groupby("downstream_model")
        }
        if set(by_model) != {BASELINE_MODEL, EMBEDDING_MODEL}:
            raise ValueError(f"Both downstream models are required for {dataset}")

        baseline_group = by_model[BASELINE_MODEL]
        embedding_group = by_model[EMBEDDING_MODEL]
        strategies = sorted(set(baseline_group.index) & set(embedding_group.index))
        baseline_best = baseline_group.loc[
            baseline_group["mean_macro_f1"].idxmax()
        ]
        embedding_best = embedding_group.loc[
            embedding_group["mean_macro_f1"].idxmax()
        ]
        correlation = spearmanr(
            baseline_group.loc[strategies, "mean_macro_f1"],
            embedding_group.loc[strategies, "mean_macro_f1"],
        ).statistic
        rows.append(
            {
                "dataset": dataset,
                "tfidf_best_strategy": baseline_best.name,
                "tfidf_best_budget": int(baseline_best["budget"]),
                "tfidf_best_macro_f1": float(baseline_best["mean_macro_f1"]),
                "embedding_best_strategy": embedding_best.name,
                "embedding_best_budget": int(embedding_best["budget"]),
                "embedding_best_macro_f1": float(
                    embedding_best["mean_macro_f1"]
                ),
                "embedding_minus_tfidf_best_macro_f1": float(
                    embedding_best["mean_macro_f1"]
                    - baseline_best["mean_macro_f1"]
                ),
                "best_strategy_agrees": baseline_best.name == embedding_best.name,
                "strategy_rank_spearman": float(correlation),
            }
        )

    result = pd.DataFrame(rows)
    numeric_columns = [
        "tfidf_best_macro_f1",
        "embedding_best_macro_f1",
        "embedding_minus_tfidf_best_macro_f1",
        "strategy_rank_spearman",
    ]
    result[numeric_columns] = result[numeric_columns].round(4)
    return result


def make_condition_summary(
    baseline_paths: list[str],
    embedding_paths: list[str],
    *,
    cost_scenario: str = "base",
) -> pd.DataFrame:
    baseline = _load_results(baseline_paths, BASELINE_MODEL, cost_scenario)
    embedding = _load_results(embedding_paths, EMBEDDING_MODEL, cost_scenario)
    for name, frame in ((BASELINE_MODEL, baseline), (EMBEDDING_MODEL, embedding)):
        _validate_grid(frame, name)
    result = (
        pd.concat([baseline, embedding], ignore_index=True)
        .groupby(
            ["dataset", "downstream_model", "strategy", "budget"],
            as_index=False,
        )
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["dataset", "downstream_model", "strategy", "budget"])
        .reset_index(drop=True)
    )
    result[["mean_macro_f1", "std_macro_f1"]] = result[
        ["mean_macro_f1", "std_macro_f1"]
    ].round(4)
    return result


def make_merged_embedding_results(paths: list[str]) -> pd.DataFrame:
    result = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    key_columns = [
        "dataset",
        "strategy",
        "budget",
        "seed",
        "cost_scenario",
    ]
    missing = set(key_columns).difference(result.columns)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")
    result = result.drop_duplicates(key_columns, keep="last")
    scenarios = set(result["cost_scenario"].unique())
    if scenarios != {"low", "base", "high"}:
        raise ValueError(
            "Merged embedding results must contain low, base, and high "
            f"cost scenarios; found {sorted(scenarios)}"
        )
    for scenario in sorted(scenarios):
        _validate_grid(
            result[result["cost_scenario"] == scenario],
            f"{EMBEDDING_MODEL}/{scenario}",
        )
    return result.sort_values(key_columns).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    result = make_summary(
        [args.baseline_results],
        args.embedding_results,
        cost_scenario=args.cost_scenario,
    )
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    condition_result = make_condition_summary(
        [args.baseline_results],
        args.embedding_results,
        cost_scenario=args.cost_scenario,
    )
    condition_output_path = Path(args.condition_output_csv)
    condition_output_path.parent.mkdir(parents=True, exist_ok=True)
    condition_result.to_csv(condition_output_path, index=False)
    merged_embedding_result = make_merged_embedding_results(
        args.embedding_results
    )
    merged_embedding_output_path = Path(args.merged_embedding_output_csv)
    merged_embedding_output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_embedding_result.to_csv(merged_embedding_output_path, index=False)
    print(f"Wrote {len(result)} rows to {output_path}")
    print(
        f"Wrote {len(condition_result)} rows to {condition_output_path}"
    )
    print(
        f"Wrote {len(merged_embedding_result)} rows to "
        f"{merged_embedding_output_path}"
    )
    print(
        "Best-strategy agreement: "
        f"{int(result['best_strategy_agrees'].sum())}/{len(result)}"
    )


if __name__ == "__main__":
    main()
