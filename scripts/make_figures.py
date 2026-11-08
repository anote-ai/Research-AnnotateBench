#!/usr/bin/env python3
"""Generate learning-curve and Pareto figures from benchmark CSV results."""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "annotatebench-matplotlib"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from annotatebench.evaluate import pareto_frontier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/benchmark_results.csv")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--cost-scenario", default="base")
    parser.add_argument("--llm-review-fraction", type=float, default=0.0)
    parser.add_argument("--human-review-cost-per-label", type=float, default=0.0952)
    return parser.parse_args()


def make_figures(
    results_csv: str | Path,
    figures_dir: str | Path,
    cost_scenario: str = "base",
    llm_review_fraction: float = 0.0,
    human_review_cost_per_label: float = 0.0952,
) -> list[Path]:
    results_path = Path(results_csv)
    output_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_path)
    required = {"dataset", "strategy", "budget", "seed", "macro_f1", "total_cost"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")
    if "cost_scenario" in df.columns:
        df = df[df["cost_scenario"] == cost_scenario].copy()
        if df.empty:
            raise ValueError(f"No rows found for cost scenario: {cost_scenario}")

    written: list[Path] = []
    for dataset, dataset_df in df.groupby("dataset"):
        slug = str(dataset).replace(" ", "_").lower()
        title_context = _title_context(dataset_df)
        summary = (
            dataset_df.assign(
                cost_estimation_method=dataset_df.get("cost_estimation_method", "human_cost_scenario")
            )
            .groupby(["strategy", "budget", "cost_estimation_method"], as_index=False)
            .agg(
                macro_f1=("macro_f1", "mean"),
                macro_f1_std=("macro_f1", "std"),
                total_cost=("total_cost", "mean"),
            )
        )
        summary["macro_f1_std"] = summary["macro_f1_std"].fillna(0.0)
        summary = add_review_cost_sensitivity(
            summary,
            review_fraction=llm_review_fraction,
            human_cost_per_label=human_review_cost_per_label,
        )
        written.append(_plot_learning_curve(summary, slug, output_dir, title_context))
        written.append(_plot_pareto(summary, slug, output_dir, title_context))
    return written


def add_review_cost_sensitivity(
    summary: pd.DataFrame,
    *,
    review_fraction: float,
    human_cost_per_label: float,
) -> pd.DataFrame:
    if not 0.0 <= review_fraction <= 1.0:
        raise ValueError("review_fraction must be in [0, 1].")
    if human_cost_per_label < 0:
        raise ValueError("human_cost_per_label must be non-negative.")
    output = summary.copy()
    output["frontier_eligible"] = True
    if review_fraction == 0 or "llm_annotator" not in set(output["strategy"]):
        return output
    llm_mask = output["strategy"] == "llm_annotator"
    output.loc[llm_mask, "frontier_eligible"] = False
    reviewed = output[llm_mask].copy()
    reviewed["strategy"] = f"llm_annotator_review_{int(review_fraction * 100)}pct"
    reviewed["cost_estimation_method"] = "review_sensitivity"
    reviewed["total_cost"] = reviewed["total_cost"] + (
        reviewed["budget"] * review_fraction * human_cost_per_label
    )
    reviewed["frontier_eligible"] = True
    return pd.concat([output, reviewed], ignore_index=True)


def _title_context(df: pd.DataFrame) -> str:
    strategies = set(df["strategy"].dropna().astype(str))
    if strategies == {"llm_annotator"}:
        return "LLM annotator seed-0 benchmark"
    if "llm_annotator" in strategies:
        return "gold vs. LLM"
    return "gold-label reveal simulation"


def _plot_learning_curve(
    df: pd.DataFrame,
    dataset_slug: str,
    output_dir: Path,
    title_context: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for strategy, strategy_df in df.groupby("strategy"):
        ordered = strategy_df.sort_values("budget")
        ax.errorbar(
            ordered["budget"],
            ordered["macro_f1"],
            yerr=ordered["macro_f1_std"],
            marker="o",
            capsize=3,
            linewidth=1.5,
            label=strategy,
        )
    ax.set_xlabel("Number of labeled examples")
    ax.set_ylabel("Macro F1")
    ax.set_title(f"Learning curves: {dataset_slug} {title_context}")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output = output_dir / f"learning_curve_{dataset_slug}.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _plot_pareto(
    df: pd.DataFrame,
    dataset_slug: str,
    output_dir: Path,
    title_context: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    marker_by_method = {
        "measured": "o",
        "estimated_from_stratified_sample": "s",
        "human_cost_scenario": "^",
        "review_sensitivity": "D",
    }
    for (strategy, method), strategy_df in df.groupby(["strategy", "cost_estimation_method"]):
        marker = marker_by_method.get(str(method), "o")
        if strategy == "llm_annotator":
            label = "LLM estimated" if method == "estimated_from_stratified_sample" else "LLM measured"
            if "frontier_eligible" in strategy_df and not strategy_df["frontier_eligible"].any():
                label += " (API only)"
        elif strategy == "llm_annotator_review_10pct":
            label = "LLM + 10% human review"
        else:
            label = str(strategy)
        ax.scatter(strategy_df["total_cost"], strategy_df["macro_f1"], label=label, s=36, marker=marker)
        for _, row in strategy_df.iterrows():
            ax.annotate(
                str(int(row["budget"])),
                (row["total_cost"], row["macro_f1"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )

    frontier_frame = df[df.get("frontier_eligible", True)]
    frontier = pareto_frontier(list(zip(frontier_frame["total_cost"], frontier_frame["macro_f1"])))
    if frontier:
        costs, f1s = zip(*frontier)
        ax.plot(costs, f1s, color="black", linewidth=1.5, label="Pareto frontier")
    positive_costs = df.loc[df["total_cost"] > 0, "total_cost"]
    if len(positive_costs) and positive_costs.max() / positive_costs.min() > 100:
        ax.set_xscale("log")
    ax.set_xlabel("Estimated annotation cost (USD)")
    ax.set_ylabel("Macro F1")
    ax.set_title(f"Cost-performance frontier: {dataset_slug} {title_context}")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output = output_dir / f"pareto_{dataset_slug}.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> None:
    args = parse_args()
    written = make_figures(
        args.results,
        args.figures_dir,
        args.cost_scenario,
        args.llm_review_fraction,
        args.human_review_cost_per_label,
    )
    print("Wrote figures:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
