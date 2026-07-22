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
    return parser.parse_args()


def make_figures(
    results_csv: str | Path,
    figures_dir: str | Path,
    cost_scenario: str = "base",
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
        written.append(_plot_learning_curve(summary, slug, output_dir, title_context))
        written.append(_plot_pareto(summary, slug, output_dir, title_context))
    return written


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
    }
    for (strategy, method), strategy_df in df.groupby(["strategy", "cost_estimation_method"]):
        marker = marker_by_method.get(str(method), "o")
        if strategy == "llm_annotator":
            label = "LLM estimated" if method == "estimated_from_stratified_sample" else "LLM measured"
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

    frontier = pareto_frontier(list(zip(df["total_cost"], df["macro_f1"])))
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
    written = make_figures(args.results, args.figures_dir, args.cost_scenario)
    print("Wrote figures:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
