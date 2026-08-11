from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_downstream_model_robustness_summary import (
    make_merged_embedding_results,
    make_summary,
)


def _write_results(path: Path, model_shift: float, seeds: range = range(5)) -> None:
    rows = []
    for seed in seeds:
        for budget in (50, 100, 250, 500, 1000):
            for strategy, score in (
                ("random", 0.60),
                ("uncertainty_al", 0.62),
                ("diversity_al", 0.64),
                ("hybrid_al", 0.70),
            ):
                rows.append(
                    {
                        "dataset": "demo",
                        "strategy": strategy,
                        "budget": budget,
                        "seed": seed,
                        "macro_f1": score + model_shift + budget / 100000,
                        "cost_scenario": "base",
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_make_summary_reports_cross_model_agreement(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    embedding = tmp_path / "embedding.csv"
    _write_results(baseline, 0.0)
    _write_results(embedding, 0.1)

    summary = make_summary([str(baseline)], [str(embedding)])

    assert len(summary) == 1
    assert summary.loc[0, "best_strategy_agrees"]
    assert summary.loc[0, "tfidf_best_strategy"] == "hybrid_al"
    assert summary.loc[0, "embedding_best_macro_f1"] == pytest.approx(0.81)
    assert summary.loc[
        0, "embedding_minus_tfidf_best_macro_f1"
    ] == pytest.approx(0.1)


def test_make_summary_requires_all_five_seeds(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    embedding = tmp_path / "embedding.csv"
    _write_results(baseline, 0.0)
    _write_results(embedding, 0.1, seeds=range(3))

    with pytest.raises(ValueError, match="exactly seeds"):
        make_summary([str(baseline)], [str(embedding)])


def test_make_summary_rejects_condition_with_only_three_seeds(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    embedding = tmp_path / "embedding.csv"
    _write_results(baseline, 0.0)
    _write_results(embedding, 0.1)
    frame = pd.read_csv(embedding)
    frame = frame[
        ~(
            (frame["strategy"] == "hybrid_al")
            & (frame["budget"] == 1000)
            & (frame["seed"].isin([3, 4]))
        )
    ]
    frame.to_csv(embedding, index=False)

    with pytest.raises(ValueError, match="incomplete five-seed conditions"):
        make_summary([str(baseline)], [str(embedding)])


def test_merge_embedding_results_requires_and_preserves_three_scenarios(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.csv"
    complete = tmp_path / "complete.csv"
    _write_results(base, 0.0)
    frame = pd.read_csv(base)
    pd.concat(
        [
            frame.assign(cost_scenario=scenario)
            for scenario in ("low", "base", "high")
        ],
        ignore_index=True,
    ).to_csv(complete, index=False)

    merged = make_merged_embedding_results([str(complete)])

    assert len(merged) == 300
    assert set(merged["cost_scenario"]) == {"low", "base", "high"}
