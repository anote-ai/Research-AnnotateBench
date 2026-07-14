#!/usr/bin/env python3
"""Create statistical appendix tables from row-level annotation outputs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from annotatebench.metrics import bonferroni_significant, gwet_ac1, paired_bootstrap_p_value
from annotatebench.metrics.lari import expected_calibration_error, lari_score


APPENDIX_COLUMNS = [
    "dataset_name",
    "split",
    "metric",
    "group_key",
    "group_value",
    "estimate",
    "ci_lower",
    "ci_upper",
    "n_examples",
    "n_annotators",
    "n_resamples",
    "notes",
]

SIGNIFICANCE_COLUMNS = [
    "dataset_name",
    "split",
    "comparison",
    "metric",
    "delta",
    "p_value",
    "p_value_bonferroni",
    "significant",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", nargs="+", required=True)
    parser.add_argument("--output-csv", default="results/statistical_appendix.csv")
    parser.add_argument("--significance-csv", default="results/statistical_significance.csv")
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    appendix, significance = make_statistical_appendix(
        args.annotations,
        n_resamples=args.n_resamples,
        random_state=args.random_state,
    )
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    appendix.to_csv(output_path, index=False)

    significance_path = Path(args.significance_csv)
    significance_path.parent.mkdir(parents=True, exist_ok=True)
    significance.to_csv(significance_path, index=False)
    print(f"Wrote {len(appendix)} appendix rows to {output_path}")
    print(f"Wrote {len(significance)} significance rows to {significance_path}")


def make_statistical_appendix(
    annotation_paths: list[str] | list[Path],
    *,
    n_resamples: int = 1000,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.concat([pd.read_csv(path) for path in annotation_paths], ignore_index=True)
    required = {
        "dataset_name",
        "split",
        "example_id",
        "annotator_id",
        "gold_label",
        "predicted_label",
        "confidence",
        "correct",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing annotation columns: {sorted(missing)}")

    df["correct_bool"] = df["correct"].map(_parse_bool)
    df["confidence_float"] = df["confidence"].astype(float)

    appendix_rows: list[dict[str, object]] = []
    significance_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(random_state)

    for (dataset_name, split), group in df.groupby(["dataset_name", "split"], sort=True):
        appendix_rows.extend(
            _metric_rows(
                group,
                dataset_name=dataset_name,
                split=split,
                group_key="overall",
                group_value="all",
                n_resamples=n_resamples,
                rng=rng,
            )
        )
        appendix_rows.extend(
            _agreement_rows(
                group,
                dataset_name=dataset_name,
                split=split,
                n_resamples=n_resamples,
            )
        )
        significance_rows.extend(
            _significance_rows(
                group,
                dataset_name=dataset_name,
                split=split,
                n_resamples=n_resamples,
                rng=rng,
            )
        )

    return (
        pd.DataFrame(appendix_rows, columns=APPENDIX_COLUMNS),
        pd.DataFrame(significance_rows, columns=SIGNIFICANCE_COLUMNS),
    )


def _metric_rows(
    group: pd.DataFrame,
    *,
    dataset_name: str,
    split: str,
    group_key: str,
    group_value: str,
    n_resamples: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    metric_functions: dict[str, Callable[[pd.DataFrame], float]] = {
        "accuracy": lambda frame: float(accuracy_score(frame["gold_label"], frame["predicted_label"])),
        "macro_f1": lambda frame: float(
            f1_score(
                frame["gold_label"],
                frame["predicted_label"],
                labels=sorted(set(group["gold_label"]) | set(group["predicted_label"])),
                average="macro",
                zero_division=0,
            )
        ),
        "ece": lambda frame: expected_calibration_error(
            frame["confidence_float"].tolist(),
            frame["correct_bool"].tolist(),
        ),
    }
    metric_functions["lari"] = lambda frame: lari_score(
        metric_functions["macro_f1"](frame),
        frame["confidence_float"].tolist(),
        frame["correct_bool"].tolist(),
    )

    rows = []
    n_annotators = group["annotator_id"].nunique()
    for metric, metric_fn in metric_functions.items():
        estimate = metric_fn(group)
        draws = _bootstrap_metric(group, metric_fn, n_resamples=n_resamples, rng=rng)
        lower, upper = np.quantile(draws, [0.025, 0.975])
        rows.append(
            {
                "dataset_name": dataset_name,
                "split": split,
                "metric": metric,
                "group_key": group_key,
                "group_value": group_value,
                "estimate": estimate,
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "n_examples": group["example_id"].nunique(),
                "n_annotators": n_annotators,
                "n_resamples": n_resamples,
                "notes": "",
            }
        )
    return rows


def _agreement_rows(
    group: pd.DataFrame,
    *,
    dataset_name: str,
    split: str,
    n_resamples: int,
) -> list[dict[str, object]]:
    annotators = sorted(group["annotator_id"].dropna().unique())
    if len(annotators) < 2:
        return [
            {
                "dataset_name": dataset_name,
                "split": split,
                "metric": "gwet_ac1",
                "group_key": "agreement",
                "group_value": "all",
                "estimate": "",
                "ci_lower": "",
                "ci_upper": "",
                "n_examples": group["example_id"].nunique(),
                "n_annotators": len(annotators),
                "n_resamples": n_resamples,
                "notes": "insufficient_annotators",
            }
        ]

    rows = []
    labels = sorted(set(group["gold_label"]) | set(group["predicted_label"]))
    for left_index, left in enumerate(annotators):
        for right in annotators[left_index + 1 :]:
            paired = _paired_annotator_rows(group, left, right)
            if paired.empty:
                continue
            score = gwet_ac1(
                paired["predicted_label_left"].tolist(),
                paired["predicted_label_right"].tolist(),
                labels=labels,
            )
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "split": split,
                    "metric": "gwet_ac1",
                    "group_key": "annotator_pair",
                    "group_value": f"{left} vs {right}",
                    "estimate": score,
                    "ci_lower": "",
                    "ci_upper": "",
                    "n_examples": paired["example_id"].nunique(),
                    "n_annotators": 2,
                    "n_resamples": n_resamples,
                    "notes": "",
                }
            )
    return rows


def _significance_rows(
    group: pd.DataFrame,
    *,
    dataset_name: str,
    split: str,
    n_resamples: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    rows = []
    annotators = sorted(group["annotator_id"].dropna().unique())
    for left_index, left in enumerate(annotators):
        for right in annotators[left_index + 1 :]:
            paired = _paired_annotator_rows(group, left, right)
            if paired.empty:
                continue
            left_correct = paired["correct_bool_left"].astype(float).tolist()
            right_correct = paired["correct_bool_right"].astype(float).tolist()
            p_value = paired_bootstrap_p_value(
                left_correct,
                right_correct,
                n_resamples=n_resamples,
                random_state=int(rng.integers(0, 1_000_000)),
            )
            delta = float(np.mean(right_correct) - np.mean(left_correct))
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "split": split,
                    "comparison": f"{right} minus {left}",
                    "metric": "accuracy",
                    "delta": delta,
                    "p_value": p_value,
                    "p_value_bonferroni": "",
                    "significant": "",
                    "notes": "",
                }
            )
    if rows:
        flags = bonferroni_significant([float(row["p_value"]) for row in rows])
        for row, significant in zip(rows, flags):
            row["p_value_bonferroni"] = min(1.0, float(row["p_value"]) * len(rows))
            row["significant"] = significant
    return rows


def _bootstrap_metric(
    group: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    *,
    n_resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    annotators = list(dict.fromkeys(group["annotator_id"].tolist()))
    draws = np.empty(n_resamples, dtype=float)
    if len(annotators) > 1:
        grouped = {annotator: frame for annotator, frame in group.groupby("annotator_id", sort=False)}
        for index in range(n_resamples):
            selected = rng.choice(annotators, size=len(annotators), replace=True)
            sample = pd.concat([grouped[annotator] for annotator in selected], ignore_index=True)
            draws[index] = metric_fn(sample)
    else:
        for index in range(n_resamples):
            sample = group.sample(n=len(group), replace=True, random_state=int(rng.integers(0, 1_000_000)))
            draws[index] = metric_fn(sample)
    return draws


def _paired_annotator_rows(group: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    left_rows = group[group["annotator_id"] == left][
        ["example_id", "predicted_label", "correct_bool"]
    ].rename(columns={"predicted_label": "predicted_label_left", "correct_bool": "correct_bool_left"})
    right_rows = group[group["annotator_id"] == right][
        ["example_id", "predicted_label", "correct_bool"]
    ].rename(columns={"predicted_label": "predicted_label_right", "correct_bool": "correct_bool_right"})
    return left_rows.merge(right_rows, on="example_id", how="inner")


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


if __name__ == "__main__":
    main()
