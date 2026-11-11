from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_incomplete_row_costs_do_not_create_partial_batch_cost():
    sys.path.insert(0, str(ROOT / "scripts"))
    from recompute_llm_dataset_seed import complete_annotation_cost

    assert complete_annotation_cost(pd.Series([0.01, None])) is None
    assert complete_annotation_cost(pd.Series([0.01, 0.02])) == pytest.approx(0.03)


def test_matched_gold_command_rejects_selected_dry_run(tmp_path):
    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()
    selected = sorted(__import__("numpy").random.default_rng(1).choice(1200, size=250, replace=False))
    rows = [
        {
            "example_id": f"fixture_train_{index}",
            "notes": "dry_run" if position == 0 else "",
            "raw_response": "{}",
        }
        for position, index in enumerate(selected)
    ]
    pd.DataFrame(rows).to_csv(annotation_dir / "fixture_llm_annotator_seed1.csv", index=False)
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_matched_gold_summary import audit_annotation_cache

    try:
        audit_annotation_cache(annotation_dir, ["fixture"], [1], 250)
    except ValueError as exc:
        assert "dry_run" in str(exc)
    else:
        raise AssertionError("Selected dry-run row was not rejected.")
