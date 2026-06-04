"""Tests for CostModel, UncertaintySamplingMetrics, cost_efficiency_curve, annotation_roi."""
from __future__ import annotations

import pytest

from annotatebench.core import (
    AnnotationStrategy,
    CostModel,
    NLPTask,
    UncertaintySamplingMetrics,
)
from annotatebench.data import make_learning_curve
from annotatebench.evaluate import annotation_roi, cost_efficiency_curve


# ---------------------------------------------------------------------------
# CostModel
# ---------------------------------------------------------------------------

class TestCostModel:
    def test_total_api_cost_zero_review(self) -> None:
        model = CostModel(api_cost_per_sample=0.01, human_cost_per_sample=0.10)
        assert model.total_api_cost(100) == pytest.approx(1.0)

    def test_total_human_cost(self) -> None:
        model = CostModel(api_cost_per_sample=0.01, human_cost_per_sample=0.10)
        assert model.total_human_cost(50) == pytest.approx(5.0)

    def test_cheaper_option_api(self) -> None:
        model = CostModel(api_cost_per_sample=0.01, human_cost_per_sample=0.10)
        assert model.cheaper_option(100) == "api"

    def test_cheaper_option_human(self) -> None:
        model = CostModel(api_cost_per_sample=0.50, human_cost_per_sample=0.10)
        assert model.cheaper_option(100) == "human"

    def test_quality_adjusted_cost_api_lower_than_human(self) -> None:
        # API is cheaper per sample but has higher error rate
        model = CostModel(
            api_cost_per_sample=0.01,
            human_cost_per_sample=0.10,
            api_error_rate=0.10,
            human_error_rate=0.01,
        )
        api_qac = model.quality_adjusted_cost(1000, use_api=True)
        human_qac = model.quality_adjusted_cost(1000, use_api=False)
        # API quality-adjusted cost should be smaller (cheaper, even after error penalty)
        assert api_qac < human_qac

    def test_invalid_error_rate_raises(self) -> None:
        with pytest.raises(ValueError):
            CostModel(api_cost_per_sample=0.01, human_cost_per_sample=0.10, api_error_rate=1.5)

    def test_review_cost_included(self) -> None:
        model = CostModel(
            api_cost_per_sample=0.01,
            human_cost_per_sample=0.10,
            api_error_rate=0.10,
            review_cost_per_sample=1.0,
        )
        # 100 samples * 0.01 + round(0.10 * 100) * 1.0 = 1.0 + 10.0 = 11.0
        assert model.total_api_cost(100) == pytest.approx(11.0)


# ---------------------------------------------------------------------------
# UncertaintySamplingMetrics
# ---------------------------------------------------------------------------

class TestUncertaintySamplingMetrics:
    def _make_metrics(self) -> UncertaintySamplingMetrics:
        entropy = [0.9, 0.8, 0.3, 0.2, 0.7, 0.1]
        selected = [True, True, False, False, True, False]
        labels = [1, 0, 1, 0, 1, 0]
        return UncertaintySamplingMetrics(
            entropy_scores=entropy,
            selected_mask=selected,
            oracle_labels=labels,
        )

    def test_mean_entropy_selected(self) -> None:
        m = self._make_metrics()
        # selected entropies: 0.9, 0.8, 0.7  -> mean = 0.8
        assert m.mean_entropy_selected() == pytest.approx(0.8)

    def test_mean_entropy_unselected(self) -> None:
        m = self._make_metrics()
        # unselected: 0.3, 0.2, 0.1 -> mean = 0.2
        assert m.mean_entropy_unselected() == pytest.approx(0.2)

    def test_informativeness_gain_positive(self) -> None:
        m = self._make_metrics()
        assert m.informativeness_gain() > 0

    def test_selection_rate(self) -> None:
        m = self._make_metrics()
        assert m.selection_rate() == pytest.approx(0.5)

    def test_class_balance_selected(self) -> None:
        m = self._make_metrics()
        balance = m.class_balance_selected()
        assert set(balance.keys()) == {0, 1}
        assert sum(balance.values()) == pytest.approx(1.0)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            UncertaintySamplingMetrics(
                entropy_scores=[0.5, 0.6],
                selected_mask=[True],
                oracle_labels=[1, 0],
            )


# ---------------------------------------------------------------------------
# cost_efficiency_curve
# ---------------------------------------------------------------------------

class TestCostEfficiencyCurve:
    def test_returns_correct_length(self) -> None:
        curve = make_learning_curve(AnnotationStrategy.RANDOM, NLPTask.CLASSIFICATION)
        pairs = cost_efficiency_curve(curve)
        assert len(pairs) == len(curve.points)

    def test_cumulative_cost_monotone(self) -> None:
        curve = make_learning_curve(AnnotationStrategy.HYBRID_AL, NLPTask.NER)
        pairs = cost_efficiency_curve(curve)
        costs = [c for c, _ in pairs]
        assert all(costs[i] <= costs[i + 1] for i in range(len(costs) - 1))

    def test_f1_values_in_range(self) -> None:
        curve = make_learning_curve(AnnotationStrategy.LLM_ANNOTATOR, NLPTask.QA)
        for _, f1 in cost_efficiency_curve(curve):
            assert 0.0 <= f1 <= 1.0


# ---------------------------------------------------------------------------
# annotation_roi
# ---------------------------------------------------------------------------

class TestAnnotationROI:
    def test_keys_present(self) -> None:
        curve = make_learning_curve(AnnotationStrategy.UNCERTAINTY_AL, NLPTask.CLASSIFICATION)
        model = CostModel(api_cost_per_sample=0.02, human_cost_per_sample=0.15)
        roi = annotation_roi(curve, model, baseline_f1=0.4)
        assert set(roi.keys()) == {"f1_gain", "total_cost_usd", "f1_per_dollar", "cost_per_f1_point"}

    def test_f1_gain_nonnegative(self) -> None:
        curve = make_learning_curve(AnnotationStrategy.RANDOM, NLPTask.CLASSIFICATION)
        model = CostModel(api_cost_per_sample=0.02, human_cost_per_sample=0.15)
        roi = annotation_roi(curve, model, baseline_f1=0.0)
        assert roi["f1_gain"] >= 0.0

    def test_f1_per_dollar_positive(self) -> None:
        curve = make_learning_curve(AnnotationStrategy.HYBRID_AL, NLPTask.CLASSIFICATION)
        model = CostModel(api_cost_per_sample=0.01, human_cost_per_sample=0.10)
        roi = annotation_roi(curve, model, baseline_f1=0.3)
        assert roi["f1_per_dollar"] > 0.0

    def test_high_baseline_clips_gain(self) -> None:
        curve = make_learning_curve(AnnotationStrategy.RANDOM, NLPTask.CLASSIFICATION)
        model = CostModel(api_cost_per_sample=0.02, human_cost_per_sample=0.15)
        roi = annotation_roi(curve, model, baseline_f1=0.99)
        assert roi["f1_gain"] == 0.0
