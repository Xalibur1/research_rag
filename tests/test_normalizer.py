"""Tests for compare.normalizer."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compare.normalizer import normalize_metrics, normalize_paper, normalize_papers_batch


@pytest.fixture
def mapping():
    return {
        "acc": "accuracy",
        "f1_score": "f1",
        "t_in": "tokens_in",
        "t_out": "tokens_out",
        "latency": "latency_ms",
        "cost": "cost_usd",
    }


class TestNormalizeMetrics:
    def test_legacy_names_remapped(self, mapping):
        raw = {"acc": 0.9, "f1_score": 0.85}
        normed, warnings = normalize_metrics(raw, mapping)
        assert normed == {"accuracy": 0.9, "f1": 0.85}
        assert warnings == []

    def test_unknown_names_preserved(self, mapping):
        raw = {"bleu": 28.0, "custom_metric": 0.5}
        normed, warnings = normalize_metrics(raw, mapping)
        assert "bleu" in normed
        assert "custom_metric" in normed
        assert warnings == []

    def test_collision_warns(self, mapping):
        raw = {"acc": 0.9, "accuracy": 0.95}
        normed, warnings = normalize_metrics(raw, mapping)
        assert len(warnings) == 1
        assert "Collision" in warnings[0]


class TestNormalizePaper:
    def test_metrics_normalized(self, mapping):
        paper = {
            "paper_id": "test",
            "metrics": {"acc": 0.9, "latency": 100},
        }
        normed, warnings = normalize_paper(paper, mapping)
        assert normed["metrics"]["accuracy"] == 0.9
        assert normed["metrics"]["latency_ms"] == 100

    def test_missing_canonical_filled(self, mapping):
        paper = {"paper_id": "test", "metrics": {"accuracy": 0.9}}
        normed, warnings = normalize_paper(
            paper, mapping, canonical_metrics=["accuracy", "f1", "latency_ms"]
        )
        assert normed["metrics"]["f1"] is None
        assert normed["metrics"]["latency_ms"] is None
        assert any("f1" in w for w in warnings)

    def test_year_converted_to_int(self, mapping):
        paper = {"paper_id": "test", "year": "2025", "metrics": {}}
        normed, _ = normalize_paper(paper, mapping)
        assert normed["year"] == 2025

    def test_dataset_string_to_list(self, mapping):
        paper = {"paper_id": "test", "dataset": "SST-2", "metrics": {}}
        normed, _ = normalize_paper(paper, mapping)
        assert normed["dataset"] == ["SST-2"]

    def test_original_paper_not_mutated(self, mapping):
        paper = {"paper_id": "test", "metrics": {"acc": 0.9}}
        normalize_paper(paper, mapping)
        assert "acc" in paper["metrics"]  # original unchanged


class TestNormalizePapersBatch:
    def test_batch_returns_all_papers(self, mapping):
        papers = [
            {"paper_id": "a", "metrics": {"acc": 0.9}},
            {"paper_id": "b", "metrics": {"f1_score": 0.8}},
        ]
        normed, warnings = normalize_papers_batch(papers, mapping)
        assert len(normed) == 2
        assert normed[0]["metrics"]["accuracy"] == 0.9
        assert normed[1]["metrics"]["f1"] == 0.8
