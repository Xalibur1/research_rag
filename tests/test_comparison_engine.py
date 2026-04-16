"""Tests for compare.comparison_engine."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compare.comparison_engine import (
    build_metric_table,
    collect_all_metrics,
    compute_pairwise_deltas,
    compute_rankings,
    compute_summary_stats,
    filter_papers,
    find_leaders,
    group_papers,
    parse_filters,
    sort_papers,
)


@pytest.fixture
def sample_papers():
    return [
        {"paper_id": "a", "task": "qa", "model": "M1", "year": 2024,
         "metrics": {"accuracy": 0.90, "latency_ms": 100}},
        {"paper_id": "b", "task": "qa", "model": "M2", "year": 2025,
         "metrics": {"accuracy": 0.85, "latency_ms": 50}},
        {"paper_id": "c", "task": "sum", "model": "M1", "year": 2025,
         "metrics": {"accuracy": 0.70, "latency_ms": 200, "bleu": 30.0}},
    ]


@pytest.fixture
def table(sample_papers):
    return build_metric_table(sample_papers, ["accuracy", "latency_ms"])


class TestCollectAllMetrics:
    def test_discovers_all_keys(self, sample_papers):
        names = collect_all_metrics(sample_papers)
        assert "accuracy" in names
        assert "latency_ms" in names
        assert "bleu" in names

    def test_sorted(self, sample_papers):
        names = collect_all_metrics(sample_papers)
        assert names == sorted(names)


class TestBuildMetricTable:
    def test_correct_shape(self, table):
        assert len(table) == 3
        for row in table.values():
            assert "accuracy" in row
            assert "latency_ms" in row

    def test_missing_metric_is_none(self, sample_papers):
        table = build_metric_table(sample_papers, ["bleu"])
        assert table["a"]["bleu"] is None
        assert table["c"]["bleu"] == 30.0


class TestComputeRankings:
    def test_accuracy_desc(self, table):
        rankings = compute_rankings(table, "accuracy")
        assert rankings[0]["paper_id"] == "a"
        assert rankings[0]["rank"] == 1
        assert rankings[0]["delta_to_best"] == 0.0
        assert rankings[1]["paper_id"] == "b"

    def test_latency_asc(self, table):
        rankings = compute_rankings(table, "latency_ms")
        assert rankings[0]["paper_id"] == "b"  # 50 is lowest
        assert rankings[0]["value"] == 50

    def test_empty_metric(self):
        table = {"a": {"x": None}, "b": {"x": None}}
        assert compute_rankings(table, "x") == []


class TestComputeSummaryStats:
    def test_correct_stats(self, table):
        stats = compute_summary_stats(table, "accuracy")
        assert stats["count"] == 3
        assert stats["min"] == 0.70
        assert stats["max"] == 0.90

    def test_empty_metric(self):
        table = {"a": {"x": None}}
        stats = compute_summary_stats(table, "x")
        assert stats["count"] == 0
        assert stats["mean"] is None


class TestPairwiseDeltas:
    def test_correct_deltas(self, table):
        deltas = compute_pairwise_deltas(table, "accuracy")
        assert len(deltas) == 3  # 3 choose 2
        # All pairs: (a,b), (a,c), (b,c)
        ab = next(d for d in deltas if d["a_paper_id"] == "a" and d["b_paper_id"] == "b")
        assert ab["delta"] == pytest.approx(0.05, abs=1e-5)

    def test_empty_metric(self):
        table = {"a": {"x": None}, "b": {"x": None}}
        assert compute_pairwise_deltas(table, "x") == []


class TestFindLeaders:
    def test_leaders_found(self, table):
        leaders = find_leaders(table)
        accuracy_leader = next(l for l in leaders if l["metric"] == "accuracy")
        assert accuracy_leader["paper_id"] == "a"
        latency_leader = next(l for l in leaders if l["metric"] == "latency_ms")
        assert latency_leader["paper_id"] == "b"


class TestFilterPapers:
    def test_year_gte(self, sample_papers):
        result = filter_papers(sample_papers, "year>=2025")
        assert len(result) == 2
        assert all(p["year"] >= 2025 for p in result)

    def test_task_eq(self, sample_papers):
        result = filter_papers(sample_papers, "task=qa")
        assert len(result) == 2

    def test_multiple_filters(self, sample_papers):
        result = filter_papers(sample_papers, "year>=2025,task=qa")
        assert len(result) == 1
        assert result[0]["paper_id"] == "b"

    def test_no_filter_returns_all(self, sample_papers):
        assert filter_papers(sample_papers, None) == sample_papers


class TestParseFilters:
    def test_string_parsing(self):
        parsed = parse_filters("year>=2022,task=qa")
        assert len(parsed) == 2

    def test_empty(self):
        assert parse_filters(None) == []
        assert parse_filters("") == []


class TestSortPapers:
    def test_sort_asc(self, sample_papers):
        result = sort_papers(sample_papers, ["accuracy"])
        assert result[0]["paper_id"] == "c"

    def test_sort_desc(self, sample_papers):
        result = sort_papers(sample_papers, ["-accuracy"])
        assert result[0]["paper_id"] == "a"

    def test_default_sorts_by_paper_id(self, sample_papers):
        result = sort_papers(sample_papers)
        assert result[0]["paper_id"] == "a"


class TestGroupPapers:
    def test_group_by_task(self, sample_papers):
        groups = group_papers(sample_papers, ["task"])
        assert "qa" in groups
        assert "sum" in groups
        assert len(groups["qa"]) == 2

    def test_group_by_multiple(self, sample_papers):
        groups = group_papers(sample_papers, ["task", "model"])
        assert len(groups) == 3  # qa|M1, qa|M2, sum|M1

    def test_no_group_returns_all(self, sample_papers):
        groups = group_papers(sample_papers, None)
        assert "all" in groups
        assert len(groups["all"]) == 3
