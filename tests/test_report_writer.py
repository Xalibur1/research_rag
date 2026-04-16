"""Tests for compare.report_writer."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compare.report_writer import build_report_data, write_json_report, write_md_report


@pytest.fixture
def sample_papers():
    return [
        {
            "paper_id": "a",
            "title": "Paper A",
            "model": "M1",
            "task": "qa",
            "dataset": ["D1"],
            "metrics": {"accuracy": 0.90, "latency_ms": 100},
            "inputs": {},
            "outputs": {},
        },
        {
            "paper_id": "b",
            "title": "Paper B",
            "model": "M2",
            "task": "qa",
            "dataset": ["D1"],
            "metrics": {"accuracy": 0.85, "latency_ms": 50},
            "inputs": {},
            "outputs": {},
        },
    ]


@pytest.fixture
def report_data(sample_papers):
    return build_report_data(
        papers=sample_papers,
        errors=[],
        metrics=["accuracy", "latency_ms"],
        input_dir="/test",
        file_count=2,
    )


class TestBuildReportData:
    def test_structure(self, report_data):
        assert "run_info" in report_data
        assert "schema" in report_data
        assert "papers" in report_data
        assert "comparisons" in report_data
        assert "summaries" in report_data
        assert "anomalies" in report_data

    def test_paper_count(self, report_data):
        assert report_data["run_info"]["paper_count"] == 2
        assert len(report_data["papers"]) == 2

    def test_metrics_listed(self, report_data):
        assert "accuracy" in report_data["schema"]["canonical_metrics"]
        assert "latency_ms" in report_data["schema"]["canonical_metrics"]

    def test_leaders_present(self, report_data):
        leaders = report_data["comparisons"]["leaders"]
        assert len(leaders) == 2
        acc_leader = next(l for l in leaders if l["metric"] == "accuracy")
        assert acc_leader["paper_id"] == "a"

    def test_pairwise_deltas(self, report_data):
        deltas = report_data["comparisons"]["pairwise_deltas"]
        assert len(deltas) > 0

    def test_per_metric_tables(self, report_data):
        tables = report_data["comparisons"]["per_metric_tables"]
        assert len(tables) == 2


class TestWriteJsonReport:
    def test_writes_valid_json(self, report_data, tmp_path):
        out = tmp_path / "report.json"
        write_json_report(report_data, str(out))
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["run_info"]["paper_count"] == 2

    def test_deterministic(self, report_data, tmp_path):
        """Same data produces same output (ignoring timestamp)."""
        out1 = tmp_path / "r1.json"
        out2 = tmp_path / "r2.json"
        write_json_report(report_data, str(out1))
        write_json_report(report_data, str(out2))
        d1 = json.loads(out1.read_text())
        d2 = json.loads(out2.read_text())
        # Timestamps will differ; compare everything else
        d1["run_info"].pop("timestamp", None)
        d2["run_info"].pop("timestamp", None)
        assert d1 == d2


class TestWriteMdReport:
    def test_writes_markdown(self, report_data, tmp_path):
        out = tmp_path / "report.md"
        write_md_report(report_data, str(out))
        assert out.exists()
        content = out.read_text()
        assert "# Multi-Paper Comparison Report" in content
        assert "accuracy" in content

    def test_contains_leaderboards(self, report_data, tmp_path):
        out = tmp_path / "report.md"
        write_md_report(report_data, str(out))
        content = out.read_text()
        assert "Leaderboard" in content
        assert "| Rank |" in content

    def test_contains_synopses(self, report_data, tmp_path):
        out = tmp_path / "report.md"
        write_md_report(report_data, str(out))
        content = out.read_text()
        assert "Synopses" in content
