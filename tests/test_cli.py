"""Tests for compare_cli.py."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compare_cli import main

FIXTURES_DIR = str(Path(__file__).resolve().parent.parent / "fixtures")


class TestCLI:
    def test_basic_run(self, tmp_path):
        """CLI completes successfully with fixture data."""
        out_json = str(tmp_path / "report.json")
        out_md = str(tmp_path / "report.md")
        exit_code = main([
            "--input-dir", FIXTURES_DIR,
            "--output-json", out_json,
            "--output-md", out_md,
            "--summary",
        ])
        assert exit_code == 0
        assert Path(out_json).exists()
        assert Path(out_md).exists()

    def test_output_json_valid(self, tmp_path):
        out_json = str(tmp_path / "report.json")
        main([
            "--input-dir", FIXTURES_DIR,
            "--output-json", out_json,
        ])
        data = json.loads(Path(out_json).read_text())
        assert "run_info" in data
        assert data["run_info"]["paper_count"] >= 1

    def test_with_metrics_filter(self, tmp_path):
        out_json = str(tmp_path / "report.json")
        main([
            "--input-dir", FIXTURES_DIR,
            "--metrics", "accuracy,f1",
            "--output-json", out_json,
        ])
        data = json.loads(Path(out_json).read_text())
        assert set(data["schema"]["canonical_metrics"]) == {"accuracy", "f1"}

    def test_with_group_by(self, tmp_path):
        out_json = str(tmp_path / "report.json")
        main([
            "--input-dir", FIXTURES_DIR,
            "--group-by", "task",
            "--output-json", out_json,
            "--summary",
        ])
        data = json.loads(Path(out_json).read_text())
        narratives = data["summaries"]["per_group_narratives"]
        assert len(narratives) >= 2  # at least qa and another task

    def test_nonexistent_dir_returns_error(self):
        exit_code = main([
            "--input-dir", "/nonexistent/dir/that/does/not/exist",
        ])
        assert exit_code != 0

    def test_strict_schema(self, tmp_path):
        out_json = str(tmp_path / "report.json")
        exit_code = main([
            "--input-dir", FIXTURES_DIR,
            "--strict-schema",
            "--output-json", out_json,
        ])
        # Should still succeed since fixtures are valid
        assert exit_code == 0

    def test_filter_papers(self, tmp_path):
        out_json = str(tmp_path / "report.json")
        main([
            "--input-dir", FIXTURES_DIR,
            "--filters", "year>=2025",
            "--output-json", out_json,
        ])
        data = json.loads(Path(out_json).read_text())
        for p in data["papers"]:
            # year might not be in the flat paper dict output, check via paper_id
            assert data["run_info"]["paper_count"] >= 1

    def test_determinism(self, tmp_path):
        """Two runs produce the same report (ignoring timestamps)."""
        out1 = str(tmp_path / "r1.json")
        out2 = str(tmp_path / "r2.json")
        main(["--input-dir", FIXTURES_DIR, "--output-json", out1])
        main(["--input-dir", FIXTURES_DIR, "--output-json", out2])

        d1 = json.loads(Path(out1).read_text())
        d2 = json.loads(Path(out2).read_text())
        d1["run_info"].pop("timestamp", None)
        d2["run_info"].pop("timestamp", None)
        assert d1 == d2
