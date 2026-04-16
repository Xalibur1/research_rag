"""Tests for compare.ingestion."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compare.ingestion import (
    _load_single_file,
    deduplicate,
    discover_files,
    ingest,
    load_files,
)


@pytest.fixture
def tmp_dir(tmp_path):
    """Create a temp directory with some paper JSON files."""
    papers = [
        {"paper_id": "p1", "title": "Paper One", "metrics": {"accuracy": 0.9}},
        {"paper_id": "p2", "title": "Paper Two", "metrics": {"accuracy": 0.8}},
    ]
    for i, p in enumerate(papers):
        fp = tmp_path / f"paper_{i+1}.json"
        fp.write_text(json.dumps(p))

    # Write a corrupt file
    (tmp_path / "corrupt.json").write_text("{bad json")

    # Write a non-json file (should be excluded by pattern)
    (tmp_path / "readme.txt").write_text("not json")

    return tmp_path


class TestDiscoverFiles:
    def test_finds_json_files(self, tmp_dir):
        files = discover_files(str(tmp_dir), ["*.json"])
        assert len(files) == 3  # paper_1, paper_2, corrupt

    def test_glob_pattern(self, tmp_dir):
        files = discover_files(str(tmp_dir), ["paper_*.json"])
        assert len(files) == 2

    def test_nonexistent_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            discover_files("/nonexistent/path")

    def test_no_matching_files(self, tmp_dir):
        files = discover_files(str(tmp_dir), ["*.csv"])
        assert files == []


class TestLoadSingleFile:
    def test_valid_file(self, tmp_dir):
        result = _load_single_file(tmp_dir / "paper_1.json")
        assert result is not None
        assert result["paper_id"] == "p1"
        assert "provenance" in result

    def test_corrupt_file_returns_none(self, tmp_dir):
        result = _load_single_file(tmp_dir / "corrupt.json")
        assert result is None


class TestLoadFiles:
    def test_loads_valid_files(self, tmp_dir):
        paths = discover_files(str(tmp_dir), ["paper_*.json"])
        loaded, errors = load_files(paths, parallel=1)
        assert len(loaded) == 2
        assert len(errors) == 0

    def test_handles_corrupt_files(self, tmp_dir):
        paths = discover_files(str(tmp_dir), ["*.json"])
        loaded, errors = load_files(paths, parallel=1)
        assert len(loaded) == 2
        assert len(errors) == 1


class TestDeduplicate:
    def test_no_duplicates(self):
        papers = [{"paper_id": "a"}, {"paper_id": "b"}]
        result = deduplicate(papers)
        assert len(result) == 2

    def test_keeps_highest_version(self):
        papers = [
            {"paper_id": "x", "version": "1.0"},
            {"paper_id": "x", "version": "2.0"},
        ]
        result = deduplicate(papers)
        assert len(result) == 1
        assert result[0]["version"] == "2.0"

    def test_keeps_latest_timestamp_as_tiebreaker(self):
        papers = [
            {"paper_id": "x", "version": "1.0",
             "provenance": {"created_at": "2024-01-01T00:00:00Z"}},
            {"paper_id": "x", "version": "1.0",
             "provenance": {"created_at": "2025-06-01T00:00:00Z"}},
        ]
        result = deduplicate(papers)
        assert len(result) == 1
        assert result[0]["provenance"]["created_at"] == "2025-06-01T00:00:00Z"


class TestIngest:
    def test_full_pipeline(self, tmp_dir):
        papers, errors = ingest(str(tmp_dir), patterns=["paper_*.json"], strict=False, parallel=1)
        assert len(papers) == 2
        assert len(errors) == 0

    def test_with_corrupt_file(self, tmp_dir):
        papers, errors = ingest(str(tmp_dir), patterns=["*.json"], strict=False, parallel=1)
        assert len(papers) == 2
        assert len(errors) >= 1  # corrupt file error
