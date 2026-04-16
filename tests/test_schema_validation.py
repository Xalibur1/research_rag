"""Tests for compare.schema_validator."""

import json
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compare.schema_validator import validate_paper, validate_papers_batch

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "paper_schema.json"


@pytest.fixture
def schema():
    with open(SCHEMA_PATH) as f:
        return json.load(f)


@pytest.fixture
def valid_paper():
    return {
        "paper_id": "test-001",
        "title": "Test Paper",
        "year": 2025,
        "metrics": {"accuracy": 0.9},
    }


class TestValidatePaper:
    def test_valid_paper_passes(self, schema, valid_paper):
        errors = validate_paper(valid_paper, schema=schema, strict=True)
        assert errors == []

    def test_missing_paper_id_fails_strict(self, schema):
        errors = validate_paper({"title": "No ID"}, schema=schema, strict=True)
        assert any("paper_id" in e for e in errors)

    def test_missing_paper_id_fails_lenient(self, schema):
        errors = validate_paper({"title": "No ID"}, schema=schema, strict=False)
        assert len(errors) == 1
        assert "paper_id" in errors[0]

    def test_extra_fields_allowed(self, schema, valid_paper):
        valid_paper["custom_field"] = "extra"
        errors = validate_paper(valid_paper, schema=schema, strict=True)
        assert errors == []

    def test_wrong_year_type_fails_strict(self, schema, valid_paper):
        valid_paper["year"] = "not-a-number"
        errors = validate_paper(valid_paper, schema=schema, strict=True)
        assert len(errors) > 0

    def test_metrics_with_null_value_passes(self, schema, valid_paper):
        valid_paper["metrics"] = {"accuracy": 0.9, "f1": None}
        errors = validate_paper(valid_paper, schema=schema, strict=True)
        assert errors == []

    def test_metrics_with_string_value_fails(self, schema, valid_paper):
        valid_paper["metrics"] = {"accuracy": "high"}
        errors = validate_paper(valid_paper, schema=schema, strict=True)
        assert len(errors) > 0


class TestValidatePapersBatch:
    def test_batch_returns_valid_and_errors(self, schema):
        papers = [
            {"paper_id": "a", "title": "A"},
            {"title": "No ID"},  # invalid
            {"paper_id": "b", "title": "B"},
        ]
        valid, errs = validate_papers_batch(papers, schema=schema, strict=True)
        assert len(valid) == 2
        assert len(errs) == 1

    def test_batch_lenient_keeps_papers_with_id(self, schema):
        papers = [
            {"paper_id": "a", "year": "bad"},  # year wrong type, but lenient
            {"title": "No ID"},  # truly invalid even in lenient
        ]
        valid, errs = validate_papers_batch(papers, schema=schema, strict=False)
        assert len(valid) == 1
        assert valid[0]["paper_id"] == "a"
