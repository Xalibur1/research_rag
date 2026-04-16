"""
Schema validation for paper JSON artifacts.

Uses jsonschema (draft-07) to validate each paper dict against
the canonical paper_schema.json.
"""

import json
import logging
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema import Draft7Validator, ValidationError

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "paper_schema.json"
_CACHED_SCHEMA: dict | None = None


def _load_schema(schema_path: Path | None = None) -> dict:
    """Load and cache the JSON schema."""
    global _CACHED_SCHEMA
    path = schema_path or _SCHEMA_PATH
    if _CACHED_SCHEMA is None or schema_path is not None:
        with open(path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        if schema_path is None:
            _CACHED_SCHEMA = schema
        return schema
    return _CACHED_SCHEMA


def validate_paper(
    data: dict[str, Any],
    schema: dict | None = None,
    strict: bool = True,
) -> list[str]:
    """Validate a single paper dict against the JSON schema.

    Args:
        data: Paper dictionary to validate.
        schema: Optional pre-loaded schema dict. If None, loads from disk.
        strict: If True, every validation error is reported.
                If False, only the ``paper_id`` requirement is checked.

    Returns:
        List of human-readable error strings. Empty list means valid.
    """
    if schema is None:
        schema = _load_schema()

    errors: list[str] = []

    if strict:
        validator = Draft7Validator(schema)
        for err in sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path)):
            path = ".".join(str(p) for p in err.absolute_path) or "(root)"
            errors.append(f"{path}: {err.message}")
    else:
        # Lenient: only check required fields
        if "paper_id" not in data or not isinstance(data.get("paper_id"), str):
            errors.append("paper_id: 'paper_id' is a required string property")

    return errors


def validate_papers_batch(
    papers: list[dict[str, Any]],
    schema: dict | None = None,
    strict: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Validate a batch of papers.

    Returns:
        (valid_papers, error_records) where each error_record is
        {\"paper_id\": str|None, \"source\": str|None, \"errors\": list[str]}.
    """
    if schema is None:
        schema = _load_schema()

    valid: list[dict] = []
    error_records: list[dict] = []

    for paper in papers:
        pid = paper.get("paper_id", None)
        src = (paper.get("provenance") or {}).get("source_path", None)
        errs = validate_paper(paper, schema=schema, strict=strict)
        if errs:
            logger.warning("Validation errors for paper_id=%s: %s", pid, errs)
            error_records.append({
                "paper_id": pid,
                "source": src,
                "errors": errs,
            })
            if not strict:
                # In lenient mode, still include the paper if paper_id exists
                if pid is not None:
                    valid.append(paper)
        else:
            valid.append(paper)

    return valid, error_records
