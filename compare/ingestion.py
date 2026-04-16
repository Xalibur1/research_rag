"""
Multi-file ingestion: discovery, loading, validation, and deduplication.

Designed for filesystem-based JSON staging — no external database.
"""

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from compare.schema_validator import validate_paper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_files(
    input_dir: str,
    patterns: list[str] | None = None,
) -> list[Path]:
    """Discover JSON files in *input_dir* matching glob *patterns*.

    Args:
        input_dir: Root directory to search (non-recursive by default;
                   use ``**/*.json`` for recursive).
        patterns: Shell glob patterns, e.g. ``["*.json"]``.

    Returns:
        Sorted list of unique Path objects.
    """
    if patterns is None:
        patterns = ["*.json"]

    base = Path(input_dir).resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {base}")

    found: set[Path] = set()
    for pat in patterns:
        found.update(base.glob(pat))

    return sorted(found)


# ---------------------------------------------------------------------------
# Single-file loader (used by parallel workers)
# ---------------------------------------------------------------------------

def _load_single_file(path: Path) -> dict[str, Any] | None:
    """Load and parse a single JSON file.

    Returns the parsed dict with ``provenance.source_path`` injected,
    or None if the file is unreadable / unparseable.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning("Skipping %s: root is not a JSON object", path)
            return None
        # Inject provenance
        prov = data.setdefault("provenance", {})
        if "source_path" not in prov:
            prov["source_path"] = str(path)
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Skipping unreadable file %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Batch loading (parallel)
# ---------------------------------------------------------------------------

def load_files(
    paths: list[Path],
    parallel: int = 4,
) -> tuple[list[dict[str, Any]], list[dict]]:
    """Load JSON files, optionally in parallel.

    Returns:
        (loaded_papers, load_errors) where each load_error is
        ``{\"source\": str, \"error\": str}``.
    """
    loaded: list[dict] = []
    errors: list[dict] = []

    if parallel <= 1 or len(paths) <= 1:
        for p in paths:
            result = _load_single_file(p)
            if result is None:
                errors.append({"source": str(p), "error": "Failed to load/parse"})
            else:
                loaded.append(result)
    else:
        with ProcessPoolExecutor(max_workers=parallel) as pool:
            future_to_path = {pool.submit(_load_single_file, p): p for p in paths}
            for future in as_completed(future_to_path):
                p = future_to_path[future]
                try:
                    result = future.result()
                    if result is None:
                        errors.append({"source": str(p), "error": "Failed to load/parse"})
                    else:
                        loaded.append(result)
                except Exception as exc:
                    logger.warning("Worker error for %s: %s", p, exc)
                    errors.append({"source": str(p), "error": str(exc)})

    # Stable ordering: sort by source_path for determinism
    loaded.sort(key=lambda p: (p.get("provenance") or {}).get("source_path", ""))
    return loaded, errors


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _parse_version(v: Any) -> tuple:
    """Convert version to a comparable tuple."""
    if v is None:
        return (0,)
    if isinstance(v, (int, float)):
        return (v,)
    # Try dotted version string: "1.2.3" -> (1, 2, 3)
    try:
        return tuple(int(x) for x in str(v).split("."))
    except ValueError:
        return (str(v),)


def _parse_timestamp(ts: Any) -> datetime:
    """Parse ISO-8601 timestamp, returning epoch for unparseable values."""
    if ts is None:
        return datetime.min
    try:
        # Handle common ISO-8601 formats
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return datetime.min


def deduplicate(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate papers by ``paper_id``.

    When duplicates exist, keep the one with the highest ``version``;
    ties are broken by latest ``provenance.created_at``.

    Returns:
        Deduplicated list in stable order (sorted by paper_id).
    """
    buckets: dict[str, list[dict]] = {}
    for paper in papers:
        pid = paper.get("paper_id", "")
        buckets.setdefault(pid, []).append(paper)

    result: list[dict] = []
    for pid in sorted(buckets.keys()):
        candidates = buckets[pid]
        if len(candidates) == 1:
            result.append(candidates[0])
        else:
            # Sort: highest version first, then latest timestamp
            candidates.sort(
                key=lambda p: (
                    _parse_version(p.get("version")),
                    _parse_timestamp((p.get("provenance") or {}).get("created_at")),
                ),
                reverse=True,
            )
            winner = candidates[0]
            logger.info(
                "Deduplicated paper_id='%s': kept version=%s, discarded %d duplicate(s)",
                pid,
                winner.get("version"),
                len(candidates) - 1,
            )
            result.append(winner)

    return result


# ---------------------------------------------------------------------------
# Full ingestion pipeline
# ---------------------------------------------------------------------------

def ingest(
    input_dir: str,
    patterns: list[str] | None = None,
    schema: dict | None = None,
    strict: bool = True,
    parallel: int = 4,
) -> tuple[list[dict[str, Any]], list[dict]]:
    """Full ingestion pipeline: discover → load → validate → deduplicate.

    Returns:
        (papers, all_errors) — papers are validated and deduplicated.
    """
    all_errors: list[dict] = []

    # 1. Discover
    paths = discover_files(input_dir, patterns)
    if not paths:
        logger.warning("No files found in %s with patterns %s", input_dir, patterns)
        return [], [{"source": input_dir, "error": "No matching files found"}]

    # 2. Load
    loaded, load_errors = load_files(paths, parallel=parallel)
    all_errors.extend(load_errors)

    # 3. Validate
    from compare.schema_validator import validate_papers_batch
    valid, val_errors = validate_papers_batch(loaded, schema=schema, strict=strict)
    all_errors.extend(val_errors)

    # 4. Deduplicate
    deduped = deduplicate(valid)

    return deduped, all_errors
