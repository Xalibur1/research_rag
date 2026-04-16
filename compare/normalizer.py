"""
Metric name normalization for paper artifacts.

Remaps legacy / shorthand metric names to canonical names using the
normalization_map.json mapping table.
"""

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAP_PATH = Path(__file__).resolve().parent.parent / "schemas" / "normalization_map.json"
_CACHED_MAP: dict | None = None


def _load_mapping(map_path: Path | None = None) -> dict[str, str]:
    """Load and cache the normalization mapping."""
    global _CACHED_MAP
    path = map_path or _MAP_PATH
    if _CACHED_MAP is None or map_path is not None:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Strip comment keys
        mapping = {k: v for k, v in raw.items() if not k.startswith("_")}
        if map_path is None:
            _CACHED_MAP = mapping
        return mapping
    return _CACHED_MAP


def normalize_metrics(
    metrics: dict[str, Any],
    mapping: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Normalize metric keys in a metrics dict.

    Args:
        metrics: Raw metrics dict (key -> numeric value).
        mapping: Legacy-to-canonical mapping. Loaded from disk if None.

    Returns:
        (normalized_metrics, warnings) where warnings list unmapped
        or colliding names.
    """
    if mapping is None:
        mapping = _load_mapping()

    normalized: dict[str, Any] = {}
    warnings: list[str] = []

    for key, value in metrics.items():
        canonical = mapping.get(key.lower(), key.lower())
        if key.lower() in mapping:
            logger.info("Normalized metric '%s' -> '%s'", key, canonical)
        if canonical in normalized:
            warnings.append(
                f"Collision: '{key}' maps to '{canonical}' which already exists; "
                f"keeping first value ({normalized[canonical]}), discarding {value}"
            )
        else:
            normalized[canonical] = value

    return normalized, warnings


def normalize_paper(
    paper: dict[str, Any],
    mapping: dict[str, str] | None = None,
    canonical_metrics: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Normalize a full paper dict.

    - Remaps metric names via the mapping table.
    - Fills missing canonical metrics with None and flags them.
    - Normalizes ``year`` to int if possible.

    Args:
        paper: Raw paper dict.
        mapping: Legacy-to-canonical mapping.
        canonical_metrics: If provided, ensures each metric key exists
                           (filled with None if absent).

    Returns:
        (normalized_paper, warnings)
    """
    if mapping is None:
        mapping = _load_mapping()

    paper = deepcopy(paper)
    all_warnings: list[str] = []

    # --- Normalize metrics ---
    raw_metrics = paper.get("metrics") or {}
    normed, metric_warnings = normalize_metrics(raw_metrics, mapping)
    all_warnings.extend(metric_warnings)

    # Fill missing canonical metrics with None
    if canonical_metrics:
        for cm in canonical_metrics:
            if cm not in normed:
                normed[cm] = None
                all_warnings.append(
                    f"Paper '{paper.get('paper_id', '?')}': metric '{cm}' missing, filled with null"
                )

    paper["metrics"] = normed

    # --- Normalize year to int ---
    if "year" in paper and paper["year"] is not None:
        try:
            paper["year"] = int(paper["year"])
        except (ValueError, TypeError):
            all_warnings.append(
                f"Paper '{paper.get('paper_id', '?')}': could not convert year "
                f"'{paper['year']}' to int"
            )

    # --- Normalize dataset to list ---
    ds = paper.get("dataset")
    if isinstance(ds, str):
        paper["dataset"] = [ds]

    return paper, all_warnings


def normalize_papers_batch(
    papers: list[dict[str, Any]],
    mapping: dict[str, str] | None = None,
    canonical_metrics: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Normalize a batch of papers. Returns (normalized_papers, all_warnings)."""
    if mapping is None:
        mapping = _load_mapping()

    all_normalized: list[dict] = []
    all_warnings: list[str] = []

    for paper in papers:
        normed, warns = normalize_paper(paper, mapping, canonical_metrics)
        all_normalized.append(normed)
        all_warnings.extend(warns)

    return all_normalized, all_warnings
