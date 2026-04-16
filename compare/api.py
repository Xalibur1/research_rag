"""
Python API entry point for multi-paper comparison.

Usage:
    from compare import compare_papers

    result = compare_papers(
        input_dir="./artifacts",
        metrics=["accuracy", "f1", "latency_ms"],
        group_by=["task", "model"],
        filters="year>=2022",
    )
"""

import logging
from typing import Any

from compare.comparison_engine import (
    collect_all_metrics,
    filter_papers,
    sort_papers,
)
from compare.ingestion import discover_files, ingest
from compare.normalizer import normalize_papers_batch
from compare.report_writer import build_report_data, write_json_report, write_md_report

logger = logging.getLogger(__name__)


def compare_papers(
    input_dir: str,
    patterns: list[str] | None = None,
    metrics: list[str] | None = None,
    group_by: list[str] | None = None,
    filters: dict | str | None = None,
    parallel: int = 4,
    strict_schema: bool = True,
    return_md: bool = True,
    output_json: str | None = None,
    output_md: str | None = None,
    sort_keys: list[str] | None = None,
    summary: bool = True,
) -> dict[str, Any]:
    """Compare multiple paper JSON artifacts.

    End-to-end pipeline:
        discover → validate → normalize → deduplicate →
        filter → sort → build table → compute stats →
        generate narratives → write reports → return result.

    Args:
        input_dir: Directory containing paper JSON files.
        patterns: Glob patterns for file discovery (default: ``["*.json"]``).
        metrics: Metric names to include. None = auto-discover all.
        group_by: Fields to group papers by (e.g. ``["task", "model"]``).
        filters: Filter spec (str, dict, or list). E.g. ``"year>=2022"``.
        parallel: Max parallel workers for file loading.
        strict_schema: If True, reject schema-invalid papers; if False,
                       only require ``paper_id``.
        return_md: If True, include rendered Markdown in result dict.
        output_json: Path to write JSON report. None = skip.
        output_md: Path to write Markdown report. None = skip.
        sort_keys: Sort keys (e.g. ``["-accuracy", "latency_ms"]``).
        summary: Include narrative summaries in report.

    Returns:
        The full report data dict (same structure written to JSON).
    """
    if patterns is None:
        patterns = ["*.json"]

    # 1. Count discovered files for reporting
    file_count = len(discover_files(input_dir, patterns))

    # 2. Ingest: discover → load → validate → deduplicate
    papers, errors = ingest(
        input_dir,
        patterns=patterns,
        strict=strict_schema,
        parallel=parallel,
    )

    if not papers:
        logger.error("No valid papers after ingestion.")
        report = build_report_data(
            papers=[],
            errors=errors,
            metrics=metrics or [],
            input_dir=input_dir,
            file_count=file_count,
            include_summary=summary,
        )
        if output_json:
            write_json_report(report, output_json)
        if output_md:
            write_md_report(report, output_md)
        return report

    # 3. Normalize
    canonical_metrics = metrics  # May be None (auto-discover after normalization)
    papers, norm_warnings = normalize_papers_batch(
        papers,
        canonical_metrics=canonical_metrics,
    )
    if norm_warnings:
        for w in norm_warnings:
            logger.warning("Normalization: %s", w)

    # 4. Filter
    papers = filter_papers(papers, filters)

    # 5. Sort
    papers = sort_papers(papers, sort_keys)

    # 6. Auto-discover metrics if not specified
    if metrics is None:
        metrics = collect_all_metrics(papers)

    # 7. Build report
    report = build_report_data(
        papers=papers,
        errors=errors,
        metrics=metrics,
        group_by=group_by,
        input_dir=input_dir,
        file_count=file_count,
        include_summary=summary,
    )

    # 8. Write outputs
    if output_json:
        write_json_report(report, output_json)
        logger.info("JSON report written to %s", output_json)

    if output_md:
        write_md_report(report, output_md)
        logger.info("Markdown report written to %s", output_md)

    # 9. Optionally embed MD string in result
    if return_md:
        md_lines = []
        # Re-generate in-memory (lightweight)
        from io import StringIO
        import tempfile, os
        tmp = os.path.join(tempfile.gettempdir(), "_compare_tmp.md")
        write_md_report(report, tmp)
        with open(tmp, "r") as f:
            report["_rendered_md"] = f.read()
        try:
            os.unlink(tmp)
        except OSError:
            pass

    return report
