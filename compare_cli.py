#!/usr/bin/env python3
"""
CLI entry point for multi-paper comparison.

Usage:
    python compare_cli.py \\
        --input-dir ./fixtures \\
        --patterns "*.json" \\
        --metrics "accuracy,f1,latency_ms" \\
        --group-by "task,model" \\
        --filters "year>=2022" \\
        --output-md ./report.md \\
        --output-json ./report.json \\
        --parallel 8 \\
        --strict-schema \\
        --summary
"""

import argparse
import logging
import sys

from compare.api import compare_papers


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="research_rag compare",
        description="Compare multiple research paper JSON artifacts side-by-side.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing paper JSON files.",
    )
    parser.add_argument(
        "--patterns",
        default="*.json",
        help='Glob patterns (comma-separated). Default: "*.json"',
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated metric names to include. Default: auto-discover.",
    )
    parser.add_argument(
        "--group-by",
        default=None,
        help="Comma-separated fields to group papers by.",
    )
    parser.add_argument(
        "--filters",
        default=None,
        help='Filter expressions (comma-separated). E.g. "year>=2022,task=qa"',
    )
    parser.add_argument(
        "--sort",
        default=None,
        help='Sort keys (comma-separated). Prefix with "-" for descending. '
             'E.g. "-accuracy,latency_ms"',
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Path for Markdown report output.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Path for JSON report output.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Number of parallel workers for file loading. Default: 4",
    )
    parser.add_argument(
        "--strict-schema",
        action="store_true",
        default=False,
        help="Reject papers that fail full schema validation.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        default=False,
        help="Include narrative summaries in the report.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Parse comma-separated arguments
    patterns = [p.strip() for p in args.patterns.split(",")]
    metrics = (
        [m.strip() for m in args.metrics.split(",")]
        if args.metrics
        else None
    )
    group_by = (
        [g.strip() for g in args.group_by.split(",")]
        if args.group_by
        else None
    )
    sort_keys = (
        [s.strip() for s in args.sort.split(",")]
        if args.sort
        else None
    )

    try:
        result = compare_papers(
            input_dir=args.input_dir,
            patterns=patterns,
            metrics=metrics,
            group_by=group_by,
            filters=args.filters,
            parallel=args.parallel,
            strict_schema=args.strict_schema,
            return_md=True,
            output_json=args.output_json,
            output_md=args.output_md,
            sort_keys=sort_keys,
            summary=args.summary,
        )
    except FileNotFoundError as e:
        logging.error("Input directory not found: %s", e)
        return 1
    except Exception as e:
        logging.error("Unexpected error: %s", e, exc_info=True)
        return 2

    # Print summary to stdout
    ri = result.get("run_info", {})
    print(f"\n{'='*60}")
    print(f"  Research RAG — Multi-Paper Comparison Complete")
    print(f"{'='*60}")
    print(f"  Files discovered:  {ri.get('file_count', '?')}")
    print(f"  Papers analyzed:   {ri.get('paper_count', '?')}")
    print(f"  Errors:            {len(ri.get('errors', []))}")
    print(f"  Metrics compared:  {', '.join(result.get('schema', {}).get('canonical_metrics', []))}")

    leaders = result.get("comparisons", {}).get("leaders", [])
    if leaders:
        print(f"\n  Metric Leaders:")
        for ld in leaders:
            print(f"    {ld['metric']:20s} → {ld['paper_id']} ({ld['value']})")

    if args.output_json:
        print(f"\n  JSON report: {args.output_json}")
    if args.output_md:
        print(f"  MD report:   {args.output_md}")
    print(f"{'='*60}\n")

    # Exit with non-zero code if critical errors
    errs = ri.get("errors", [])
    has_critical = any(
        "paper_id" not in (e if isinstance(e, dict) else {})
        for e in errs
    )
    return 1 if has_critical and args.strict_schema else 0


if __name__ == "__main__":
    sys.exit(main())
