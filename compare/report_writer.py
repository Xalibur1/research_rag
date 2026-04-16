"""
Report writer: generates machine-readable JSON and human-readable Markdown
comparison reports from processed paper data.
"""

import json
from datetime import datetime, timezone
from typing import Any

from compare.comparison_engine import (
    build_metric_table,
    collect_all_metrics,
    compute_pairwise_deltas,
    compute_rankings,
    compute_summary_stats,
    find_leaders,
    group_papers,
)
from compare.narrative import (
    generate_executive_summary,
    generate_global_findings,
    generate_group_narrative,
    generate_paper_synopsis,
    generate_tradeoff_analysis,
)

__version__ = "1.0.0"


# ---------------------------------------------------------------------------
# Build the full report data structure
# ---------------------------------------------------------------------------

def build_report_data(
    papers: list[dict[str, Any]],
    errors: list[dict],
    metrics: list[str] | None = None,
    group_by: list[str] | None = None,
    input_dir: str = "",
    file_count: int = 0,
    include_summary: bool = True,
) -> dict[str, Any]:
    """Build the complete report data dict.

    This is the single source of truth consumed by both
    ``write_json_report`` and ``write_md_report``.
    """
    if metrics is None:
        metrics = collect_all_metrics(papers)

    table = build_metric_table(papers, metrics)

    # Per-metric tables & stats
    per_metric_tables = []
    all_stats: dict[str, dict] = {}
    for m in metrics:
        rankings = compute_rankings(table, m)
        stats = compute_summary_stats(table, m)
        per_metric_tables.append({
            "metric": m,
            "rows": rankings,
            "stats": stats,
        })
        all_stats[m] = stats

    # Pairwise deltas
    pairwise = []
    for m in metrics:
        pairwise.extend(compute_pairwise_deltas(table, m))

    # Leaders
    leaders = find_leaders(table, metrics)

    # Groups & narratives
    groups = group_papers(papers, group_by)
    per_group_narratives = []
    for gk, gpapers in groups.items():
        gtable = build_metric_table(gpapers, metrics)
        gstats = {m: compute_summary_stats(gtable, m) for m in metrics}
        per_group_narratives.append({
            "group_key": gk,
            "paper_ids": [p["paper_id"] for p in gpapers],
            "stats": gstats,
            "text_summary": generate_group_narrative(gk, gpapers, gstats),
        })

    # Anomalies: papers with all-null metrics
    anomalies: list[dict] = []
    for p in papers:
        m_vals = p.get("metrics") or {}
        non_null = [v for v in m_vals.values() if v is not None]
        if not non_null and m_vals:
            anomalies.append({
                "paper_id": p["paper_id"],
                "issue": "all_metrics_null",
                "detail": "Every metric value is null; paper may have missing data.",
            })

    # Global findings
    global_findings = generate_global_findings(leaders, anomalies) if include_summary else []

    # Paper synopses
    synopses = {p["paper_id"]: generate_paper_synopsis(p) for p in papers}

    now = datetime.now(timezone.utc).isoformat()

    report: dict[str, Any] = {
        "run_info": {
            "timestamp": now,
            "version": __version__,
            "input_dir": input_dir,
            "file_count": file_count,
            "paper_count": len(papers),
            "errors": errors,
        },
        "schema": {
            "canonical_metrics": metrics,
        },
        "papers": [
            {
                "paper_id": p["paper_id"],
                "title": p.get("title"),
                "model": p.get("model"),
                "task": p.get("task"),
                "dataset": p.get("dataset"),
                "metrics": p.get("metrics", {}),
                "inputs": p.get("inputs", {}),
                "outputs": p.get("outputs", {}),
            }
            for p in papers
        ],
        "comparisons": {
            "per_metric_tables": per_metric_tables,
            "pairwise_deltas": pairwise,
            "leaders": leaders,
        },
        "summaries": {
            "per_group_narratives": per_group_narratives,
            "global_findings": global_findings,
            "synopses": synopses,
        },
        "anomalies": anomalies,
    }

    return report


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def write_json_report(data: dict[str, Any], path: str) -> None:
    """Write the report dict to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def write_md_report(data: dict[str, Any], path: str) -> None:
    """Write a human-readable Markdown report."""
    lines: list[str] = []

    # Title
    lines.append("# Multi-Paper Comparison Report")
    lines.append("")
    ri = data.get("run_info", {})
    lines.append(f"_Generated: {ri.get('timestamp', 'N/A')} | "
                 f"Version: {ri.get('version', '?')} | "
                 f"Papers: {ri.get('paper_count', '?')}_")
    lines.append("")

    # Executive summary
    leaders = data.get("comparisons", {}).get("leaders", [])
    gf = data.get("summaries", {}).get("global_findings", [])
    lines.append(generate_executive_summary(
        file_count=ri.get("file_count", 0),
        paper_count=ri.get("paper_count", 0),
        error_count=len(ri.get("errors", [])),
        leaders=leaders,
        global_findings=gf,
    ))
    lines.append("")

    # Leaderboard tables
    lines.append("---")
    lines.append("")
    lines.append("## Metric Leaderboards")
    lines.append("")
    for mt in data.get("comparisons", {}).get("per_metric_tables", []):
        metric = mt["metric"]
        rows = mt["rows"]
        stats = mt.get("stats", {})
        if not rows:
            continue

        lines.append(f"### {metric}")
        lines.append("")
        lines.append("| Rank | Paper | Value | Δ to Best |")
        lines.append("|------|-------|-------|-----------|")
        for r in rows[:10]:  # Top 10
            lines.append(
                f"| {r['rank']} | {r['paper_id']} | {r['value']} | {r['delta_to_best']} |"
            )
        if stats.get("count", 0) > 0:
            lines.append(
                f"\n_Stats — mean: {stats['mean']}, median: {stats['median']}, "
                f"stdev: {stats['stdev']}_"
            )
        lines.append("")

    # Trade-offs
    lines.append("---")
    lines.append("")
    # Build a quick table for trade-off analysis
    papers_list = data.get("papers", [])
    metrics_list = data.get("schema", {}).get("canonical_metrics", [])
    if papers_list:
        table = build_metric_table(
            [{"paper_id": p["paper_id"], "metrics": p.get("metrics", {})} for p in papers_list],
            metrics_list,
        )
        lines.append(generate_tradeoff_analysis(table))
        lines.append("")

    # Group narratives
    group_narrs = data.get("summaries", {}).get("per_group_narratives", [])
    if group_narrs and not (len(group_narrs) == 1 and group_narrs[0]["group_key"] == "all"):
        lines.append("---")
        lines.append("")
        lines.append("## Group Analysis")
        lines.append("")
        for gn in group_narrs:
            lines.append(gn["text_summary"])
            lines.append("")

    # Paper synopses
    synopses = data.get("summaries", {}).get("synopses", {})
    if synopses:
        lines.append("---")
        lines.append("")
        lines.append("## Paper Synopses")
        lines.append("")
        for pid, syn in sorted(synopses.items()):
            lines.append(f"- {syn}")
        lines.append("")

    # Anomalies
    anomalies = data.get("anomalies", [])
    if anomalies:
        lines.append("---")
        lines.append("")
        lines.append("## ⚠ Anomalies")
        lines.append("")
        for a in anomalies:
            lines.append(f"- **{a['paper_id']}**: {a['issue']} — {a.get('detail', '')}")
        lines.append("")

    # Errors
    errs = ri.get("errors", [])
    if errs:
        lines.append("---")
        lines.append("")
        lines.append("## Ingestion Errors")
        lines.append("")
        for e in errs:
            lines.append(f"- `{e.get('source', e.get('paper_id', '?'))}`: {e.get('error', e.get('errors', '?'))}")
        lines.append("")

    content = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
