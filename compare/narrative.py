"""
Template-driven narrative generation for multi-paper comparison reports.

No LLM calls — purely deterministic, template-based.
"""

from typing import Any


def generate_paper_synopsis(paper: dict[str, Any]) -> str:
    """Generate a 2–3 line method summary for a single paper."""
    pid = paper.get("paper_id", "unknown")
    title = paper.get("title", "Untitled")
    model = paper.get("model", "unspecified model")
    task = paper.get("task", "unspecified task")
    datasets = paper.get("dataset", [])
    if isinstance(datasets, str):
        datasets = [datasets]
    ds_str = ", ".join(datasets) if datasets else "unspecified datasets"

    metrics = paper.get("metrics") or {}
    metric_parts = []
    for k, v in sorted(metrics.items()):
        if v is not None:
            metric_parts.append(f"{k}={v}")
    metrics_str = ", ".join(metric_parts[:5]) if metric_parts else "no reported metrics"

    lines = [
        f"**{title}** (`{pid}`): Uses *{model}* for *{task}* on {ds_str}.",
        f"Key metrics: {metrics_str}.",
    ]

    limitations = []
    method = paper.get("method") if isinstance(paper.get("method"), dict) else None
    if method:
        limitations = method.get("limitations", [])
    if limitations:
        lines.append(f"Noted limitations: {limitations[0]}")

    return " ".join(lines)


def generate_group_narrative(
    group_key: str,
    papers: list[dict],
    stats: dict[str, dict] | None = None,
) -> str:
    """Generate a comparative narrative for a group of papers."""
    n = len(papers)
    models = sorted({p.get("model", "?") for p in papers})

    lines = [f"### Group: {group_key}", ""]
    lines.append(f"This group contains **{n} paper(s)** covering model(s): {', '.join(models)}.")

    if stats:
        lines.append("")
        lines.append("| Metric | Mean | Median | Best | Worst |")
        lines.append("|--------|------|--------|------|-------|")
        for metric, s in sorted(stats.items()):
            if s.get("count", 0) == 0:
                continue
            lines.append(
                f"| {metric} | {s['mean']} | {s['median']} | {s['min']} | {s['max']} |"
            )

    # Identify standout papers
    if stats:
        for metric, s in sorted(stats.items()):
            if s.get("count", 0) < 2:
                continue
            best_val = s["min"] if metric in {"latency_ms", "cost_usd", "tokens_in", "tokens_out"} else s["max"]
            for p in papers:
                pval = (p.get("metrics") or {}).get(metric)
                if pval == best_val:
                    lines.append(
                        f"- **{p.get('model', p['paper_id'])}** leads in *{metric}* ({best_val})."
                    )
                    break

    return "\n".join(lines)


def generate_global_findings(
    leaders: list[dict],
    anomalies: list[dict] | None = None,
) -> list[str]:
    """Generate bullet-point global findings."""
    findings: list[str] = []

    if leaders:
        top_models = {}
        for ld in leaders:
            mid = ld["paper_id"]
            top_models.setdefault(mid, []).append(ld["metric"])
        for mid, metrics in sorted(top_models.items(), key=lambda x: -len(x[1])):
            findings.append(
                f"**{mid}** is the leader in {len(metrics)} metric(s): {', '.join(metrics)}."
            )

    if anomalies:
        for a in anomalies:
            findings.append(
                f"⚠ Anomaly in *{a['paper_id']}*: {a['issue']} — {a.get('detail', '')}"
            )

    if not findings:
        findings.append("No standout findings to report.")

    return findings


def generate_executive_summary(
    file_count: int,
    paper_count: int,
    error_count: int,
    leaders: list[dict],
    global_findings: list[str],
) -> str:
    """Generate a top-level executive summary for the MD report."""
    lines = ["## Executive Summary", ""]
    lines.append(f"- **{file_count}** file(s) discovered, **{paper_count}** valid paper(s) analyzed.")
    if error_count:
        lines.append(f"- **{error_count}** error(s) encountered during ingestion/validation.")

    if leaders:
        lines.append("- **Metric leaders:**")
        for ld in leaders:
            lines.append(f"  - *{ld['metric']}*: **{ld['paper_id']}** ({ld['value']})")

    if global_findings:
        lines.append("")
        lines.append("### Key Findings")
        for f in global_findings:
            lines.append(f"- {f}")

    return "\n".join(lines)


def generate_tradeoff_analysis(
    table: dict[str, dict[str, Any]],
    metric_pairs: list[tuple[str, str]] | None = None,
) -> str:
    """Analyse trade-offs between metric pairs (e.g., accuracy vs latency)."""
    if metric_pairs is None:
        # Default interesting pairs
        metric_pairs = [
            ("accuracy", "latency_ms"),
            ("accuracy", "cost_usd"),
            ("f1", "latency_ms"),
        ]

    lines = ["## Notable Trade-offs", ""]

    any_found = False
    for m_good, m_bad in metric_pairs:
        entries = []
        for pid, row in sorted(table.items()):
            vg = row.get(m_good)
            vb = row.get(m_bad)
            if vg is not None and vb is not None:
                entries.append((pid, vg, vb))
        if len(entries) < 2:
            continue

        any_found = True
        lines.append(f"### {m_good} vs {m_bad}")
        lines.append("")
        lines.append(f"| Paper | {m_good} | {m_bad} |")
        lines.append("|-------|" + "-" * (len(m_good) + 2) + "|" + "-" * (len(m_bad) + 2) + "|")
        # Sort by m_good descending
        entries.sort(key=lambda x: x[1], reverse=True)
        for pid, vg, vb in entries:
            lines.append(f"| {pid} | {vg} | {vb} |")

        # Brief commentary
        best_good = entries[0]
        best_bad = min(entries, key=lambda x: x[2])
        if best_good[0] != best_bad[0]:
            lines.append(
                f"\n> **{best_good[0]}** achieves the best *{m_good}* ({best_good[1]}) "
                f"but **{best_bad[0]}** has the lowest *{m_bad}* ({best_bad[2]}), "
                f"suggesting a trade-off."
            )
        lines.append("")

    if not any_found:
        lines.append("_No metric pairs with sufficient data for trade-off analysis._")

    return "\n".join(lines)
