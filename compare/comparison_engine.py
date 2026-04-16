"""
Comparison engine: rankings, summary stats, pairwise deltas, filtering,
sorting, and grouping across multiple paper artifacts.

All operations are deterministic (stable sorts, fixed ordering).
"""

import logging
import operator
import re
import statistics
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric table construction
# ---------------------------------------------------------------------------

def collect_all_metrics(papers: list[dict]) -> list[str]:
    """Discover all numeric metric names across papers, sorted."""
    names: set[str] = set()
    for p in papers:
        for k, v in (p.get("metrics") or {}).items():
            if isinstance(v, (int, float)) or v is None:
                names.add(k)
    return sorted(names)


def build_metric_table(
    papers: list[dict],
    metrics: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build an in-memory table: paper_id -> {metric_name: value}.

    Missing values are represented as None.
    """
    if metrics is None:
        metrics = collect_all_metrics(papers)

    table: dict[str, dict[str, Any]] = {}
    for p in papers:
        pid = p["paper_id"]
        m = p.get("metrics") or {}
        table[pid] = {metric: m.get(metric) for metric in metrics}

    return table


# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------

# Metrics where *lower* is better
_LOWER_IS_BETTER = {"latency_ms", "cost_usd", "tokens_in", "tokens_out"}


def compute_rankings(
    table: dict[str, dict[str, Any]],
    metric: str,
    ascending: bool | None = None,
) -> list[dict]:
    """Rank papers by a single metric.

    Args:
        ascending: If None, auto-detect (lower-is-better for latency/cost).

    Returns:
        [{rank, paper_id, value, delta_to_best}] sorted by rank.
    """
    if ascending is None:
        ascending = metric in _LOWER_IS_BETTER

    entries = []
    for pid, metrics_dict in table.items():
        val = metrics_dict.get(metric)
        if val is not None:
            entries.append((pid, val))

    entries.sort(key=lambda x: x[1], reverse=not ascending)

    if not entries:
        return []

    best_val = entries[0][1]
    rankings = []
    for rank, (pid, val) in enumerate(entries, start=1):
        rankings.append({
            "rank": rank,
            "paper_id": pid,
            "value": val,
            "delta_to_best": round(val - best_val, 6),
        })

    return rankings


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary_stats(
    table: dict[str, dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    """Compute mean, median, min, max, stdev for a metric."""
    values = [
        v for v in (row.get(metric) for row in table.values())
        if v is not None
    ]
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None, "stdev": None, "count": 0}

    return {
        "mean": round(statistics.mean(values), 6),
        "median": round(statistics.median(values), 6),
        "min": min(values),
        "max": max(values),
        "stdev": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
        "count": len(values),
    }


# ---------------------------------------------------------------------------
# Pairwise deltas
# ---------------------------------------------------------------------------

def compute_pairwise_deltas(
    table: dict[str, dict[str, Any]],
    metric: str,
) -> list[dict]:
    """Compute pairwise deltas for a metric (all ordered pairs).

    Returns:
        [{metric, a_paper_id, b_paper_id, delta}] where delta = a - b.
        Sorted by (a_paper_id, b_paper_id) for determinism.
    """
    pids = sorted(pid for pid, row in table.items() if row.get(metric) is not None)
    deltas = []
    for a in pids:
        for b in pids:
            if a >= b:
                continue
            va = table[a][metric]
            vb = table[b][metric]
            deltas.append({
                "metric": metric,
                "a_paper_id": a,
                "b_paper_id": b,
                "delta": round(va - vb, 6),
            })
    return deltas


# ---------------------------------------------------------------------------
# Leaders
# ---------------------------------------------------------------------------

def find_leaders(
    table: dict[str, dict[str, Any]],
    metrics: list[str] | None = None,
) -> list[dict]:
    """Find the best paper per metric.

    Returns:
        [{metric, paper_id, value}]
    """
    if metrics is None:
        metrics = sorted({m for row in table.values() for m in row})

    leaders = []
    for metric in metrics:
        rankings = compute_rankings(table, metric)
        if rankings:
            best = rankings[0]
            leaders.append({
                "metric": metric,
                "paper_id": best["paper_id"],
                "value": best["value"],
            })
    return leaders


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

_FILTER_RE = re.compile(
    r"^(?P<field>\w+)\s*(?P<op>>=|<=|!=|>|<|==|=)\s*(?P<value>.+)$"
)
_OPS = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
    "=": operator.eq,
    "!=": operator.ne,
}


def _parse_filter_value(raw: str) -> Any:
    """Try to cast filter value to int / float, else keep as string."""
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw.strip("'\"")


def parse_filters(filters: dict | str | list | None) -> list[tuple[str, Any, Any]]:
    """Parse filter specification into (field, op_func, value) triples.

    Accepts:
        - None -> []
        - str: comma-separated, e.g. "year>=2022,task=qa"
        - list[str]: ["year>=2022", "task=qa"]
        - dict: {"year": ">=2022"} (simple key-value)
    """
    if filters is None:
        return []

    raw_list: list[str] = []
    if isinstance(filters, str):
        raw_list = [f.strip() for f in filters.split(",") if f.strip()]
    elif isinstance(filters, list):
        raw_list = [str(f).strip() for f in filters if str(f).strip()]
    elif isinstance(filters, dict):
        for k, v in filters.items():
            raw_list.append(f"{k}{v}" if any(str(v).startswith(op) for op in _OPS) else f"{k}=={v}")

    parsed: list[tuple[str, Any, Any]] = []
    for raw in raw_list:
        m = _FILTER_RE.match(raw)
        if not m:
            logger.warning("Ignoring unparseable filter: '%s'", raw)
            continue
        field = m.group("field")
        op_str = m.group("op")
        value = _parse_filter_value(m.group("value"))
        parsed.append((field, _OPS[op_str], value))

    return parsed


def filter_papers(
    papers: list[dict],
    filters: dict | str | list | None = None,
) -> list[dict]:
    """Filter papers by field conditions.

    Supports top-level fields and metric fields (checked in that order).
    """
    conditions = parse_filters(filters)
    if not conditions:
        return papers

    result = []
    for p in papers:
        match = True
        for field, op_func, target in conditions:
            # Look in top-level, then metrics
            val = p.get(field)
            if val is None:
                val = (p.get("metrics") or {}).get(field)
            if val is None:
                match = False
                break
            try:
                if not op_func(val, target):
                    match = False
                    break
            except TypeError:
                match = False
                break
        if match:
            result.append(p)

    return result


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def sort_papers(
    papers: list[dict],
    keys: list[str] | None = None,
) -> list[dict]:
    """Sort papers by one or more keys.

    Key format: ``"field"`` (ascending) or ``"-field"`` (descending).
    Looks in top-level fields first, then ``metrics``.
    """
    if not keys:
        return sorted(papers, key=lambda p: p.get("paper_id", ""))

    def _sort_key(p: dict) -> tuple:
        parts = []
        for k in keys:
            desc = k.startswith("-")
            field = k.lstrip("-")
            val = p.get(field)
            if val is None:
                val = (p.get("metrics") or {}).get(field)
            if val is None:
                val = float("inf") if not desc else float("-inf")
            if desc and isinstance(val, (int, float)):
                val = -val
            parts.append(val)
        return tuple(parts)

    return sorted(papers, key=_sort_key)


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def group_papers(
    papers: list[dict],
    group_by: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Group papers by one or more fields.

    The group key is a ``|``-separated string of field values.
    """
    if not group_by:
        return {"all": papers}

    groups: dict[str, list[dict]] = {}
    for p in papers:
        parts = []
        for field in group_by:
            val = p.get(field, "unknown")
            if isinstance(val, list):
                val = ",".join(str(v) for v in val)
            parts.append(str(val))
        key = " | ".join(parts)
        groups.setdefault(key, []).append(p)

    # Sort keys for determinism
    return dict(sorted(groups.items()))
