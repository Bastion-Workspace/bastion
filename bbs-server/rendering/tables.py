"""
Simple ASCII tables for BBS listings.
"""

from __future__ import annotations

from typing import Dict, List, Sequence


def table_line_width(n_cols: int, col_widths: Sequence[int]) -> int:
    """Pixel-free width of a row from render_table's fmt_row: '| ' + cells + ' |' joiners."""
    if n_cols <= 0:
        return 0
    return sum(col_widths) + 3 * n_cols + 1


def ordered_column_names(
    table_schema: Dict[str, Any], row_dicts: List[Dict[str, Any]]
) -> List[str]:
    """
    Column order: schema columns[].name first, then any row keys not in schema (sorted).
    """
    names: List[str] = []
    seen: set[str] = set()
    columns = table_schema.get("columns") if isinstance(table_schema, dict) else None
    if isinstance(columns, list):
        for c in columns:
            if isinstance(c, dict):
                nm = c.get("name")
                if isinstance(nm, str) and nm and nm not in seen:
                    names.append(nm)
                    seen.add(nm)
    extra: set[str] = set()
    for rd in row_dicts:
        for k in rd:
            if k not in seen:
                extra.add(k)
    names.extend(sorted(extra))
    return names


def fit_column_widths(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    max_width: int,
    min_col: int = 4,
    max_col: int = 28,
) -> List[int]:
    """
    Per-column display widths so the formatted header line fits in max_width when possible.
    Shrinks widest columns first until table_line_width <= max_width or all at min_col.
    """
    n = len(headers)
    if n == 0:
        return []
    w: List[int] = []
    for i in range(n):
        m = len(headers[i])
        for row in rows:
            if i < len(row):
                m = max(m, len(str(row[i])))
        w.append(min(max_col, max(min_col, m)))
    while table_line_width(n, w) > max_width and any(x > min_col for x in w):
        mx = max(w)
        idx = w.index(mx)
        w[idx] -= 1
    return w


def render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    col_widths: Sequence[int] | None = None,
    max_width: int = 78,
) -> str:
    if not headers:
        return ""
    n = len(headers)
    if col_widths is None:
        col_widths = [max(4, min(max_width // max(n, 1), 28)) for _ in range(n)]
    lines: List[str] = []

    def fmt_row(cells: Sequence[str]) -> str:
        parts: List[str] = []
        for i, c in enumerate(cells):
            wi = col_widths[i] if i < len(col_widths) else 12
            s = str(c) if c is not None else ""
            if len(s) > wi:
                s = s[: wi - 1] + "."
            parts.append(s.ljust(wi))
        return "| " + " | ".join(parts) + " |"

    head = fmt_row(headers)
    sep = "-" * len(head)
    lines.append(head)
    lines.append(sep)
    for row in rows:
        r = list(row) + [""] * (n - len(row))
        lines.append(fmt_row(r[:n]))
    return "\n".join(lines)
