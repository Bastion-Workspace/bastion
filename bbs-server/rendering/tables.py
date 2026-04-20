"""
Simple ASCII tables for BBS listings.
"""

from __future__ import annotations

from typing import List, Sequence


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
            w = col_widths[i] if i < len(col_widths) else 12
            s = str(c) if c is not None else ""
            if len(s) > w:
                s = s[: w - 1] + "."
            parts.append(s.ljust(w))
        return "| " + " | ".join(parts) + " |"

    head = fmt_row(headers)
    sep = "-" * len(head)
    lines.append(head)
    lines.append(sep)
    for row in rows:
        r = list(row) + [""] * (n - len(row))
        lines.append(fmt_row(r[:n]))
    return "\n".join(lines)
