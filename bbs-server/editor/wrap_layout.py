"""
Word-wrap layout for the BBS editor: logical (row, col) <-> visual row index for telnet display.
"""

from __future__ import annotations

from typing import List, Tuple


def _wrap_line_segments(line: str, width: int) -> List[Tuple[int, int]]:
    """
    Partition one logical line into visual rows. Each segment is (lo, hi) where
    line[lo:hi] is shown on one row; valid cursor columns are lo..hi inclusive (gap model).
    """
    w = max(8, width)
    n = len(line)
    if n == 0:
        return [(0, 0)]
    out: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        j = min(i + w, n)
        if j < n:
            sp = line.rfind(" ", i, j + 1)
            if sp > i:
                j = sp
        out.append((i, j))
        while j < n and line[j] == " ":
            j += 1
        i = j
    return out


class WrapLayout:
    """Maps between buffer lines and visual (wrapped) rows."""

    def __init__(self, lines: List[str], width: int) -> None:
        self._width = max(8, width)
        self._lines = lines
        self._segments: List[Tuple[int, int, int]] = []
        self._rebuild()

    def _rebuild(self) -> None:
        self._segments.clear()
        for r, line in enumerate(self._lines):
            for lo, hi in _wrap_line_segments(line, self._width):
                self._segments.append((r, lo, hi))

    def refresh(self, lines: List[str]) -> None:
        self._lines = lines
        self._rebuild()

    def total_visual(self) -> int:
        return len(self._segments)

    def segment_at(self, vi: int) -> Tuple[int, int, int]:
        if not self._segments:
            return (0, 0, 0)
        vi = max(0, min(vi, len(self._segments) - 1))
        return self._segments[vi]

    def visual_from_logical(self, row: int, col: int) -> int:
        """Visual row index that should contain the cursor."""
        if not self._segments:
            return 0
        col = max(0, col)
        best = 0
        best_lo = -1
        for i, (r, lo, hi) in enumerate(self._segments):
            if r != row:
                continue
            if lo <= col <= hi:
                if lo >= best_lo:
                    best_lo = lo
                    best = i
        return best

    def logical_at_visual(self, vi: int, prefer_col: int) -> Tuple[int, int]:
        """Cursor position after moving to visual row vi."""
        if not self._segments:
            return (0, 0)
        vi = max(0, min(vi, len(self._segments) - 1))
        r, lo, hi = self._segments[vi]
        line = self._lines[r] if r < len(self._lines) else ""
        L = len(line)
        col = min(max(prefer_col, lo), min(hi, L))
        return (r, col)

    def step_visual(self, vi: int, delta: int, prefer_col: int) -> Tuple[int, int]:
        """Move by delta visual rows."""
        if not self._segments:
            return (0, 0)
        n = len(self._segments)
        vi2 = max(0, min(n - 1, vi + delta))
        return self.logical_at_visual(vi2, prefer_col)
