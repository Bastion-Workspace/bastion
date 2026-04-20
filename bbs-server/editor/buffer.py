"""
Mutable line-based text buffer with cursor for the BBS editor.
"""

from __future__ import annotations

from typing import List, Tuple


class TextBuffer:
    def __init__(self, text: str) -> None:
        if not text:
            self.lines: List[str] = [""]
        else:
            self.lines = text.split("\n")
        self.row = 0
        self.col = 0
        self._clamp_cursor()

    def to_text(self) -> str:
        return "\n".join(self.lines)

    def _clamp_cursor(self) -> None:
        self.row = max(0, min(self.row, len(self.lines) - 1))
        line = self.lines[self.row]
        self.col = max(0, min(self.col, len(line)))

    def set_cursor(self, row: int, col: int) -> None:
        self.row = row
        self.col = col
        self._clamp_cursor()

    def insert_char(self, ch: str) -> None:
        if not ch:
            return
        line = self.lines[self.row]
        self.lines[self.row] = line[: self.col] + ch + line[self.col :]
        self.col += len(ch)

    def insert_tab(self, spaces: int = 4) -> None:
        self.insert_char(" " * spaces)

    def newline(self) -> None:
        line = self.lines[self.row]
        left = line[: self.col]
        right = line[self.col :]
        self.lines[self.row] = left
        self.lines.insert(self.row + 1, right)
        self.row += 1
        self.col = 0

    def backspace(self) -> None:
        if self.col > 0:
            line = self.lines[self.row]
            self.lines[self.row] = line[: self.col - 1] + line[self.col :]
            self.col -= 1
        elif self.row > 0:
            prev_len = len(self.lines[self.row - 1])
            self.lines[self.row - 1] += self.lines[self.row]
            del self.lines[self.row]
            self.row -= 1
            self.col = prev_len

    def delete(self) -> None:
        line = self.lines[self.row]
        if self.col < len(line):
            self.lines[self.row] = line[: self.col] + line[self.col + 1 :]
        elif self.row < len(self.lines) - 1:
            self.lines[self.row] += self.lines[self.row + 1]
            del self.lines[self.row + 1]

    def move_up(self) -> None:
        if self.row > 0:
            self.row -= 1
            self._clamp_cursor()

    def move_down(self) -> None:
        if self.row < len(self.lines) - 1:
            self.row += 1
            self._clamp_cursor()

    def move_left(self) -> None:
        if self.col > 0:
            self.col -= 1
        elif self.row > 0:
            self.row -= 1
            self.col = len(self.lines[self.row])

    def move_right(self) -> None:
        line = self.lines[self.row]
        if self.col < len(line):
            self.col += 1
        elif self.row < len(self.lines) - 1:
            self.row += 1
            self.col = 0

    def move_home(self) -> None:
        self.col = 0

    def move_end(self) -> None:
        self.col = len(self.lines[self.row])

    def move_vert(self, delta: int) -> None:
        if delta == 0:
            return
        target = self.row + delta
        target = max(0, min(target, len(self.lines) - 1))
        self.row = target
        self._clamp_cursor()

    def cursor_row_col_1based(self) -> Tuple[int, int]:
        return self.row + 1, self.col + 1

    def cursor_offset(self) -> int:
        """Byte offset of cursor in the full text, matching web UI active_editor.cursor_offset."""
        return sum(len(line) + 1 for line in self.lines[: self.row]) + self.col
