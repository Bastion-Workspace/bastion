"""
Paginated terminal output with --More-- prompt.
"""

from __future__ import annotations

from typing import Awaitable, Callable, List, Optional

from rendering.ansi import Theme


async def _erase_pager_prompt_line(
    write_lines: Callable[[str], Awaitable[None]], theme: Theme
) -> None:
    """Remove the last printed --More-- line so it does not scroll in as part of the body."""
    if not theme.clear_screen:
        return
    # Cursor is on the blank line after the prompt; move up to the prompt row and erase it.
    await write_lines("\x1b[1A\x1b[2K\r")


async def paginate_text(
    lines: List[str],
    page_height: int,
    theme: Theme,
    write_lines: Callable[[str], Awaitable[None]],
    read_more: Callable[[], Awaitable[str]],
    *,
    after_line_input_drain: Optional[Callable[[], Awaitable[None]]] = None,
) -> None:
    if page_height < 5:
        page_height = 5
    total = len(lines)
    if total == 0:
        return
    if after_line_input_drain:
        await after_line_input_drain()

    back_starts: List[int] = []
    start = 0

    while True:
        chunk_end = min(start + page_height, total)
        for j in range(start, chunk_end):
            await write_lines(lines[j] + "\r\n")
        if chunk_end >= total:
            break
        await write_lines(
            theme.dim
            + "--More-- Space/Enter/n=next  b/p=prev  Q=quit"
            + theme.reset
            + "\r\n"
        )
        action = await read_more()
        if action == "quit":
            break
        if action == "prev":
            if back_starts:
                start = back_starts.pop()
            await _erase_pager_prompt_line(write_lines, theme)
            continue
        if action == "next":
            back_starts.append(start)
            start = chunk_end
            await _erase_pager_prompt_line(write_lines, theme)
            continue
        continue
