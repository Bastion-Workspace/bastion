"""
Full-screen editor frame: word-wrapped viewport + status line (ANSI).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from rendering.ansi import Theme

from .wrap_layout import WrapLayout

if TYPE_CHECKING:
    from .buffer import TextBuffer


def render_frame(
    buf: "TextBuffer",
    layout: WrapLayout,
    scroll_visual: int,
    term_height: int,
    term_width: int,
    theme: Theme,
    title: str,
    dirty: bool,
    help_hint: bool,
) -> str:
    """
    Build complete screen: title row, wrapped body, status line(s).
    Cursor row uses inverse video at the column within the wrapped segment.
    """
    reset = theme.reset
    dim = theme.dim
    bold = theme.bold
    title_c = theme.title_color()
    inv_on = "\033[7m"
    inv_off = reset

    tw = max(40, term_width)
    status_lines = 2 if help_hint else 1
    body_h = max(3, term_height - 1 - status_lines)
    text_w = tw - 1

    lines_out: List[str] = []

    t = (title or "Document")[: text_w - 10]
    lines_out.append(f"{title_c}{bold}{t}{reset}")

    total_v = layout.total_visual()
    last_scroll = max(0, total_v - body_h)
    sv = max(0, min(scroll_visual, last_scroll))

    vi_cursor = layout.visual_from_logical(buf.row, buf.col)

    for k in range(body_h):
        vidx = sv + k
        if vidx >= total_v:
            lines_out.append(dim + "~" + reset)
            continue
        r, lo, hi = layout.segment_at(vidx)
        line = buf.lines[r] if r < len(buf.lines) else ""
        raw = line[lo:hi]
        seg = raw.ljust(text_w)[:text_w]

        if vidx == vi_cursor and r == buf.row and lo <= buf.col <= hi:
            cx = buf.col - lo
            if cx < len(raw):
                ch = seg[cx : cx + 1] or " "
                left = seg[:cx]
                right = seg[cx + 1 :]
                line_out = f"{left}{inv_on}{ch}{inv_off}{right}"
            else:
                line_out = dim + seg + reset + inv_on + " " + inv_off
        else:
            line_out = dim + seg + reset
        lines_out.append(line_out[:tw])

    ln, co = buf.cursor_row_col_1based()
    mod = " *MOD*" if dirty else ""
    status1 = (
        f"{dim}Ln {ln}:{co}{mod}  ^S save  ^Q quit  ^G help  ^A AI  ^P/^N up/dn  ^F/^B lr{reset}"
    )
    if help_hint:
        status2 = (
            f"{dim}Arrows/PgUp/PgDn; ^P/^N if arrows fail; wrapped lines.{reset}"
        )
    else:
        status2 = ""
    lines_out.append(status1[:tw])
    if status2:
        lines_out.append(status2[:tw])

    return "\r\n".join(lines_out) + "\r\n"


def scroll_ensure_visible(scroll_visual: int, cursor_vi: int, viewport_h: int, total_visual: int) -> int:
    if total_visual <= 0:
        return 0
    if cursor_vi < scroll_visual:
        return cursor_vi
    if cursor_vi >= scroll_visual + viewport_h:
        return cursor_vi - viewport_h + 1
    return scroll_visual
