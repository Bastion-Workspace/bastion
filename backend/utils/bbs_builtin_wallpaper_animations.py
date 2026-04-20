"""
Built-in BBS wallpaper animations (no user document). Referenced by sentinel ids in bbs_wallpaper_config.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

# Stable sentinel stored in `animation_document_id` when display_mode is `animated`.
BUILTIN_ANIM_MATRIX_RAIN = "__bastion_builtin_matrix_rain__"
BUILTIN_ANIM_SNOWMAN = "__bastion_builtin_snowman__"

_DEFAULT_COLS = 80
_DEFAULT_ROWS = 22
_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ@#$%^&*+=-[]|/\\"
# Dark-to-bright ramp (plain ASCII only — avoids ANSI when BBS truncates lines to term width).
_BRIGHTNESS = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B$@"


def _glyph_for_depth(k: int, tail: int, rng: random.Random) -> str:
    ch = rng.choice(_CHARS)
    if k == 0:
        idx = min(len(_BRIGHTNESS) - 1, len(_BRIGHTNESS) * 3 // 4 + (ord(ch) % 4))
    elif k < max(3, tail // 5):
        idx = min(len(_BRIGHTNESS) - 1, len(_BRIGHTNESS) // 2 + k * 2)
    else:
        idx = max(0, (len(_BRIGHTNESS) * (tail - k)) // max(tail, 1) - 1)
    return _BRIGHTNESS[idx] if _BRIGHTNESS[idx] != " " else ch


def _build_frame(
    rng: random.Random,
    cols: int,
    rows: int,
    heads: List[float],
    tails: List[int],
) -> List[str]:
    grid: List[List[str]] = [[" " for _ in range(cols)] for _ in range(rows)]

    for c in range(cols):
        h = heads[c]
        tail = tails[c]
        for k in range(tail):
            rr = int(h) - k
            if 0 <= rr < rows:
                grid[rr][c] = _glyph_for_depth(k, tail, rng)

    return ["".join(grid[r]) for r in range(rows)]


def matrix_rain_animation_payload(
    fps: float,
    loop: bool,
    *,
    cols: int = _DEFAULT_COLS,
    rows: int = _DEFAULT_ROWS,
) -> Dict[str, Any]:
    """Deterministic Matrix-style rain using plain single-width glyphs (no ANSI)."""
    cols = max(20, min(512, int(cols)))
    rows = max(6, min(120, int(rows)))
    rng = random.Random(42)
    heads = [rng.uniform(-rows * 2, rows) for _ in range(cols)]
    max_tail = min(24, max(6, rows))
    tails = [rng.randint(5, max_tail) for _ in range(cols)]
    # Enough frames for columns starting near -2*rows to advance through the grid to the
    # bottom drain zone (heads step ~0.55–0.79 per frame). Clamp keeps bundle size reasonable.
    min_step = 0.52
    span = rows * 2 + rows + max_tail + 16
    frame_count = int(span / min_step) + 24
    frame_count = max(96, min(400, frame_count))
    frames: List[str] = []
    for _ in range(frame_count):
        lines = _build_frame(rng, cols, rows, heads, tails)
        frames.append("\n".join(lines))
        for c in range(cols):
            heads[c] += 0.55 + (c % 7) * 0.04
            if heads[c] > rows + tails[c] + 4:
                heads[c] = rng.uniform(-rows * 2, -2)
                tails[c] = rng.randint(5, max_tail)
    return {
        "frames": frames,
        "fps": float(fps),
        "loop": bool(loop),
    }


_SNOWMAN_LARGE = [
    "    .-.     ",
    "   ( o o )   ",
    "  >(  :  )<  ",
    "   /|   |\\   ",
]
_SNOWMAN_SMALL = [
    "  ( o o )  ",
    " (   :   ) ",
]


def _center_block_lines(lines: List[str], cols: int) -> List[str]:
    block_w = max(len(s) for s in lines) if lines else 0
    block_w = min(block_w, cols) if cols else 0
    out: List[str] = []
    for raw in lines:
        core = raw.rstrip()
        if len(core) > block_w:
            core = core[:block_w]
        left = max(0, (cols - block_w) // 2)
        s = " " * left + core.ljust(block_w)[:block_w]
        if len(s) < cols:
            s += " " * (cols - len(s))
        else:
            s = s[:cols]
        out.append(s)
    return out


def snowman_winter_animation_payload(
    fps: float,
    loop: bool,
    *,
    cols: int = _DEFAULT_COLS,
    rows: int = _DEFAULT_ROWS,
) -> Dict[str, Any]:
    """Gentle snowfall above a centered ASCII snowman; plain ASCII for BBS terminals."""
    cols = max(20, min(512, int(cols)))
    rows = max(6, min(120, int(rows)))
    rng = random.Random(44)

    if rows >= 11:
        snow_lines = list(_SNOWMAN_LARGE)
        ground_h = 1
    elif rows >= 8:
        snow_lines = list(_SNOWMAN_SMALL)
        ground_h = 1
    else:
        snow_lines = [" (o) "]
        ground_h = 0

    snow_h = len(snow_lines)
    sky_h = rows - snow_h - ground_h
    if sky_h < 3:
        snow_lines = list(_SNOWMAN_SMALL) if snow_h > 2 else snow_lines
        snow_h = len(snow_lines)
        ground_h = 1 if rows >= snow_h + 4 else 0
        sky_h = max(2, rows - snow_h - ground_h)

    centered_snow = _center_block_lines(snow_lines, cols)
    flake_n = max(10, min(96, cols // 2 + 6))
    flakes: List[Dict[str, Any]] = []
    for _ in range(flake_n):
        flakes.append(
            {
                "y": rng.uniform(-float(sky_h) - 4, float(sky_h)),
                "x": rng.randint(0, cols - 1),
                "vy": rng.uniform(0.32, 0.72),
                "ch": rng.choice(".*'`"),
            }
        )

    min_vy = 0.32
    frame_count = int((sky_h * 2.6 + 36) / min_vy) + 24
    frame_count = max(72, min(320, frame_count))

    frames: List[str] = []
    for _ in range(frame_count):
        grid: List[List[str]] = [[" " for _ in range(cols)] for _ in range(rows)]
        # Snowman sits directly under the sky; the ground row is beneath his feet (not above).
        snow_top = sky_h

        for i, row_str in enumerate(centered_snow):
            r = snow_top + i
            if r < rows:
                for c in range(cols):
                    grid[r][c] = row_str[c] if c < len(row_str) else " "

        if ground_h:
            gr = snow_top + snow_h
            if gr < rows:
                grid[gr] = [c for c in ("-" * cols)[:cols]]

        for f in flakes:
            yi = int(f["y"])
            if 0 <= yi < sky_h:
                grid[yi][f["x"]] = str(f["ch"])

        frames.append("\n".join("".join(row) for row in grid))

        for f in flakes:
            f["y"] = float(f["y"]) + float(f["vy"])
            if f["y"] >= float(sky_h) - 0.01:
                f["y"] = rng.uniform(-float(sky_h) - 6, -0.5)
                f["x"] = rng.randint(0, cols - 1)
                f["vy"] = rng.uniform(0.32, 0.72)
                f["ch"] = rng.choice(".*'`")
            if rng.random() < 0.06:
                f["x"] = max(0, min(cols - 1, int(f["x"]) + rng.choice([-1, 0, 1])))

    return {
        "frames": frames,
        "fps": float(fps),
        "loop": bool(loop),
    }
