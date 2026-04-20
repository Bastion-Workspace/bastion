"""
Parse BBS animated wallpaper sources: ASCII Motion JSON export or comma-line frame dumps.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

BBS_ANIM_MAX_FRAMES = 120
BBS_ANIM_MAX_RAW_CHARS = 600_000
BBS_ANIM_MAX_FRAME_CHARS = 120_000


def _clip_frames(frames: List[str]) -> List[str]:
    if len(frames) > BBS_ANIM_MAX_FRAMES:
        return frames[:BBS_ANIM_MAX_FRAMES]
    return frames


def _parse_ascii_motion(obj: Dict[str, Any]) -> Optional[Tuple[List[str], float, bool]]:
    frames_raw = obj.get("frames")
    if not isinstance(frames_raw, list) or not frames_raw:
        return None
    out: List[str] = []
    for fr in frames_raw:
        if not isinstance(fr, dict):
            continue
        cs = fr.get("contentString")
        if isinstance(cs, str) and cs.strip():
            text = cs
        else:
            lines = fr.get("content")
            if isinstance(lines, list):
                text = "\n".join(str(x) for x in lines)
            else:
                continue
        if len(text) > BBS_ANIM_MAX_FRAME_CHARS:
            text = text[:BBS_ANIM_MAX_FRAME_CHARS]
        out.append(text)
    if not out:
        return None
    anim = obj.get("animation") if isinstance(obj.get("animation"), dict) else {}
    fps = 12.0
    if isinstance(anim.get("frameRate"), (int, float)) and float(anim["frameRate"]) > 0:
        fps = float(anim["frameRate"])
    else:
        first = frames_raw[0] if frames_raw and isinstance(frames_raw[0], dict) else {}
        dur = first.get("duration")
        if isinstance(dur, (int, float)) and float(dur) > 0:
            fps = min(30.0, max(1.0, 1000.0 / float(dur)))
    loop = True
    if "looping" in anim:
        loop = bool(anim.get("looping"))
    fps = min(30.0, max(1.0, fps))
    return _clip_frames(out), fps, loop


# Lines containing only a comma (optional spaces) separate frames — matches ASCII Motion paste style.
_FRAME_SEP_PATTERN = re.compile(r"(?:\r?\n)+\s*,\s*(?:\r?\n)+", re.MULTILINE)


def _parse_comma_line_frames(raw: str) -> Optional[Tuple[List[str], float, bool]]:
    text = raw.strip("\ufeff")
    if not text or len(text) > BBS_ANIM_MAX_RAW_CHARS:
        return None
    parts = [p.strip("\r\n") for p in _FRAME_SEP_PATTERN.split(text)]
    parts = [p for p in parts if p.strip()]
    if len(parts) < 2:
        return None
    return _clip_frames(parts), 12.0, True


def parse_bbs_animation_document(text: str) -> Optional[Dict[str, Any]]:
    """
    Return a dict: frames (list of multiline strings), fps (float), loop (bool).
    None if unrecognized or invalid.
    """
    if not text or len(text) > BBS_ANIM_MAX_RAW_CHARS:
        return None
    stripped = text.lstrip("\ufeff").strip()
    if not stripped:
        return None
    if stripped.startswith("{"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(obj, dict) and isinstance(obj.get("frames"), list):
            parsed = _parse_ascii_motion(obj)
            if parsed:
                frames, fps, loop = parsed
                return {"frames": frames, "fps": fps, "loop": loop}
        return None
    parsed = _parse_comma_line_frames(text)
    if parsed:
        frames, fps, loop = parsed
        return {"frames": frames, "fps": fps, "loop": loop}
    return None
