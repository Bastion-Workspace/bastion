"""
BBS / user wallpaper library: multiple ASCII art items, static or timed cycle display.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

BBS_WALLPAPER_MAX_CONTENT_LEN = 16384
BBS_WALLPAPER_MAX_ITEMS = 20
BBS_WALLPAPER_CYCLE_MIN = 5
BBS_WALLPAPER_CYCLE_MAX = 86400


def validate_wallpaper_content(value: str) -> Tuple[bool, str]:
    """Allow tab, LF, CR, and printable characters (ord >= 32)."""
    if value is None:
        value = ""
    if len(value) > BBS_WALLPAPER_MAX_CONTENT_LEN:
        return (
            False,
            f"Wallpaper exceeds maximum length ({BBS_WALLPAPER_MAX_CONTENT_LEN} characters)",
        )
    for ch in value:
        o = ord(ch)
        if o < 32 and ch not in "\n\r\t":
            return (False, "Wallpaper contains disallowed control characters")
    return (True, "")


class WallpaperItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default="Untitled", max_length=120)
    content: str = Field(default="", max_length=BBS_WALLPAPER_MAX_CONTENT_LEN)
    enabled: bool = True

    @field_validator("content")
    @classmethod
    def check_content(cls, v: str) -> str:
        ok, err = validate_wallpaper_content(v or "")
        if not ok:
            raise ValueError(err)
        return v or ""


class BbsWallpaperConfig(BaseModel):
    version: int = 1
    display_mode: Literal["static", "cycle", "animated"] = "static"
    active_id: str = ""
    cycle_interval_seconds: int = Field(
        default=30, ge=BBS_WALLPAPER_CYCLE_MIN, le=BBS_WALLPAPER_CYCLE_MAX
    )
    items: List[WallpaperItem] = Field(
        default_factory=list, max_length=BBS_WALLPAPER_MAX_ITEMS
    )
    animation_document_id: str = Field(default="", max_length=120)
    animation_fps: float = Field(default=12.0, ge=1.0, le=30.0)
    animation_loop: bool = True

    def normalized(self) -> "BbsWallpaperConfig":
        if self.display_mode == "animated" and not (self.animation_document_id or "").strip():
            base = self.model_copy(update={"display_mode": "static"})
        else:
            base = self
        if not base.items:
            return base.model_copy(update={"active_id": ""})
        ids = {it.id for it in base.items}
        aid = base.active_id if base.active_id in ids else base.items[0].id
        return base.model_copy(update={"active_id": aid})


def resolve_display_wallpaper(config: BbsWallpaperConfig, now: Optional[datetime] = None) -> str:
    """Resolved multiline text for BBS (or other clients) at time `now` (UTC)."""
    cfg = config.normalized()
    if cfg.display_mode == "animated":
        return ""
    if not cfg.items:
        return ""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)

    enabled_ordered = [it for it in cfg.items if it.enabled]
    if cfg.display_mode == "cycle" and len(enabled_ordered) >= 2:
        interval = max(
            BBS_WALLPAPER_CYCLE_MIN,
            min(cfg.cycle_interval_seconds, BBS_WALLPAPER_CYCLE_MAX),
        )
        now_ts = int(now.timestamp())
        idx = (now_ts // interval) % len(enabled_ordered)
        return enabled_ordered[idx].content or ""

    for it in cfg.items:
        if it.id == cfg.active_id:
            return it.content or ""
    return cfg.items[0].content or ""


def cycling_active(config: BbsWallpaperConfig) -> bool:
    cfg = config.normalized()
    if cfg.display_mode == "animated":
        return False
    n = sum(1 for it in cfg.items if it.enabled)
    return cfg.display_mode == "cycle" and n >= 2


def config_from_legacy_string(legacy: str) -> BbsWallpaperConfig:
    """Single legacy `bbs_wallpaper` string -> one-item library."""
    text = legacy or ""
    ok, _ = validate_wallpaper_content(text)
    if not ok:
        text = ""
    iid = str(uuid.uuid4())
    item = WallpaperItem(id=iid, name="Imported", content=text, enabled=True)
    return BbsWallpaperConfig(
        version=1,
        display_mode="static",
        active_id=iid if text.strip() else "",
        cycle_interval_seconds=30,
        items=[item] if text.strip() else [],
        animation_document_id="",
        animation_fps=12.0,
        animation_loop=True,
    )


def empty_bbs_wallpaper_config() -> BbsWallpaperConfig:
    return BbsWallpaperConfig(
        version=1,
        display_mode="static",
        active_id="",
        cycle_interval_seconds=30,
        items=[],
        animation_document_id="",
        animation_fps=12.0,
        animation_loop=True,
    )


def parse_bbs_wallpaper_config_json(raw: Optional[str]) -> Optional[BbsWallpaperConfig]:
    if not raw or not str(raw).strip():
        return None
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        return BbsWallpaperConfig.model_validate(data)
    except Exception:
        return None


def validate_full_config(config: BbsWallpaperConfig) -> Tuple[bool, str]:
    """Structural validation before persist."""
    cfg = config.normalized()
    if cfg.display_mode == "animated":
        doc = (cfg.animation_document_id or "").strip()
        if not doc:
            return (
                False,
                "Animated mode requires a document upload, or choose the built-in Matrix rain demo",
            )
        if len(doc) < 8:
            return (False, "Invalid animation document id")
    seen: set[str] = set()
    for it in cfg.items:
        if it.id in seen:
            return (False, "Duplicate wallpaper item id")
        seen.add(it.id)
        ok, err = validate_wallpaper_content(it.content)
        if not ok:
            return (False, f"Item {it.name!r}: {err}")
    return (True, "")
