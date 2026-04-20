"""
Web UI wallpaper: built-in static paths or user document image; persisted per user.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Optional, Tuple

from pydantic import BaseModel, Field

# Must match shipped files under frontend/public/wallpapers/ and frontend manifest keys.
UI_WALLPAPER_BUILTIN_KEYS = frozenset(
    {"honeycomb", "green", "lineoleum", "mono", "wheat"}
)

UiWallpaperSource = Literal["none", "builtin", "document"]
UiWallpaperSize = Literal["cover", "contain", "auto"]
UiWallpaperRepeat = Literal["no-repeat", "repeat"]

_DOCUMENT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class UiWallpaperConfig(BaseModel):
    version: int = 1
    enabled: bool = False
    source: UiWallpaperSource = "none"
    builtin_key: Optional[str] = None
    document_id: Optional[str] = None
    opacity: float = Field(default=0.62, ge=0.0, le=1.0)
    scrim_opacity: float = Field(default=0.22, ge=0.0, le=1.0)
    blur_px: float = Field(default=0.0, ge=0.0, le=20.0)
    size: UiWallpaperSize = "cover"
    repeat: UiWallpaperRepeat = "no-repeat"

    def normalized(self) -> "UiWallpaperConfig":
        if not self.enabled or self.source == "none":
            return self.model_copy(
                update={
                    "source": "none",
                    "builtin_key": None,
                    "document_id": None,
                }
            )
        if self.source == "builtin":
            key = (self.builtin_key or "").strip()
            if key not in UI_WALLPAPER_BUILTIN_KEYS:
                return empty_ui_wallpaper_config()
            return self.model_copy(
                update={"builtin_key": key, "document_id": None}
            )
        # document
        doc = (self.document_id or "").strip()
        if not doc or not _DOCUMENT_ID_PATTERN.match(doc):
            return empty_ui_wallpaper_config()
        return self.model_copy(
            update={"document_id": doc, "builtin_key": None}
        )


def empty_ui_wallpaper_config() -> UiWallpaperConfig:
    return UiWallpaperConfig(
        version=1,
        enabled=False,
        source="none",
        builtin_key=None,
        document_id=None,
        opacity=0.62,
        scrim_opacity=0.22,
        blur_px=0.0,
        size="cover",
        repeat="no-repeat",
    )


def parse_ui_wallpaper_config_json(raw: Optional[str]) -> Optional[UiWallpaperConfig]:
    if not raw or not str(raw).strip():
        return None
    try:
        data = json.loads(str(raw))
        if not isinstance(data, dict):
            return None
        return UiWallpaperConfig.model_validate(data)
    except Exception:
        return None


def validate_ui_wallpaper_config(config: UiWallpaperConfig) -> Tuple[bool, str]:
    cfg = config.model_copy()
    if not cfg.enabled:
        return True, ""
    if cfg.source == "none":
        return False, "When wallpaper is enabled, choose a built-in image or upload a custom image"
    if cfg.source == "builtin":
        key = (cfg.builtin_key or "").strip()
        if key not in UI_WALLPAPER_BUILTIN_KEYS:
            return False, "Unknown or missing builtin wallpaper key"
        return True, ""
    if cfg.source == "document":
        doc = (cfg.document_id or "").strip()
        if not doc:
            return False, "document_id required for document source"
        if not _DOCUMENT_ID_PATTERN.match(doc):
            return False, "Invalid document_id"
        return True, ""
    return False, "Invalid source"


def validate_and_normalize_payload(config: Any) -> Tuple[Optional[UiWallpaperConfig], str]:
    if isinstance(config, UiWallpaperConfig):
        cfg = config
    elif isinstance(config, dict):
        try:
            cfg = UiWallpaperConfig.model_validate(config)
        except Exception as e:
            return None, str(e)
    else:
        return None, "Invalid config type"
    ok, err = validate_ui_wallpaper_config(cfg)
    if not ok:
        return None, err
    return cfg.normalized(), ""
