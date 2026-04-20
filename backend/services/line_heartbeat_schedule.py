"""
Agent line heartbeat schedule: validate schedule_type, timezone-aware cron, interval minimum.

Used by agent_line_service (persist next_beat_at), team_heartbeat_utils, and REST preview.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from croniter import croniter

logger = logging.getLogger(__name__)

SCHEDULE_TYPES = frozenset({"none", "interval", "cron"})
MIN_INTERVAL_SECONDS = 60


def _normalize_schedule_type(raw: Any) -> str:
    if raw is None or raw == "":
        return "none"
    s = str(raw).strip().lower()
    if s in SCHEDULE_TYPES:
        return s
    return "invalid"


def _infer_missing_schedule_type(cfg: Dict[str, Any]) -> None:
    """If schedule_type is absent, infer from cron_expression or interval_seconds (legacy rows)."""
    if not isinstance(cfg, dict):
        return
    if cfg.get("schedule_type") is not None and str(cfg.get("schedule_type")).strip() != "":
        return
    if (cfg.get("cron_expression") or "").strip():
        cfg["schedule_type"] = "cron"
        return
    try:
        v = int(cfg.get("interval_seconds") or 0)
        if v > 0:
            cfg["schedule_type"] = "interval"
    except (TypeError, ValueError):
        pass


def _timezone_or_utc(tz_str: Optional[str]):
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        ZoneInfo = None  # type: ignore
    if not tz_str or str(tz_str).strip().upper() == "UTC":
        return timezone.utc
    if ZoneInfo is None:
        return timezone.utc
    try:
        return ZoneInfo(str(tz_str).strip())
    except Exception:
        return None


def validate_heartbeat_schedule(cfg: Optional[Dict[str, Any]]) -> List[str]:
    """Return human-readable errors; empty list means schedule fields are consistent."""
    errors: List[str] = []
    if not cfg or not isinstance(cfg, dict):
        return errors
    cfg = dict(cfg)
    _infer_missing_schedule_type(cfg)
    st = _normalize_schedule_type(cfg.get("schedule_type"))
    if st == "invalid":
        errors.append("schedule_type must be one of: none, interval, cron")
        return errors
    if st == "interval":
        raw = cfg.get("interval_seconds")
        try:
            sec = int(raw) if raw is not None and raw != "" else 0
        except (TypeError, ValueError):
            errors.append("interval_seconds must be a positive integer")
            return errors
        if sec < MIN_INTERVAL_SECONDS:
            errors.append(f"interval_seconds must be at least {MIN_INTERVAL_SECONDS} (matches heartbeat poll cadence)")
        if (cfg.get("cron_expression") or "").strip():
            errors.append("cron_expression must be empty when schedule_type is interval")
    elif st == "cron":
        cron = (cfg.get("cron_expression") or "").strip()
        if not cron:
            errors.append("cron_expression is required when schedule_type is cron")
        else:
            try:
                tz = _timezone_or_utc(cfg.get("timezone"))
                if tz is None:
                    errors.append("timezone must be a valid IANA time zone name")
                else:
                    now = datetime.now(timezone.utc).astimezone(tz).replace(tzinfo=None)
                    croniter(cron, now)
            except Exception as e:
                errors.append(f"Invalid cron expression: {e}")
        raw = cfg.get("interval_seconds")
        if raw is not None and raw != "":
            try:
                if int(raw) > 0:
                    errors.append("interval_seconds must be empty when schedule_type is cron")
            except (TypeError, ValueError):
                errors.append("interval_seconds must be empty when schedule_type is cron")
    else:
        # none
        pass
    return errors


def compute_next_beat_at(
    heartbeat_config: Optional[Dict[str, Any]],
    from_time: Optional[datetime] = None,
) -> Optional[datetime]:
    """Next UTC beat time from config, or None if no periodic schedule."""
    if not heartbeat_config or not isinstance(heartbeat_config, dict):
        return None
    cfg = dict(heartbeat_config)
    _infer_missing_schedule_type(cfg)
    st = _normalize_schedule_type(cfg.get("schedule_type"))
    if st not in ("interval", "cron"):
        return None
    from_time = from_time or datetime.now(timezone.utc)
    tz_str = (cfg.get("timezone") or "UTC").strip() or "UTC"

    if st == "interval":
        try:
            sec = int(cfg.get("interval_seconds") or 0)
        except (TypeError, ValueError):
            return None
        if sec < MIN_INTERVAL_SECONDS:
            return None
        return from_time + timedelta(seconds=sec)

    cron = (cfg.get("cron_expression") or "").strip()
    if not cron:
        return None
    tz = _timezone_or_utc(tz_str if st == "cron" else "UTC")
    if tz is None:
        return None
    try:
        now_in_tz = from_time.astimezone(tz)
        now_naive = now_in_tz.replace(tzinfo=None)
        it = croniter(cron, now_naive)
        next_naive = it.get_next(datetime)
        next_in_tz = next_naive.replace(tzinfo=tz)
        return next_in_tz.astimezone(timezone.utc)
    except Exception as e:
        logger.debug("compute_next_beat_at cron failed: %s", e)
        return None


def preview_occurrences(
    heartbeat_config: Optional[Dict[str, Any]],
    count: int = 5,
    from_time: Optional[datetime] = None,
) -> Tuple[List[str], List[str]]:
    """
    Returns (errors, iso8601_utc_strings).
    Uses validate_heartbeat_schedule first; then walks forward from from_time.
    """
    cfg_in = heartbeat_config or {}
    errs = validate_heartbeat_schedule(cfg_in if isinstance(cfg_in, dict) else {})
    if errs:
        return errs, []
    cfg = dict(cfg_in) if isinstance(cfg_in, dict) else {}
    _infer_missing_schedule_type(cfg)
    st = _normalize_schedule_type(cfg.get("schedule_type"))
    if st == "none":
        return [], []
    count = max(1, min(int(count or 5), 24))
    t = from_time or datetime.now(timezone.utc)
    out: List[datetime] = []
    if st == "interval":
        try:
            sec = int(cfg.get("interval_seconds") or 0)
        except (TypeError, ValueError):
            return ["Invalid interval_seconds"], []
        if sec < MIN_INTERVAL_SECONDS:
            return [f"interval_seconds must be at least {MIN_INTERVAL_SECONDS}"], []
        cur = t
        for _ in range(count):
            cur = cur + timedelta(seconds=sec)
            out.append(cur)
        return [], [x.isoformat() for x in out]
    if st == "cron":
        cron = (cfg.get("cron_expression") or "").strip()
        tz_str = (cfg.get("timezone") or "UTC").strip() or "UTC"
        tz = _timezone_or_utc(tz_str)
        if tz is None:
            return ["Invalid timezone"], []
        try:
            cur_utc = t
            for _ in range(count):
                now_in_tz = cur_utc.astimezone(tz)
                now_naive = now_in_tz.replace(tzinfo=None)
                it = croniter(cron, now_naive)
                next_naive = it.get_next(datetime)
                next_in_tz = next_naive.replace(tzinfo=tz)
                cur_utc = next_in_tz.astimezone(timezone.utc)
                out.append(cur_utc)
        except Exception as e:
            return [f"Cron preview failed: {e}"], []
        return [], [x.isoformat() for x in out]
    return [], []
