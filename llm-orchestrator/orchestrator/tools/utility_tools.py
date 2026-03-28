"""
Utility Tools - Pure in-process state and date/number manipulation for Agent Factory.

Zone 1 (orchestrator): no gRPC, no backend_tool_client. Used as pipeline steps and
by LLM agent steps for counters, date windows, parsing scraped dates, and accumulators.
"""

import logging
import re
from datetime import datetime, timedelta, time
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# adjust_number
# ---------------------------------------------------------------------------


class AdjustNumberInputs(BaseModel):
    """Required inputs for adjust_number_tool."""
    value: float = Field(description="Current numeric value to adjust")


class AdjustNumberParams(BaseModel):
    """Optional parameters for adjust_number_tool."""
    amount: float = Field(default=1.0, description="Amount to adjust by")
    operation: str = Field(default="add", description="'add' or 'subtract'")
    min_value: Optional[float] = Field(default=None, description="Floor clamp")
    max_value: Optional[float] = Field(default=None, description="Ceiling clamp")


class AdjustNumberOutputs(BaseModel):
    """Typed outputs for adjust_number_tool."""
    value: float = Field(description="New value after adjustment")
    previous: float = Field(description="Value before adjustment")
    clamped: bool = Field(description="Whether min/max clamp was applied")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


def adjust_number_tool(
    value: float,
    amount: float = 1.0,
    operation: str = "add",
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> dict:
    """
    Increment or decrement a numeric value with optional clamping.
    Use in loops for counters, rate limits, or scoring.
    """
    try:
        prev = float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        prev = 0.0
    delta = amount if operation.strip().lower() == "add" else -amount
    new_value = prev + delta
    clamped = False
    if min_value is not None and new_value < min_value:
        new_value = min_value
        clamped = True
    if max_value is not None and new_value > max_value:
        new_value = max_value
        clamped = True
    suffix = " (clamped)" if clamped else ""
    return {
        "value": new_value,
        "previous": prev,
        "clamped": clamped,
        "formatted": f"{prev} → {new_value}{suffix}",
    }


register_action(
    name="adjust_number",
    category="utility",
    description="Increment or decrement a numeric value with optional clamping; use for counters and rate limits",
    inputs_model=AdjustNumberInputs,
    params_model=AdjustNumberParams,
    outputs_model=AdjustNumberOutputs,
    tool_function=adjust_number_tool,
)

# ---------------------------------------------------------------------------
# adjust_date (uses dateutil.relativedelta for months/years)
# ---------------------------------------------------------------------------


class AdjustDateInputs(BaseModel):
    """Required inputs for adjust_date_tool."""
    date: str = Field(description="ISO 8601 date/datetime, or 'now' for current time")


class AdjustDateParams(BaseModel):
    """Optional parameters for adjust_date_tool."""
    amount: int = Field(default=1, description="Units to shift (negative for backward)")
    unit: str = Field(
        default="days",
        description="minutes, hours, days, weeks, months, years",
    )
    set_time: Optional[str] = Field(
        default=None,
        description="'start_of_day', 'end_of_day', or 'HH:MM:SS'",
    )
    timezone: str = Field(default="UTC", description="Timezone for 'now' resolution")


class AdjustDateOutputs(BaseModel):
    """Typed outputs for adjust_date_tool."""
    date: str = Field(description="Adjusted ISO 8601 datetime")
    date_only: str = Field(description="Date portion YYYY-MM-DD")
    original: str = Field(description="Input normalized to ISO")
    day_of_week: str = Field(description="Day name e.g. Monday")
    unix_timestamp: float = Field(description="Epoch seconds")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


def _parse_dt_and_tz(date_str: str, tz_name: str = "UTC") -> datetime:
    """Parse date string or 'now' into datetime; apply timezone if available."""
    import pytz
    try:
        tz = pytz.timezone(tz_name) if tz_name else pytz.UTC
    except Exception:
        tz = pytz.UTC
    s = (date_str or "").strip().lower()
    if s in ("now", ""):
        return datetime.now(tz)
    try:
        from dateutil import parser as dateutil_parser
        dt = dateutil_parser.parse(date_str)
        if dt.tzinfo is None:
            dt = tz.localize(dt)
        return dt
    except Exception:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_str.strip()[:26], fmt.replace("%z", "").rstrip("T"))
            if dt.tzinfo is None:
                dt = tz.localize(dt)
            return dt
        except ValueError:
            continue
    dt = datetime.now(tz)
    return dt


def _apply_set_time(dt: datetime, set_time: Optional[str]) -> datetime:
    """Apply set_time: start_of_day, end_of_day, or HH:MM:SS."""
    if not set_time or not set_time.strip():
        return dt
    s = set_time.strip().lower()
    if s == "start_of_day":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if s == "end_of_day":
        return dt.replace(hour=23, minute=59, second=59, microsecond=999999)
    if re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", s):
        parts = s.split(":")
        h = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 0
        sec = int(parts[2]) if len(parts) > 2 else 0
        return dt.replace(hour=h, minute=m, second=sec, microsecond=0)
    return dt


def adjust_date_tool(
    date: str,
    amount: int = 1,
    unit: str = "days",
    set_time: Optional[str] = None,
    timezone: str = "UTC",
) -> dict:
    """
    Shift a date forward or backward by a duration. Accepts 'now' for current time.
    Units: minutes, hours, days, weeks, months, years.
    """
    try:
        dt = _parse_dt_and_tz(date, timezone)
    except Exception as e:
        logger.warning("adjust_date parse failed: %s", e)
        import pytz
        dt = datetime.now(pytz.timezone(timezone) if timezone else pytz.UTC)
    original_iso = dt.isoformat()
    unit = (unit or "days").strip().lower()
    try:
        from dateutil.relativedelta import relativedelta
        if unit == "minutes":
            delta = timedelta(minutes=amount)
        elif unit == "hours":
            delta = timedelta(hours=amount)
        elif unit == "days":
            delta = timedelta(days=amount)
        elif unit == "weeks":
            delta = timedelta(weeks=amount)
        elif unit in ("months", "years"):
            delta = relativedelta(**{unit: amount})
            dt = dt + delta
            dt = _apply_set_time(dt, set_time)
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            return {
                "date": dt.isoformat(),
                "date_only": dt.strftime("%Y-%m-%d"),
                "original": original_iso,
                "day_of_week": day_names[dt.weekday()],
                "unix_timestamp": dt.timestamp(),
                "formatted": f"{original_iso} → {dt.isoformat()} ({amount:+d} {unit})",
            }
        else:
            delta = timedelta(days=amount)
        dt = dt + delta
    except ImportError:
        if unit in ("months", "years"):
            dt = dt + timedelta(days=amount * (30 if unit == "months" else 365))
        else:
            delta = timedelta(
                minutes=amount if unit == "minutes" else amount * (60 if unit == "hours" else 24 * (7 if unit == "weeks" else 1))
            )
            dt = dt + delta
    dt = _apply_set_time(dt, set_time)
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return {
        "date": dt.isoformat(),
        "date_only": dt.strftime("%Y-%m-%d"),
        "original": original_iso,
        "day_of_week": day_names[dt.weekday()],
        "unix_timestamp": dt.timestamp(),
        "formatted": f"{original_iso} → {dt.isoformat()} ({amount:+d} {unit})",
    }


register_action(
    name="adjust_date",
    category="utility",
    description="Shift a date forward or backward by minutes, hours, days, weeks, months, or years; input 'now' for current time",
    inputs_model=AdjustDateInputs,
    params_model=AdjustDateParams,
    outputs_model=AdjustDateOutputs,
    tool_function=adjust_date_tool,
)

# ---------------------------------------------------------------------------
# parse_date
# ---------------------------------------------------------------------------


class ParseDateInputs(BaseModel):
    """Required inputs for parse_date_tool."""
    text: str = Field(
        description="Date string to parse (e.g. 'Feb 19, 2026', '19/02/2026', 'yesterday')"
    )


class ParseDateParams(BaseModel):
    """Optional parameters for parse_date_tool."""
    timezone: str = Field(default="UTC", description="Assumed timezone if input has none")
    reference_date: Optional[str] = Field(
        default=None, description="Reference for relative dates; defaults to now"
    )


class ParseDateOutputs(BaseModel):
    """Typed outputs for parse_date_tool."""
    date: str = Field(description="Normalized ISO 8601 datetime")
    date_only: str = Field(description="Date portion YYYY-MM-DD")
    time_only: Optional[str] = Field(default=None, description="Time portion HH:MM:SS or None")
    day_of_week: str = Field(description="Day name")
    unix_timestamp: float = Field(description="Epoch seconds")
    relative: str = Field(description="Human-friendly relative description")
    success: bool = Field(description="Whether parsing succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


def _parse_relative_date(text: str, ref_dt: datetime) -> Optional[datetime]:
    """Handle yesterday, today, tomorrow, last week, next week, N days ago, etc."""
    t = text.strip().lower()
    if t in ("today", "now"):
        return ref_dt.replace(hour=0, minute=0, second=0, microsecond=0) if t == "today" else ref_dt
    if t == "yesterday":
        return (ref_dt - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    if t == "tomorrow":
        return (ref_dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    m = re.match(r"^(\d+)\s+days?\s+ago$", t)
    if m:
        return (ref_dt - timedelta(days=int(m.group(1)))).replace(hour=0, minute=0, second=0, microsecond=0)
    m = re.match(r"^in\s+(\d+)\s+days?$", t)
    if m:
        return (ref_dt + timedelta(days=int(m.group(1)))).replace(hour=0, minute=0, second=0, microsecond=0)
    if "last week" in t:
        return (ref_dt - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
    if "next week" in t:
        return (ref_dt + timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
    return None


def _relative_description(dt: datetime, ref_dt: datetime) -> str:
    """Produce a short relative description."""
    delta = dt - ref_dt
    days = delta.days
    if days == 0:
        return "today"
    if days == -1:
        return "yesterday"
    if days == 1:
        return "tomorrow"
    if -7 <= days < 0:
        return f"{abs(days)} days ago"
    if 0 < days <= 7:
        return f"in {days} days"
    if -30 <= days < -7:
        return f"{abs(days)} days ago"
    if 7 < days <= 30:
        return f"in {days} days"
    return dt.strftime("%Y-%m-%d")


def parse_date_tool(
    text: str,
    timezone: str = "UTC",
    reference_date: Optional[str] = None,
) -> dict:
    """
    Parse a date string in common formats and normalize to ISO 8601.
    Handles relative phrases like 'yesterday', '3 days ago', 'today'.
    """
    import pytz
    tz = pytz.timezone(timezone) if timezone else pytz.UTC
    ref_dt = _parse_dt_and_tz(reference_date or "now", timezone) if reference_date else datetime.now(tz)
    if not text or not str(text).strip():
        return {
            "date": ref_dt.isoformat(),
            "date_only": ref_dt.strftime("%Y-%m-%d"),
            "time_only": ref_dt.strftime("%H:%M:%S"),
            "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][ref_dt.weekday()],
            "unix_timestamp": ref_dt.timestamp(),
            "relative": "now",
            "success": False,
            "error": "Empty input",
            "formatted": "Parse failed: empty input",
        }
    text = str(text).strip()
    dt = _parse_relative_date(text, ref_dt)
    if dt is not None:
        if dt.tzinfo is None:
            dt = tz.localize(dt)
        rel = _relative_description(dt, ref_dt)
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return {
            "date": dt.isoformat(),
            "date_only": dt.strftime("%Y-%m-%d"),
            "time_only": dt.strftime("%H:%M:%S"),
            "day_of_week": day_names[dt.weekday()],
            "unix_timestamp": dt.timestamp(),
            "relative": rel,
            "success": True,
            "error": None,
            "formatted": f"Parsed '{text}' → {dt.strftime('%Y-%m-%d')} ({day_names[dt.weekday()]})",
        }
    try:
        from dateutil import parser as dateutil_parser
        dt = dateutil_parser.parse(text)
        if dt.tzinfo is None:
            dt = tz.localize(dt)
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        rel = _relative_description(dt, ref_dt)
        time_only = dt.strftime("%H:%M:%S") if (dt.hour or dt.minute or dt.second) else None
        return {
            "date": dt.isoformat(),
            "date_only": dt.strftime("%Y-%m-%d"),
            "time_only": time_only,
            "day_of_week": day_names[dt.weekday()],
            "unix_timestamp": dt.timestamp(),
            "relative": rel,
            "success": True,
            "error": None,
            "formatted": f"Parsed '{text}' → {dt.isoformat()} ({day_names[dt.weekday()]})",
        }
    except Exception as e:
        err = str(e)
        return {
            "date": ref_dt.isoformat(),
            "date_only": ref_dt.strftime("%Y-%m-%d"),
            "time_only": None,
            "day_of_week": "",
            "unix_timestamp": ref_dt.timestamp(),
            "relative": "",
            "success": False,
            "error": err,
            "formatted": f"Parse failed: {err}",
        }


register_action(
    name="parse_date",
    category="utility",
    description="Parse a date string in any common format (or relative like 'yesterday') and normalize to ISO 8601",
    inputs_model=ParseDateInputs,
    params_model=ParseDateParams,
    outputs_model=ParseDateOutputs,
    tool_function=parse_date_tool,
)

# ---------------------------------------------------------------------------
# compare_dates
# ---------------------------------------------------------------------------


class CompareDatesInputs(BaseModel):
    """Required inputs for compare_dates_tool."""
    date_a: str = Field(description="First date (ISO 8601 or 'now')")
    date_b: str = Field(description="Second date (ISO 8601 or 'now')")


class CompareDatesOutputs(BaseModel):
    """Typed outputs for compare_dates_tool."""
    a_is_before_b: bool = Field(description="True if date_a is before date_b")
    a_is_after_b: bool = Field(description="True if date_a is after date_b")
    a_equals_b: bool = Field(description="True if date_a equals date_b")
    difference_days: float = Field(description="Signed difference in days (a - b)")
    difference_hours: float = Field(description="Signed difference in hours")
    difference_minutes: float = Field(description="Signed difference in minutes")
    absolute_days: float = Field(description="Absolute difference in days")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


def compare_dates_tool(date_a: str, date_b: str) -> dict:
    """
    Compare two dates and return structured booleans and differences.
    Inputs can be ISO 8601 or 'now'.
    """
    try:
        dt_a = _parse_dt_and_tz(date_a or "now", "UTC")
        dt_b = _parse_dt_and_tz(date_b or "now", "UTC")
    except Exception as e:
        logger.warning("compare_dates parse failed: %s", e)
        import pytz
        dt_a = dt_b = datetime.now(pytz.UTC)
    delta = dt_a - dt_b
    total_seconds = delta.total_seconds()
    diff_days = total_seconds / (24 * 3600)
    diff_hours = total_seconds / 3600
    diff_mins = total_seconds / 60
    a_before = total_seconds < 0
    a_after = total_seconds > 0
    a_eq = abs(total_seconds) < 1e-6
    abs_days = abs(diff_days)
    if a_before:
        desc = f"{dt_a.isoformat()} is {abs_days:.1f} days before {dt_b.isoformat()}"
    elif a_after:
        desc = f"{dt_a.isoformat()} is {abs_days:.1f} days after {dt_b.isoformat()}"
    else:
        desc = f"{dt_a.isoformat()} equals {dt_b.isoformat()}"
    return {
        "a_is_before_b": a_before,
        "a_is_after_b": a_after,
        "a_equals_b": a_eq,
        "difference_days": round(diff_days, 6),
        "difference_hours": round(diff_hours, 6),
        "difference_minutes": round(diff_mins, 6),
        "absolute_days": round(abs_days, 6),
        "formatted": desc,
    }


register_action(
    name="compare_dates",
    category="utility",
    description="Compare two dates and return whether A is before/after/equal to B and the difference in days/hours/minutes",
    inputs_model=CompareDatesInputs,
    outputs_model=CompareDatesOutputs,
    tool_function=compare_dates_tool,
)

# ---------------------------------------------------------------------------
# set_value
# ---------------------------------------------------------------------------


class SetValueInputs(BaseModel):
    """Required inputs for set_value_tool."""
    value: Any = Field(description="Value to store (text, number, boolean, record, or list)")


class SetValueOutputs(BaseModel):
    """Typed outputs for set_value_tool."""
    value: Any = Field(description="The stored value")
    value_type: str = Field(description="One of: text, number, boolean, list, record")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


def _value_type(v: Any) -> str:
    if v is None:
        return "text"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "record"
    return "text"


def set_value_tool(value: Any) -> dict:
    """
    Store a value in playbook state (identity). Use to initialize counters,
    rename upstream outputs, or inject constants.
    """
    vt = _value_type(value)
    formatted = f"Stored {vt}: {value!r}" if vt != "record" and vt != "list" else f"Stored {vt} (len={len(value)})"
    return {
        "value": value,
        "value_type": vt,
        "formatted": formatted,
    }


register_action(
    name="set_value",
    category="utility",
    description="Store a value in state (identity); use to initialize counters or pass constants to downstream steps",
    inputs_model=SetValueInputs,
    outputs_model=SetValueOutputs,
    tool_function=set_value_tool,
)

# ---------------------------------------------------------------------------
# toggle_boolean
# ---------------------------------------------------------------------------


class ToggleBooleanInputs(BaseModel):
    """Required inputs for toggle_boolean_tool."""
    value: bool = Field(description="Current boolean value to flip")


class ToggleBooleanOutputs(BaseModel):
    """Typed outputs for toggle_boolean_tool."""
    value: bool = Field(description="The flipped value")
    previous: bool = Field(description="Value before flip")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


def toggle_boolean_tool(value: bool) -> dict:
    """
    Flip a boolean. Useful for alternating behavior across loop iterations.
    """
    prev = bool(value) if value is not None else False
    new_val = not prev
    return {
        "value": new_val,
        "previous": prev,
        "formatted": f"{prev} → {new_val}",
    }


register_action(
    name="toggle_boolean",
    category="utility",
    description="Flip a boolean (true to false, false to true); use for alternating behavior in loops",
    inputs_model=ToggleBooleanInputs,
    outputs_model=ToggleBooleanOutputs,
    tool_function=toggle_boolean_tool,
)

# ---------------------------------------------------------------------------
# append_to_list
# ---------------------------------------------------------------------------


class AppendToListInputs(BaseModel):
    """Required inputs for append_to_list_tool."""
    items: Any = Field(description="Existing list (or empty for first call)")
    new_item: Any = Field(description="Item to append")


class AppendToListOutputs(BaseModel):
    """Typed outputs for append_to_list_tool."""
    items: List[Any] = Field(description="Updated list")
    count: int = Field(description="Length after append")
    last_added: Any = Field(description="The item that was appended")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


def append_to_list_tool(items: Any, new_item: Any) -> dict:
    """
    Add an item to an accumulator list. Self-reference items from previous
    step to collect results across loop iterations.
    """
    if items is None:
        lst = []
    elif isinstance(items, list):
        lst = list(items)
    else:
        lst = [items]
    lst.append(new_item)
    return {
        "items": lst,
        "count": len(lst),
        "last_added": new_item,
        "formatted": f"Appended 1 item; list length is now {len(lst)}",
    }


register_action(
    name="append_to_list",
    category="utility",
    description="Append an item to a list; wire items from previous step (or empty) to accumulate across loop iterations",
    inputs_model=AppendToListInputs,
    outputs_model=AppendToListOutputs,
    tool_function=append_to_list_tool,
)

# ---------------------------------------------------------------------------
# get_list_length
# ---------------------------------------------------------------------------


class GetListLengthInputs(BaseModel):
    """Required inputs for get_list_length_tool."""
    items: Any = Field(description="A list to measure")


class GetListLengthOutputs(BaseModel):
    """Typed outputs for get_list_length_tool."""
    count: int = Field(description="Length of the list")
    is_empty: bool = Field(description="True if list is empty")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


def get_list_length_tool(items: Any) -> dict:
    """
    Get the length of a list. Use in branch conditions without expression evaluation.
    """
    if items is None:
        n = 0
        lst = []
    elif isinstance(items, list):
        lst = items
        n = len(lst)
    else:
        lst = [items]
        n = 1
    return {
        "count": n,
        "is_empty": n == 0,
        "formatted": f"List length: {n}",
    }


register_action(
    name="get_list_length",
    category="utility",
    description="Get the length of a list; useful for branch conditions",
    inputs_model=GetListLengthInputs,
    outputs_model=GetListLengthOutputs,
    tool_function=get_list_length_tool,
)

# ---------------------------------------------------------------------------
# get_day_window
# ---------------------------------------------------------------------------


class GetDayWindowInputs(BaseModel):
    """No required inputs; use params for offset_days and timezone."""


class GetDayWindowParams(BaseModel):
    """Optional parameters for get_day_window_tool."""
    offset_days: int = Field(default=0, description="0=today, -1=yesterday, 1=tomorrow")
    timezone: str = Field(default="UTC", description="IANA timezone")


class GetDayWindowOutputs(BaseModel):
    """Typed outputs for get_day_window_tool."""
    day_start: str = Field(description="ISO 8601 start of day (00:00:00)")
    day_end: str = Field(description="ISO 8601 end of day (23:59:59)")
    today_date: str = Field(description="Date portion YYYY-MM-DD")
    current_time: str = Field(description="Current time ISO 8601")
    day_of_week: str = Field(description="Day name e.g. Monday")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


def get_day_window_tool(
    offset_days: int = 0,
    timezone: str = "UTC",
) -> dict:
    """
    Return the start and end of a day as ISO 8601 datetimes. Use as first step in playbooks
    when calendar or time-based steps need today's window; wire {day_window.day_start} and
    {day_window.day_end} to downstream steps.
    """
    try:
        ref_dt = _parse_dt_and_tz("now", timezone)
    except Exception as e:
        logger.warning("get_day_window parse failed: %s", e)
        import pytz
        ref_dt = datetime.now(pytz.timezone(timezone) if timezone else pytz.UTC)
    ref_dt = ref_dt + timedelta(days=offset_days)
    day_start_dt = _apply_set_time(ref_dt, "start_of_day")
    day_end_dt = _apply_set_time(ref_dt, "end_of_day")
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return {
        "day_start": day_start_dt.isoformat(),
        "day_end": day_end_dt.isoformat(),
        "today_date": ref_dt.strftime("%Y-%m-%d"),
        "current_time": ref_dt.isoformat(),
        "day_of_week": day_names[ref_dt.weekday()],
        "formatted": f"{ref_dt.strftime('%Y-%m-%d')} ({day_names[ref_dt.weekday()]}): {day_start_dt.isoformat()} — {day_end_dt.isoformat()}",
    }


register_action(
    name="get_day_window",
    category="utility",
    description="Return start and end of day as ISO 8601; use as first step, wire day_start/day_end to calendar steps",
    inputs_model=GetDayWindowInputs,
    params_model=GetDayWindowParams,
    outputs_model=GetDayWindowOutputs,
    tool_function=get_day_window_tool,
)
