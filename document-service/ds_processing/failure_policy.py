"""Classify processing failures and compute retry backoff (no external dependencies)."""

from __future__ import annotations

import asyncio
import random
from typing import Literal, Tuple

ErrorKind = Literal["transient", "terminal", "timeout", "dependency"]


def classify_error(exc: BaseException) -> ErrorKind:
    """Map an exception to a coarse error kind for retry policy."""
    from ds_processing.dep_guard import DependencyUnavailable

    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"
    if isinstance(exc, DependencyUnavailable):
        return "dependency"
    if isinstance(exc, (ConnectionError, OSError)):
        return "transient"
    # gRPC / aio unavailability
    name = type(exc).__name__
    if "Unavailable" in name or "Dead" in name or "ConnectionReset" in name:
        return "transient"
    if "InvalidArgument" in name or "ValidationError" in name or isinstance(exc, (ValueError, KeyError, TypeError)):
        return "terminal"
    return "transient"


def is_retryable(kind: ErrorKind) -> bool:
    return kind in ("transient", "timeout", "dependency")


def backoff_seconds(attempt_number: int, cap: float = 600.0) -> float:
    """Exponential backoff with jitter; attempt_number is 1-based after a failed attempt."""
    n = max(1, attempt_number)
    base = min(cap, float(5 * (2 ** (n - 1))))
    return base + random.uniform(0.0, 5.0)


def should_retry(kind: ErrorKind, current_attempt_number: int, max_attempts: int) -> Tuple[bool, str]:
    """current_attempt_number is the DB attempt_count after increment for this failed try."""
    if not is_retryable(kind):
        return False, "terminal"
    if current_attempt_number >= max_attempts:
        return False, "max_attempts"
    return True, ""
