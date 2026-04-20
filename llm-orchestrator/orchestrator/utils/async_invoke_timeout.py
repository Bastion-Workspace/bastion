"""
Optional asyncio timeouts for LLM and LangGraph ainvoke calls.

Defaults are off (no wait_for) so interactive behavior is unchanged unless env sets limits.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def invoke_with_optional_timeout(awaitable: Awaitable[T], timeout_sec: Optional[float]) -> T:
    """
    Await ``awaitable``, optionally bounded by ``timeout_sec``.

    ``timeout_sec`` of None or <= 0 means no outer timeout (same as a plain await).
    """
    if timeout_sec is None or timeout_sec <= 0:
        return await awaitable
    try:
        return await asyncio.wait_for(awaitable, timeout=float(timeout_sec))
    except asyncio.TimeoutError:
        logger.warning("Invoke exceeded timeout_sec=%s", timeout_sec)
        raise
