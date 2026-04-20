"""
Shared connection budget for Telnet and SSH BBS listeners.
"""

from __future__ import annotations

import asyncio
from typing import Optional


class ConnectionBudget:
    """Limit concurrent BBS sessions across transports."""

    def __init__(self, max_connections: int) -> None:
        self._max = max(1, int(max_connections))
        self._lock = asyncio.Lock()
        self._active = 0

    async def acquire(self) -> bool:
        async with self._lock:
            if self._active >= self._max:
                return False
            self._active += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            if self._active > 0:
                self._active -= 1

    def count(self) -> int:
        """Approximate count for UI; may race slightly without holding the lock."""
        return self._active


_budget: Optional[ConnectionBudget] = None


def set_connection_budget(budget: ConnectionBudget) -> None:
    global _budget
    _budget = budget


def get_connection_budget() -> ConnectionBudget:
    if _budget is None:
        raise RuntimeError("ConnectionBudget not configured")
    return _budget
