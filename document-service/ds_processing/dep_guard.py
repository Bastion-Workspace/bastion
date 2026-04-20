"""Lightweight per-process circuit guard for external dependencies (embedding, vector, kg)."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, Literal, Optional

from ds_config import settings

logger = logging.getLogger(__name__)

State = Literal["closed", "open", "half_open"]


class DependencyUnavailable(Exception):
    """Raised when a dependency is in the open (tripped) circuit state."""

    def __init__(self, name: str, message: str = ""):
        self.name = name
        super().__init__(message or f"dependency {name} unavailable (circuit open)")


@dataclass
class _Circuit:
    state: State = "closed"
    failures: int = 0
    opened_at: float = 0.0


class DependencyGuard:
    """Rolling failure counter with cooldown; not distributed across replicas."""

    def __init__(self) -> None:
        self._circuits: Dict[str, _Circuit] = {}

    def _get(self, name: str) -> _Circuit:
        if name not in self._circuits:
            self._circuits[name] = _Circuit()
        return self._circuits[name]

    def record_success(self, name: str) -> None:
        c = self._get(name)
        c.failures = 0
        c.state = "closed"

    def record_failure(self, name: str) -> None:
        cfg = settings
        threshold = getattr(cfg, "DEP_GUARD_FAILURE_THRESHOLD", 5)
        c = self._get(name)
        c.failures += 1
        if c.failures >= threshold:
            c.state = "open"
            c.opened_at = time.monotonic()
            logger.warning("Circuit opened for %s after %s failures", name, c.failures)

    @asynccontextmanager
    async def guard(self, name: str):
        c = self._get(name)
        cooldown = float(getattr(settings, "DEP_GUARD_COOLDOWN_SECONDS", 60))
        if c.state == "open":
            if time.monotonic() - c.opened_at >= cooldown:
                c.state = "half_open"
                logger.info("Circuit half_open for %s after cooldown", name)
            else:
                raise DependencyUnavailable(name)
        try:
            yield
        except Exception:
            self.record_failure(name)
            raise
        else:
            self.record_success(name)


_guard: Optional[DependencyGuard] = None


def get_dependency_guard() -> DependencyGuard:
    global _guard
    if _guard is None:
        _guard = DependencyGuard()
    return _guard
