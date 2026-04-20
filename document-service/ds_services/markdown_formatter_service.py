"""Minimal markdown formatter for file_manager (document-service)."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class _StubFormatter:
    async def format_markdown(self, *args: Any, **kwargs: Any) -> str:
        return ""


_stub: Optional[_StubFormatter] = None


async def get_markdown_formatter() -> _StubFormatter:
    global _stub
    if _stub is None:
        _stub = _StubFormatter()
    return _stub
