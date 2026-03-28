"""
Sanitize secrets in log records (Telegram Bot API URLs embed the token in the path).
"""

from __future__ import annotations

import logging
import re
from typing import Any

# https://api.telegram.org/bot<token>/method and .../file/bot<token>/...
_TELEGRAM_BOT_IN_PATH = re.compile(
    r"(https?://api\.telegram\.org/(?:file/)?bot)([^/\s\"]+)(/)"
)


def redact_telegram_secrets(text: str) -> str:
    if not text or "api.telegram.org" not in text:
        return text
    return _TELEGRAM_BOT_IN_PATH.sub(r"\1***REDACTED***\3", text)


def _redact_arg(value: Any) -> Any:
    if isinstance(value, str):
        return redact_telegram_secrets(value)
    return value


class RedactTelegramSecretsFilter(logging.Filter):
    """Strip Telegram bot tokens from log message text (httpx logs full request URLs at INFO)."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if record.args:
                record.args = tuple(_redact_arg(a) for a in record.args)
            if isinstance(record.msg, str):
                record.msg = redact_telegram_secrets(record.msg)
        except Exception:
            return True
        return True
