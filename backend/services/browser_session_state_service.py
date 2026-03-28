"""
Browser Session State Service - Encrypted storage of Playwright session state.
Used by granular browser automation tools for persistent sessions (cookies, localStorage).
"""

import base64
import logging
from typing import List, Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from config import settings
from services.database_manager.database_helpers import execute, fetch_all, fetch_one

logger = logging.getLogger(__name__)

BROWSER_SESSION_SALT = b"browser_session_salt"


class BrowserSessionStateService:
    """Encrypt and persist Playwright storage state per user/site."""

    def __init__(self) -> None:
        self._fernet: Optional[Fernet] = None
        self._initialized = False

    def _initialize_encryption(self) -> None:
        if self._initialized:
            return
        master_key_str = getattr(settings, "SECRET_KEY", "") or ""
        if not master_key_str:
            logger.warning("SECRET_KEY not set; using temporary key for browser session states")
            self._fernet = Fernet(Fernet.generate_key())
        else:
            try:
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=BROWSER_SESSION_SALT,
                    iterations=100000,
                    backend=default_backend(),
                )
                derived = base64.urlsafe_b64encode(kdf.derive(master_key_str.encode()))
                self._fernet = Fernet(derived)
            except Exception as e:
                logger.error("Failed to initialize browser session encryption: %s", e)
                self._fernet = Fernet(Fernet.generate_key())
        self._initialized = True

    def _encrypt(self, raw: str) -> str:
        self._initialize_encryption()
        assert self._fernet is not None
        encrypted = self._fernet.encrypt(raw.encode("utf-8"))
        return base64.b64encode(encrypted).decode("utf-8")

    def _decrypt(self, encrypted_b64: str) -> str:
        self._initialize_encryption()
        assert self._fernet is not None
        raw = base64.b64decode(encrypted_b64.encode("utf-8"))
        return self._fernet.decrypt(raw).decode("utf-8")

    async def save_session_state(
        self,
        user_id: str,
        site_domain: str,
        state_json: str,
        rls_context: Optional[dict] = None,
    ) -> bool:
        """Encrypt and upsert session state for user/site. Returns True on success."""
        try:
            encrypted = self._encrypt(state_json)
            await execute(
                """
                INSERT INTO browser_session_states
                    (user_id, site_domain, encrypted_state_blob, is_valid, last_used_at)
                VALUES ($1, $2, $3, true, NOW())
                ON CONFLICT (user_id, site_domain) DO UPDATE SET
                    encrypted_state_blob = EXCLUDED.encrypted_state_blob,
                    captured_at = NOW(),
                    is_valid = true,
                    last_used_at = NOW()
                """,
                user_id,
                site_domain,
                encrypted,
                rls_context=rls_context,
            )
            return True
        except Exception as e:
            logger.error("save_session_state failed: %s", e)
            return False

    async def load_session_state(
        self,
        user_id: str,
        site_domain: str,
        rls_context: Optional[dict] = None,
    ) -> Optional[str]:
        """Load and decrypt session state for user/site. Returns JSON string or None."""
        row = await fetch_one(
            """
            SELECT encrypted_state_blob
            FROM browser_session_states
            WHERE user_id = $1 AND site_domain = $2 AND is_valid = true
            """,
            user_id,
            site_domain,
            rls_context=rls_context,
        )
        if not row:
            return None
        try:
            return self._decrypt(row["encrypted_state_blob"])
        except Exception as e:
            logger.warning("Decrypt browser session state failed: %s", e)
            return None

    async def invalidate_session_state(
        self,
        user_id: str,
        site_domain: str,
        rls_context: Optional[dict] = None,
    ) -> bool:
        """Mark session as invalid. Returns True on success."""
        try:
            await execute(
                """
                UPDATE browser_session_states
                SET is_valid = false, last_used_at = NOW()
                WHERE user_id = $1 AND site_domain = $2
                """,
                user_id,
                site_domain,
                rls_context=rls_context,
            )
            return True
        except Exception as e:
            logger.error("invalidate_session_state failed: %s", e)
            return False

    async def touch_session(
        self,
        user_id: str,
        site_domain: str,
        rls_context: Optional[dict] = None,
    ) -> None:
        """Update last_used_at for the session."""
        try:
            await execute(
                """
                UPDATE browser_session_states
                SET last_used_at = NOW()
                WHERE user_id = $1 AND site_domain = $2 AND is_valid = true
                """,
                user_id,
                site_domain,
                rls_context=rls_context,
            )
        except Exception as e:
            logger.warning("touch_session failed: %s", e)

    async def list_sessions(
        self,
        user_id: str,
        rls_context: Optional[dict] = None,
    ) -> List[dict]:
        """List saved browser sessions for the user (site_domain, last_used_at, is_valid)."""
        try:
            rows = await fetch_all(
                """
                SELECT site_domain, last_used_at, is_valid
                FROM browser_session_states
                WHERE user_id = $1
                ORDER BY last_used_at DESC
                """,
                user_id,
                rls_context=rls_context,
            )
            return [
                {
                    "site_domain": r["site_domain"],
                    "last_used_at": r["last_used_at"].isoformat() if hasattr(r["last_used_at"], "isoformat") else str(r["last_used_at"]),
                    "is_valid": bool(r["is_valid"]),
                }
                for r in (rows or [])
            ]
        except Exception as e:
            logger.error("list_sessions failed: %s", e)
            return []


_browser_session_state_service: Optional[BrowserSessionStateService] = None


def get_browser_session_state_service() -> BrowserSessionStateService:
    """Singleton accessor for BrowserSessionStateService."""
    global _browser_session_state_service
    if _browser_session_state_service is None:
        _browser_session_state_service = BrowserSessionStateService()
    return _browser_session_state_service
