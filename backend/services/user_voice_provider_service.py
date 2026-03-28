"""
Per-user TTS/STT API keys (BYOK). Encrypted storage; distinct salt from user LLM providers.
"""

import base64
import logging
from typing import Any, Dict, List, Optional

import httpx
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from config import settings
from services.database_manager.database_helpers import execute, fetch_all, fetch_one, fetch_value

logger = logging.getLogger(__name__)

VOICE_PROVIDER_TYPES_TTS = frozenset({"elevenlabs", "openai"})
VOICE_PROVIDER_TYPES_STT = frozenset({"openai", "deepgram", "whisper_api"})
VOICE_PROVIDER_TYPES = VOICE_PROVIDER_TYPES_TTS | VOICE_PROVIDER_TYPES_STT

SETTING_USE_ADMIN_TTS = "use_admin_tts"
SETTING_USE_ADMIN_STT = "use_admin_stt"
SETTING_USER_TTS_PROVIDER_ID = "user_tts_provider_id"
SETTING_USER_TTS_VOICE_ID = "user_tts_voice_id"
SETTING_USER_STT_PROVIDER_ID = "user_stt_provider_id"
SETTING_USER_ADMIN_TTS_VOICE_ID = "user_admin_tts_voice_id"
SETTING_USER_ADMIN_TTS_VOICE_SERVER = "user_admin_tts_voice_server"
SETTING_USER_ADMIN_TTS_VOICE_PIPER = "user_admin_tts_voice_piper"
SETTING_USER_ADMIN_TTS_PROVIDER = "user_admin_tts_provider"
# BYOK TTS: cloud (linked API row) | piper (server local) | browser (client only)
SETTING_USER_BYOK_TTS_ENGINE = "user_byok_tts_engine"
SETTING_USER_BYOK_TTS_VOICE_PIPER = "user_byok_tts_voice_piper"
SETTING_USER_ELEVENLABS_TTS_MODEL_ID = "user_elevenlabs_tts_model_id"
SETTING_USER_ADMIN_ELEVENLABS_TTS_MODEL_ID = "user_admin_elevenlabs_tts_model_id"


def _truthy_setting(raw: Optional[str]) -> bool:
    if raw is None:
        return True
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


class UserVoiceProviderService:
    """Encrypted per-user voice provider credentials and preferences."""

    def __init__(self) -> None:
        self._fernet: Optional[Fernet] = None
        self._encryption_initialized = False

    def _initialize_encryption(self) -> None:
        if self._encryption_initialized:
            return
        master_key_str = getattr(settings, "SECRET_KEY", "") or ""
        if not master_key_str:
            logger.warning("SECRET_KEY not set; using temporary key for user voice providers")
            key = Fernet.generate_key()
            self._fernet = Fernet(key)
        else:
            try:
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b"user_voice_providers_salt",
                    iterations=100000,
                    backend=default_backend(),
                )
                derived = base64.urlsafe_b64encode(kdf.derive(master_key_str.encode()))
                self._fernet = Fernet(derived)
            except Exception as e:
                logger.error("Failed to initialize user voice provider encryption: %s", e)
                key = Fernet.generate_key()
                self._fernet = Fernet(key)
        self._encryption_initialized = True

    def _encrypt_key(self, raw: str) -> str:
        self._initialize_encryption()
        assert self._fernet is not None
        encrypted = self._fernet.encrypt(raw.encode("utf-8"))
        return base64.b64encode(encrypted).decode("utf-8")

    def _decrypt_key(self, encrypted: str) -> str:
        self._initialize_encryption()
        assert self._fernet is not None
        raw = base64.b64decode(encrypted.encode("utf-8"))
        return self._fernet.decrypt(raw).decode("utf-8")

    async def _validate_credentials(
        self,
        provider_type: str,
        provider_role: str,
        api_key: str,
        base_url: Optional[str],
    ) -> bool:
        key = (api_key or "").strip()
        if not key:
            return False
        bu = (base_url or "").strip().rstrip("/")

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                if provider_type == "elevenlabs" and provider_role == "tts":
                    r = await client.get(
                        "https://api.elevenlabs.io/v1/voices",
                        headers={"xi-api-key": key},
                    )
                    return r.status_code == 200

                if provider_type in ("openai", "whisper_api"):
                    root = bu or "https://api.openai.com/v1"
                    if not root.endswith("/v1"):
                        root = f"{root.rstrip('/')}/v1"
                    url = f"{root}/models"
                    r = await client.get(
                        url,
                        headers={"Authorization": f"Bearer {key}"},
                    )
                    return r.status_code == 200

                if provider_type == "deepgram" and provider_role == "stt":
                    r = await client.get(
                        "https://api.deepgram.com/v1/projects",
                        headers={"Authorization": f"Token {key}"},
                    )
                    return r.status_code == 200
        except Exception as e:
            logger.warning("Voice provider validation error: %s", e)
            return False

        return False

    async def add_provider(
        self,
        user_id: str,
        provider_type: str,
        provider_role: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> int:
        if provider_type not in VOICE_PROVIDER_TYPES:
            raise ValueError(f"Invalid provider_type: {provider_type}")
        if provider_role not in ("tts", "stt"):
            raise ValueError("provider_role must be tts or stt")
        if provider_role == "tts" and provider_type not in VOICE_PROVIDER_TYPES_TTS:
            raise ValueError(f"Provider {provider_type} is not valid for TTS")
        if provider_role == "stt" and provider_type not in VOICE_PROVIDER_TYPES_STT:
            raise ValueError(f"Provider {provider_type} is not valid for STT")

        key_to_store = (api_key or "").strip()
        if not await self._validate_credentials(
            provider_type, provider_role, key_to_store, base_url
        ):
            raise ValueError("Provider connection validation failed")

        encrypted = self._encrypt_key(key_to_store)
        display = display_name or provider_type.capitalize()
        provider_id = await fetch_value(
            """
            INSERT INTO user_voice_providers (
                user_id, provider_type, provider_role, display_name,
                encrypted_api_key, base_url, is_active, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, true, NOW())
            RETURNING id
            """,
            user_id,
            provider_type,
            provider_role,
            display,
            encrypted,
            (base_url or "").strip() or None,
        )
        return int(provider_id)

    async def remove_provider(self, user_id: str, provider_id: int) -> None:
        await execute(
            "DELETE FROM user_voice_providers WHERE id = $1 AND user_id = $2",
            provider_id,
            user_id,
        )

    async def list_providers(self, user_id: str) -> List[Dict[str, Any]]:
        rows = await fetch_all(
            """
            SELECT id, provider_type, provider_role, display_name, base_url, is_active,
                   created_at, updated_at
            FROM user_voice_providers
            WHERE user_id = $1
            ORDER BY provider_role, provider_type, id
            """,
            user_id,
        )
        return [dict(r) for r in rows]

    async def get_provider_row(
        self, user_id: str, provider_id: int
    ) -> Optional[Dict[str, Any]]:
        row = await fetch_one(
            """
            SELECT id, user_id, provider_type, provider_role, display_name,
                   encrypted_api_key, base_url, is_active
            FROM user_voice_providers
            WHERE id = $1 AND user_id = $2
            """,
            provider_id,
            user_id,
        )
        return dict(row) if row else None

    def _voice_service_provider_name(self, provider_type: str) -> str:
        if provider_type == "elevenlabs":
            return "elevenlabs"
        if provider_type == "openai":
            return "openai"
        return provider_type

    async def get_linked_tts_provider_type_name(self, user_id: str) -> str:
        """Voice-service provider id (e.g. elevenlabs) for the user's linked BYOK TTS row."""
        pid_raw = await self._get_setting(user_id, SETTING_USER_TTS_PROVIDER_ID)
        if not pid_raw or not str(pid_raw).strip().isdigit():
            return ""
        row = await self.get_provider_row(user_id, int(str(pid_raw).strip()))
        if not row or row.get("provider_role") != "tts":
            return ""
        return self._voice_service_provider_name(row.get("provider_type") or "")

    async def get_voice_context(
        self, user_id: str, role: str
    ) -> Optional[Dict[str, Any]]:
        """Return credentials for TTS or STT when user uses own provider; else None."""
        if role == "tts":
            if _truthy_setting(await self._get_setting(user_id, SETTING_USE_ADMIN_TTS)):
                return None
            pid_raw = await self._get_setting(user_id, SETTING_USER_TTS_PROVIDER_ID)
            if not pid_raw or not str(pid_raw).strip().isdigit():
                return None
            row = await self.get_provider_row(user_id, int(pid_raw))
            if not row or row.get("provider_role") != "tts":
                return None
            try:
                api_key = self._decrypt_key(row["encrypted_api_key"] or "")
            except InvalidToken:
                logger.warning("User voice provider key decrypt failed for user %s", user_id)
                return None
            voice_id = (await self._get_setting(user_id, SETTING_USER_TTS_VOICE_ID)) or ""
            return {
                "provider_type": self._voice_service_provider_name(row["provider_type"]),
                "api_key": api_key,
                "base_url": (row.get("base_url") or "").strip(),
                "voice_id": (voice_id or "").strip(),
            }

        if role == "stt":
            if _truthy_setting(await self._get_setting(user_id, SETTING_USE_ADMIN_STT)):
                return None
            pid_raw = await self._get_setting(user_id, SETTING_USER_STT_PROVIDER_ID)
            if not pid_raw or not str(pid_raw).strip().isdigit():
                return None
            row = await self.get_provider_row(user_id, int(pid_raw))
            if not row or row.get("provider_role") != "stt":
                return None
            try:
                api_key = self._decrypt_key(row["encrypted_api_key"] or "")
            except InvalidToken:
                logger.warning("User STT provider key decrypt failed for user %s", user_id)
                return None
            return {
                "provider_type": row["provider_type"],
                "api_key": api_key,
                "base_url": (row.get("base_url") or "").strip(),
            }

        raise ValueError("role must be tts or stt")

    async def _get_setting(self, user_id: str, key: str) -> Optional[str]:
        row = await fetch_one(
            "SELECT value FROM user_settings WHERE user_id = $1 AND key = $2",
            user_id,
            key,
        )
        return row["value"] if row else None

    async def list_provider_voices(
        self, user_id: str, provider_id: int
    ) -> List[Dict[str, Any]]:
        row = await self.get_provider_row(user_id, provider_id)
        if not row or row.get("provider_role") != "tts":
            raise ValueError("TTS provider not found")
        try:
            api_key = self._decrypt_key(row["encrypted_api_key"] or "")
        except InvalidToken as e:
            raise ValueError("Stored API key could not be decrypted") from e
        prov = self._voice_service_provider_name(row["provider_type"])
        from clients.voice_service_client import get_voice_service_client

        client = await get_voice_service_client()
        result = await client.list_voices(
            provider=prov,
            api_key=api_key,
            base_url=(row.get("base_url") or "").strip(),
        )
        if result.get("error"):
            raise RuntimeError(result["error"])
        return result.get("voices", [])


user_voice_provider_service = UserVoiceProviderService()
