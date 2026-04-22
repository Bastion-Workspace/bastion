"""
User LLM Provider Service - per-user API keys and model lists for OpenAI, OpenRouter, Groq, Ollama, vLLM.
Encrypts API keys with Fernet (same pattern as external_connections_service).
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
from models.api_models import ModelInfo
from models.provider_models import (
    CLOUD_BASE_URLS,
    PROVIDER_TYPES,
    get_base_url_for_provider,
    needs_openrouter_headers,
)
from services.database_manager.database_helpers import (
    execute,
    fetch_all,
    fetch_one,
    fetch_value,
)

logger = logging.getLogger(__name__)

# Ollama documents: api_key can be "ollama" (required but ignored)
OLLAMA_PLACEHOLDER_KEY = "ollama"


class UserLLMProviderService:
    """Service for user-level LLM provider credentials and enabled models."""

    def __init__(self) -> None:
        self._fernet: Optional[Fernet] = None
        self._encryption_initialized = False

    def _initialize_encryption(self) -> None:
        if self._encryption_initialized:
            return
        master_key_str = getattr(settings, "SECRET_KEY", "") or ""
        if not master_key_str:
            logger.warning("SECRET_KEY not set; using temporary key for user LLM providers")
            key = Fernet.generate_key()
            self._fernet = Fernet(key)
        else:
            try:
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b"user_llm_providers_salt",
                    iterations=100000,
                    backend=default_backend(),
                )
                derived = base64.urlsafe_b64encode(kdf.derive(master_key_str.encode()))
                self._fernet = Fernet(derived)
            except Exception as e:
                logger.error("Failed to initialize user LLM provider encryption: %s", e)
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

    def _base_url_for_provider(self, provider_type: str, user_base_url: Optional[str]) -> str:
        return get_base_url_for_provider(provider_type, user_base_url)

    async def _validate_provider_connection(
        self,
        provider_type: str,
        api_key: Optional[str],
        base_url: Optional[str],
    ) -> bool:
        """Call GET .../v1/models to validate credentials. Returns True if OK."""
        url_base = self._base_url_for_provider(provider_type, base_url)
        if not url_base:
            return False
        url = f"{url_base}/v1/models" if not url_base.endswith("/v1") else f"{url_base.rstrip('/')}/models"
        key = api_key or OLLAMA_PLACEHOLDER_KEY
        headers = {"Authorization": f"Bearer {key}"}
        if needs_openrouter_headers(provider_type):
            headers["HTTP-Referer"] = getattr(settings, "SITE_URL", "https://localhost")
            headers["X-Title"] = "Bastion AI Workspace"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, headers=headers)
                if response.status_code == 200:
                    return True
                logger.warning("Provider validation failed: %s %s", response.status_code, response.text[:200])
                return False
        except Exception as e:
            logger.warning("Provider validation error: %s", e)
            return False

    async def add_provider(
        self,
        user_id: str,
        provider_type: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> int:
        """Validate connection, encrypt key, insert. Returns provider id."""
        if provider_type not in PROVIDER_TYPES:
            raise ValueError(f"Invalid provider_type: {provider_type}")
        if provider_type in ("ollama", "vllm") and not base_url:
            raise ValueError("base_url required for ollama and vllm")
        key_to_store = api_key or OLLAMA_PLACEHOLDER_KEY
        if not await self._validate_provider_connection(provider_type, key_to_store, base_url):
            raise ValueError("Provider connection validation failed")
        encrypted = self._encrypt_key(key_to_store)
        display = display_name or provider_type.capitalize()
        provider_id = await fetch_value(
            """
            INSERT INTO user_llm_providers (user_id, provider_type, display_name, encrypted_api_key, base_url, is_active, updated_at)
            VALUES ($1, $2, $3, $4, $5, true, NOW())
            RETURNING id
            """,
            user_id,
            provider_type,
            display,
            encrypted,
            base_url,
        )
        logger.info("Added user LLM provider %s for user %s", provider_type, user_id)
        return provider_id

    async def list_providers(self, user_id: str) -> List[Dict[str, Any]]:
        """List providers for user; never returns decrypted keys."""
        rows = await fetch_all(
            """
            SELECT id, user_id, provider_type, display_name, base_url, is_active, created_at, updated_at
            FROM user_llm_providers
            WHERE user_id = $1
            ORDER BY provider_type, id
            """,
            user_id,
        )
        return [dict(r) for r in rows]

    async def remove_provider(self, user_id: str, provider_id: int) -> bool:
        """Delete provider and cascade to user_enabled_models."""
        await execute(
            "DELETE FROM user_llm_providers WHERE id = $1 AND user_id = $2",
            provider_id,
            user_id,
        )
        logger.info("Removed user LLM provider %s for user %s", provider_id, user_id)
        return True

    async def get_provider(self, user_id: str, provider_id: int) -> Optional[Dict[str, Any]]:
        """Get single provider row (includes encrypted key for internal use)."""
        row = await fetch_one(
            "SELECT id, user_id, provider_type, display_name, encrypted_api_key, base_url, is_active FROM user_llm_providers WHERE id = $1 AND user_id = $2",
            provider_id,
            user_id,
        )
        return dict(row) if row else None

    async def fetch_provider_models(self, user_id: str, provider_id: int) -> List[Dict[str, Any]]:
        """Call provider's /v1/models and return normalized list {id, name, provider}."""
        prov = await self.get_provider(user_id, provider_id)
        if not prov:
            return []
        key = prov["encrypted_api_key"]
        try:
            decrypted = self._decrypt_key(key) if key else OLLAMA_PLACEHOLDER_KEY
        except InvalidToken:
            logger.warning(
                "Provider %s (id=%s): API key could not be decrypted (wrong SECRET_KEY or corrupted). Re-add the provider to fix.",
                prov.get("provider_type"),
                provider_id,
            )
            return []
        url_base = self._base_url_for_provider(prov["provider_type"], prov.get("base_url"))
        url = f"{url_base}/v1/models" if not url_base.endswith("/v1") else f"{url_base.rstrip('/')}/models"
        headers = {"Authorization": f"Bearer {decrypted}"}
        if needs_openrouter_headers(prov["provider_type"]):
            headers["HTTP-Referer"] = getattr(settings, "SITE_URL", "https://localhost")
            headers["X-Title"] = "Bastion AI Workspace"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, headers=headers)
                if response.status_code != 200:
                    logger.warning("fetch_provider_models %s: %s", response.status_code, response.text[:200])
                    return []
                data = response.json()
        except Exception as e:
            logger.warning("fetch_provider_models error: %s", e)
            return []
        raw_list = data.get("data", []) if isinstance(data, dict) else []
        provider_label = prov["display_name"] or prov["provider_type"]
        provider_id_val = prov.get("id")
        out = []
        for m in raw_list:
            mid = None
            if isinstance(m, dict):
                raw_id = m.get("id") or m.get("model")
                mid = str(raw_id).strip() if raw_id is not None else None
            else:
                raw_id = getattr(m, "id", None) or getattr(m, "model", None)
                mid = str(raw_id).strip() if raw_id is not None else None
            name = m.get("name", mid) if isinstance(m, dict) else getattr(m, "name", mid)
            if mid:
                entry = {"id": str(mid), "name": name or str(mid), "provider": provider_label}
                if provider_id_val is not None:
                    entry["provider_id"] = provider_id_val
                if isinstance(m, dict):
                    if m.get("output_modalities") is not None:
                        entry["output_modalities"] = m["output_modalities"] if isinstance(m["output_modalities"], list) else []
                    if m.get("input_modalities") is not None:
                        entry["input_modalities"] = m["input_modalities"] if isinstance(m["input_modalities"], list) else []
                    if prov["provider_type"] == "openrouter":
                        entry["context_length"] = int(m.get("context_length", 0) or 0)
                        pricing = m.get("pricing") or {}
                        if isinstance(pricing, dict):
                            prompt_cost = pricing.get("prompt")
                            completion_cost = pricing.get("completion")
                            try:
                                entry["input_cost"] = float(prompt_cost) if prompt_cost is not None else None
                            except (TypeError, ValueError):
                                entry["input_cost"] = None
                            try:
                                entry["output_cost"] = float(completion_cost) if completion_cost is not None else None
                            except (TypeError, ValueError):
                                entry["output_cost"] = None
                        else:
                            entry["input_cost"] = None
                            entry["output_cost"] = None
                    elif prov["provider_type"] == "groq":
                        entry["context_length"] = int(m.get("context_window", 0) or m.get("context_length", 0) or 0)
                        entry["input_cost"] = None
                        entry["output_cost"] = None
                    elif prov["provider_type"] in ("ollama", "vllm"):
                        entry["input_cost"] = 0.0
                        entry["output_cost"] = 0.0
                        entry["context_length"] = int(m.get("context_length", 0) or 0)
                out.append(entry)
        return out

    async def set_user_enabled_models(self, user_id: str, provider_id: int, model_ids: List[str]) -> None:
        """Replace enabled models for this user and provider. model_ids is list of model id strings."""
        await execute(
            "DELETE FROM user_enabled_models WHERE user_id = $1 AND provider_id = $2",
            user_id,
            provider_id,
        )
        prov = await self.get_provider(user_id, provider_id)
        if not prov or not model_ids:
            return
        provider_label = prov["display_name"] or prov["provider_type"]
        for model_id in model_ids:
            if not model_id:
                continue
            await execute(
                """
                INSERT INTO user_enabled_models (user_id, provider_id, model_id, display_name, is_enabled)
                VALUES ($1, $2, $3, $4, true)
                ON CONFLICT (user_id, provider_id, model_id) DO UPDATE SET is_enabled = true, display_name = EXCLUDED.display_name
                """,
                user_id,
                provider_id,
                model_id,
                f"{model_id} ({provider_label})",
            )

    async def get_user_enabled_models(self, user_id: str) -> List[Dict[str, Any]]:
        """Return list of {model_id, display_name, provider_id} for user.
        model_id is composite key {provider_id}:{model_id} for multi-provider disambiguation."""
        rows = await fetch_all(
            """
            SELECT e.model_id, e.display_name, e.provider_id
            FROM user_enabled_models e
            JOIN user_llm_providers p ON p.id = e.provider_id AND p.user_id = e.user_id AND p.is_active
            WHERE e.user_id = $1 AND e.is_enabled
            ORDER BY e.display_name
            """,
            user_id,
        )
        return [
            {
                "model_id": f"{r['provider_id']}:{r['model_id']}",
                "display_name": r["display_name"] or r["model_id"],
                "provider_id": r["provider_id"],
            }
            for r in rows
        ]

    async def get_available_models_for_user(self, user_id: str) -> List[ModelInfo]:
        """Aggregate available models from all of the user's providers. Used when use_admin_models=false.
        Uses composite ID {provider_id}:{model_id} so the same model from different providers appears as
        distinct entries with provider info and pricing. Sets source='user', provider_type, provider_id."""
        providers = await self.list_providers(user_id)
        out: List[ModelInfo] = []
        for p in providers:
            pid = p.get("id")
            ptype = p.get("provider_type") or "unknown"
            if pid is None:
                continue
            raw = await self.fetch_provider_models(user_id, pid)
            for m in raw:
                mid = m.get("id")
                if not mid:
                    continue
                composite_id = f"{pid}:{mid}"
                out.append(
                    ModelInfo(
                        id=composite_id,
                        name=m.get("name") or mid,
                        provider=m.get("provider") or "Unknown",
                        context_length=int(m.get("context_length", 0) or 0),
                        input_cost=m.get("input_cost"),
                        output_cost=m.get("output_cost"),
                        output_modalities=m.get("output_modalities"),
                        input_modalities=m.get("input_modalities"),
                        source="user",
                        provider_type=ptype,
                        provider_id=pid,
                    )
                )
        out.sort(key=lambda x: (x.provider, x.name))
        return out

    async def get_llm_context_for_model(self, user_id: str, model_id: str) -> Optional[Dict[str, Any]]:
        """Resolve which provider owns this model; return {api_key (decrypted), base_url, real_model_id} for gRPC.
        Supports composite key format provider_id:model_id for multi-provider disambiguation."""
        provider_id_filter: Optional[int] = None
        real_model_id = model_id
        if ":" in model_id:
            parts = model_id.split(":", 1)
            if parts[0].isdigit():
                provider_id_filter = int(parts[0])
                real_model_id = parts[1]
        if provider_id_filter is not None:
            row = await fetch_one(
                """
                SELECT p.encrypted_api_key, p.base_url, p.provider_type
                FROM user_enabled_models e
                JOIN user_llm_providers p ON p.id = e.provider_id AND p.user_id = e.user_id AND p.is_active
                WHERE e.user_id = $1 AND e.provider_id = $2 AND e.model_id = $3 AND e.is_enabled
                LIMIT 1
                """,
                user_id,
                provider_id_filter,
                real_model_id,
            )
        else:
            row = await fetch_one(
                """
                SELECT p.encrypted_api_key, p.base_url, p.provider_type
                FROM user_enabled_models e
                JOIN user_llm_providers p ON p.id = e.provider_id AND p.user_id = e.user_id AND p.is_active
                WHERE e.user_id = $1 AND e.model_id = $2 AND e.is_enabled
                LIMIT 1
                """,
                user_id,
                model_id,
            )
        if not row:
            return None
        base_url = self._base_url_for_provider(row["provider_type"], row.get("base_url"))
        if not base_url:
            return None
        raw_key = row["encrypted_api_key"]
        try:
            api_key = self._decrypt_key(raw_key) if raw_key else OLLAMA_PLACEHOLDER_KEY
        except InvalidToken:
            logger.warning(
                "get_llm_context_for_model: API key for provider could not be decrypted (wrong SECRET_KEY or corrupted)."
            )
            return None
        return {"api_key": api_key, "base_url": base_url, "real_model_id": real_model_id, "provider_type": row["provider_type"]}


user_llm_provider_service = UserLLMProviderService()
