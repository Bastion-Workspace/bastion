"""
Per-document password encryption: Argon2id KDF, AES-256-GCM at rest,
Redis-backed unlock sessions with TTL.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import argon2
from argon2.low_level import Type, hash_secret_raw
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from config import settings

logger = logging.getLogger(__name__)

NONCE_LEN = 12
SALT_LEN = 16
ENCRYPTION_VERSION = 1

_password_hasher = argon2.PasswordHasher(
    time_cost=3,
    memory_cost=65536,
    parallelism=4,
    type=argon2.Type.ID,
)

_redis_client = None


async def _get_redis():
    global _redis_client
    if _redis_client is None and settings.REDIS_URL:
        import redis.asyncio as redis

        _redis_client = redis.from_url(settings.REDIS_URL)
        await _redis_client.ping()
    return _redis_client


def _session_key(document_id: str, user_id: str) -> str:
    return f"enc_session:{document_id}:{user_id}"


def _attempts_key(document_id: str, user_id: str) -> str:
    return f"enc_attempts:{document_id}:{user_id}"


def derive_encryption_key(password: str, salt: bytes) -> bytes:
    return hash_secret_raw(
        secret=password.encode("utf-8"),
        salt=salt,
        time_cost=3,
        memory_cost=65536,
        parallelism=4,
        hash_len=32,
        type=Type.ID,
    )


def encrypt_plaintext_bytes(plaintext: bytes, key: bytes) -> bytes:
    nonce = os.urandom(NONCE_LEN)
    aes = AESGCM(key)
    ciphertext = aes.encrypt(nonce, plaintext, None)
    return nonce + ciphertext


def decrypt_ciphertext_bytes(blob: bytes, key: bytes) -> bytes:
    if len(blob) < NONCE_LEN + 16:
        raise ValueError("Invalid ciphertext length")
    nonce = blob[:NONCE_LEN]
    ct = blob[NONCE_LEN:]
    aes = AESGCM(key)
    return aes.decrypt(nonce, ct, None)


async def _require_redis():
    r = await _get_redis()
    if r is None:
        raise RuntimeError("Redis is required for document encryption sessions")
    return r


async def _get_session_payload(document_id: str, user_id: str) -> Optional[Dict[str, str]]:
    r = await _get_redis()
    if not r:
        return None
    raw = await r.get(_session_key(document_id, user_id))
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


async def _save_session(
    document_id: str,
    user_id: str,
    session_token: str,
    derived_key_hex: str,
) -> None:
    r = await _require_redis()
    payload = json.dumps(
        {"session_token": session_token, "derived_key_hex": derived_key_hex}
    )
    await r.setex(
        _session_key(document_id, user_id),
        settings.FILE_ENCRYPTION_SESSION_TTL_SECONDS,
        payload,
    )


async def validate_session_token(
    document_id: str, user_id: str, session_token: str
) -> Optional[bytes]:
    if not session_token:
        return None
    data = await _get_session_payload(document_id, user_id)
    if not data:
        return None
    stored = data.get("session_token") or ""
    if not secrets.compare_digest(stored, session_token):
        return None
    hex_key = data.get("derived_key_hex")
    if not hex_key or len(hex_key) != 64:
        return None
    try:
        return bytes.fromhex(hex_key)
    except ValueError:
        return None


async def heartbeat_session(document_id: str, user_id: str, session_token: str) -> Optional[int]:
    key_bytes = await validate_session_token(document_id, user_id, session_token)
    if not key_bytes:
        return None
    r = await _require_redis()
    data = await _get_session_payload(document_id, user_id)
    if not data:
        return None
    payload = json.dumps(data)
    await r.setex(
        _session_key(document_id, user_id),
        settings.FILE_ENCRYPTION_SESSION_TTL_SECONDS,
        payload,
    )
    return settings.FILE_ENCRYPTION_SESSION_TTL_SECONDS


async def lock_document(document_id: str, user_id: str) -> None:
    r = await _get_redis()
    if not r:
        return
    await r.delete(_session_key(document_id, user_id))


async def _record_failed_attempt(document_id: str, user_id: str) -> int:
    r = await _require_redis()
    k = _attempts_key(document_id, user_id)
    n = await r.incr(k)
    if n == 1:
        await r.expire(k, settings.FILE_ENCRYPTION_LOCKOUT_SECONDS)
    return int(n)


async def _clear_attempts(document_id: str, user_id: str) -> None:
    r = await _get_redis()
    if r:
        await r.delete(_attempts_key(document_id, user_id))


async def _is_rate_limited(document_id: str, user_id: str) -> bool:
    r = await _get_redis()
    if not r:
        return False
    raw = await r.get(_attempts_key(document_id, user_id))
    if not raw:
        return False
    try:
        return int(raw) >= settings.FILE_ENCRYPTION_MAX_ATTEMPTS
    except (TypeError, ValueError):
        return False


async def resolve_editable_file_path(doc_info: Any, folder_service: Any) -> Path:
    """Resolve logical library path for .md / .txt / .org (bytes live on document-service)."""
    filename = getattr(doc_info, "filename", None) or ""
    user_id = getattr(doc_info, "user_id", None)
    folder_id = getattr(doc_info, "folder_id", None)
    collection_type = getattr(doc_info, "collection_type", "user")
    team_id = getattr(doc_info, "team_id", None)
    file_path_str = await folder_service.get_document_file_path(
        filename=filename,
        folder_id=folder_id,
        user_id=user_id,
        collection_type=collection_type,
        team_id=team_id,
    )
    if not file_path_str:
        raise FileNotFoundError(
            f"No path for document {getattr(doc_info, 'document_id', '')}"
        )
    return Path(file_path_str)


async def purge_search_indexes(document_service: Any, document_id: str, user_id: Optional[str]) -> None:
    try:
        await document_service.embedding_manager.delete_document_chunks(
            document_id, user_id
        )
    except Exception as e:
        logger.warning("Failed to delete vector chunks for %s: %s", document_id, e)
    if getattr(document_service, "kg_service", None):
        try:
            await document_service.kg_service.delete_document_entities(document_id)
        except Exception as e:
            logger.warning("Failed to delete KG entities for %s: %s", document_id, e)


def _editable_extensions(filename: str) -> bool:
    fn = (filename or "").lower()
    return fn.endswith((".txt", ".md", ".org"))


async def encrypt_document(
    document_service: Any,
    folder_service: Any,
    document_id: str,
    password: str,
    confirm_password: str,
    user_id: str,
) -> None:
    if password != confirm_password:
        raise ValueError("Passwords do not match")
    if len(password) < settings.PASSWORD_MIN_LENGTH:
        raise ValueError(
            f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters"
        )
    await _require_redis()

    doc = await document_service.get_document(document_id)
    if not doc:
        raise ValueError("Document not found")
    if getattr(doc, "is_encrypted", False):
        raise ValueError("Document is already encrypted")
    filename = getattr(doc, "filename", "") or ""
    if not _editable_extensions(filename):
        raise ValueError("Only .txt, .md, and .org documents can be encrypted")

    from clients.document_service_client import get_document_service_client

    dsc = get_document_service_client()
    await dsc.initialize(required=True)
    ok, _data, err = await dsc.encrypt_document_json(
        user_id,
        {
            "document_id": document_id,
            "password": password,
            "confirm_password": confirm_password,
            "user_id": user_id,
        },
    )
    if not ok:
        raise ValueError(err or "Encryption failed")

    owner_id = getattr(doc, "user_id", None)
    await purge_search_indexes(document_service, document_id, owner_id)
    from models.api_models import ProcessingStatus

    await document_service.document_repository.update_status(
        document_id, ProcessingStatus.COMPLETED
    )


async def create_decrypt_session(
    document_service: Any,
    folder_service: Any,
    document_id: str,
    password: str,
    user_id: str,
) -> Tuple[str, str]:
    if await _is_rate_limited(document_id, user_id):
        raise PermissionError("Too many failed attempts. Try again later.")

    doc = await document_service.get_document(document_id)
    if not doc or not getattr(doc, "is_encrypted", False):
        raise ValueError("Document is not encrypted")

    from clients.document_service_client import get_document_service_client

    dsc = get_document_service_client()
    await dsc.initialize(required=True)
    ok, data, err = await dsc.create_decrypt_session_json(
        user_id,
        {
            "document_id": document_id,
            "password": password,
            "user_id": user_id,
        },
    )
    if not ok or not data:
        await _record_failed_attempt(document_id, user_id)
        raise ValueError(err or "Unlock failed")
    await _clear_attempts(document_id, user_id)
    plaintext = data.get("content") or ""
    session_token = data.get("session_token") or ""
    if not session_token:
        raise ValueError("Invalid unlock response from document-service")
    return plaintext, session_token


async def try_decrypt_content_with_session(
    document_service: Any,
    folder_service: Any,
    document_id: str,
    user_id: str,
    session_token: str,
) -> Optional[str]:
    doc = await document_service.get_document(document_id)
    if not doc or not getattr(doc, "is_encrypted", False):
        return None
    from clients.document_service_client import get_document_service_client

    dsc = get_document_service_client()
    await dsc.initialize(required=True)
    ok, dec_data, _err = await dsc.try_decrypt_json(
        user_id,
        {
            "document_id": document_id,
            "session_token": session_token,
            "user_id": user_id,
        },
    )
    if not ok or not dec_data:
        return None
    return dec_data.get("content")


async def write_encrypted_content_from_session(
    document_service: Any,
    folder_service: Any,
    document_id: str,
    user_id: str,
    session_token: str,
    new_plaintext: str,
) -> None:
    doc = await document_service.get_document(document_id)
    if not doc or not getattr(doc, "is_encrypted", False):
        raise ValueError("Document is not encrypted")
    from clients.document_service_client import get_document_service_client

    dsc = get_document_service_client()
    await dsc.initialize(required=True)
    ok, _data, err = await dsc.write_encrypted_content_from_session_json(
        user_id,
        {
            "document_id": document_id,
            "user_id": user_id,
            "session_token": session_token,
            "content": new_plaintext,
        },
    )
    if not ok:
        raise PermissionError(err or "No active encryption session or save failed")


async def change_encryption_password(
    document_service: Any,
    folder_service: Any,
    document_id: str,
    old_password: str,
    new_password: str,
    user_id: str,
) -> None:
    if len(new_password) < settings.PASSWORD_MIN_LENGTH:
        raise ValueError(
            f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters"
        )
    doc = await document_service.get_document(document_id)
    if not doc or not getattr(doc, "is_encrypted", False):
        raise ValueError("Document is not encrypted")

    from clients.document_service_client import get_document_service_client

    dsc = get_document_service_client()
    await dsc.initialize(required=True)
    ok, _data, err = await dsc.change_encryption_password_json(
        user_id,
        {
            "document_id": document_id,
            "old_password": old_password,
            "new_password": new_password,
            "user_id": user_id,
        },
    )
    if not ok:
        raise ValueError(err or "Failed to change password")
    await lock_document(document_id, user_id)


async def remove_encryption(
    document_service: Any,
    folder_service: Any,
    document_id: str,
    password: str,
    user_id: str,
) -> None:
    doc = await document_service.get_document(document_id)
    if not doc or not getattr(doc, "is_encrypted", False):
        raise ValueError("Document is not encrypted")

    from clients.document_service_client import get_document_service_client

    dsc = get_document_service_client()
    await dsc.initialize(required=True)
    ok, _data, err = await dsc.remove_encryption_json(
        user_id,
        {
            "document_id": document_id,
            "password": password,
            "user_id": user_id,
        },
    )
    if not ok:
        raise ValueError(err or "Failed to remove encryption")
    await lock_document(document_id, user_id)

    from models.api_models import ProcessingStatus

    owner_id = getattr(doc, "user_id", None)
    try:
        await document_service.embedding_manager.delete_document_chunks(
            document_id, owner_id
        )
    except Exception as e:
        logger.warning("Chunk cleanup after decrypt failed for %s: %s", document_id, e)

    await document_service.document_repository.update_status(
        document_id, ProcessingStatus.EMBEDDING, user_id
    )
    await document_service._emit_document_status_update(
        document_id, ProcessingStatus.EMBEDDING.value, user_id
    )
    try:
        from services.celery_tasks.document_tasks import (
            reprocess_document_after_save_task,
        )

        reprocess_document_after_save_task.apply_async(
            args=[document_id, user_id], queue="default"
        )
    except Exception as e:
        logger.warning("Failed to queue reprocess after decrypt for %s: %s", document_id, e)
