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

from ds_config import settings

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
    """Resolve path for .md / .txt / .org (same rules as document content API)."""
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
    file_path = Path(file_path_str)
    if file_path.exists():
        return file_path

    upload_dir = Path(settings.UPLOAD_DIR)
    doc_id = getattr(doc_info, "document_id", "")
    legacy_paths = [
        upload_dir / f"{doc_id}_{filename}",
        upload_dir / filename,
    ]
    if filename.lower().endswith(".md"):
        import glob

        legacy_paths.extend(
            [
                upload_dir / "web_sources" / "rss_articles" / "*" / filename,
                upload_dir / "web_sources" / "scraped_content" / "*" / filename,
            ]
        )
    import glob as glob_mod

    for path_pattern in legacy_paths:
        matches = (
            glob_mod.glob(str(path_pattern))
            if "*" in str(path_pattern)
            else [str(path_pattern)]
        )
        if matches:
            candidate = Path(matches[0])
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"File not found for document {doc_id}")


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

    file_path = await resolve_editable_file_path(doc, folder_service)
    plaintext = file_path.read_text(encoding="utf-8")
    salt = os.urandom(SALT_LEN)
    key = derive_encryption_key(password, salt)
    blob = encrypt_plaintext_bytes(plaintext.encode("utf-8"), key)
    phash = _password_hasher.hash(password)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(blob)

    from ds_db.database_manager.database_helpers import execute

    await execute(
        """
        UPDATE document_metadata
        SET is_encrypted = TRUE,
            encryption_version = $2,
            encryption_salt = $3,
            password_hash = $4,
            exempt_from_vectorization = TRUE,
            file_size = $5,
            chunk_count = 0,
            entity_count = 0,
            updated_at = CURRENT_TIMESTAMP
        WHERE document_id = $1
        """,
        document_id,
        ENCRYPTION_VERSION,
        salt,
        phash,
        len(blob),
        rls_context={"user_id": "", "user_role": "admin"},
    )

    owner_id = getattr(doc, "user_id", None)
    await purge_search_indexes(document_service, document_id, owner_id)
    from ds_models.api_models import ProcessingStatus

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

    from ds_db.database_manager.database_helpers import fetch_one

    row = await fetch_one(
        """
        SELECT encryption_salt, password_hash
        FROM document_metadata
        WHERE document_id = $1
        """,
        document_id,
        rls_context={"user_id": "", "user_role": "admin"},
    )
    if not row or not row.get("encryption_salt") or not row.get("password_hash"):
        raise ValueError("Encryption metadata missing")

    try:
        _password_hasher.verify(row["password_hash"], password)
    except argon2.exceptions.VerifyMismatchError:
        n = await _record_failed_attempt(document_id, user_id)
        raise ValueError("Incorrect password") from None

    await _clear_attempts(document_id, user_id)

    salt = bytes(row["encryption_salt"])
    key = derive_encryption_key(password, salt)
    file_path = await resolve_editable_file_path(doc, folder_service)
    blob = file_path.read_bytes()
    plaintext = decrypt_ciphertext_bytes(blob, key).decode("utf-8")

    session_token = secrets.token_urlsafe(32)
    await _save_session(document_id, user_id, session_token, key.hex())
    return plaintext, session_token


async def try_decrypt_content_with_session(
    document_service: Any,
    folder_service: Any,
    document_id: str,
    user_id: str,
    session_token: str,
) -> Optional[str]:
    key = await validate_session_token(document_id, user_id, session_token)
    if not key:
        return None
    doc = await document_service.get_document(document_id)
    if not doc or not getattr(doc, "is_encrypted", False):
        return None
    file_path = await resolve_editable_file_path(doc, folder_service)
    blob = file_path.read_bytes()
    try:
        return decrypt_ciphertext_bytes(blob, key).decode("utf-8")
    except Exception:
        logger.warning("Decrypt failed for document %s", document_id)
        return None


async def write_encrypted_content_from_session(
    document_service: Any,
    folder_service: Any,
    document_id: str,
    user_id: str,
    session_token: str,
    new_plaintext: str,
) -> None:
    key = await validate_session_token(document_id, user_id, session_token)
    if not key:
        raise PermissionError("No active encryption session")
    doc = await document_service.get_document(document_id)
    if not doc or not getattr(doc, "is_encrypted", False):
        raise ValueError("Document is not encrypted")
    file_path = await resolve_editable_file_path(doc, folder_service)
    blob = encrypt_plaintext_bytes(new_plaintext.encode("utf-8"), key)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(blob)
    from ds_db.database_manager.database_helpers import execute

    await execute(
        """
        UPDATE document_metadata
        SET file_size = $2, updated_at = CURRENT_TIMESTAMP
        WHERE document_id = $1
        """,
        document_id,
        len(blob),
        rls_context={"user_id": "", "user_role": "admin"},
    )
    data = await _get_session_payload(document_id, user_id)
    if data:
        await _save_session(
            document_id, user_id, data["session_token"], data["derived_key_hex"]
        )


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

    from ds_db.database_manager.database_helpers import fetch_one

    row = await fetch_one(
        """
        SELECT encryption_salt, password_hash
        FROM document_metadata
        WHERE document_id = $1
        """,
        document_id,
        rls_context={"user_id": "", "user_role": "admin"},
    )
    if not row:
        raise ValueError("Document not found")
    try:
        _password_hasher.verify(row["password_hash"], old_password)
    except argon2.exceptions.VerifyMismatchError:
        raise ValueError("Incorrect current password") from None

    salt = os.urandom(SALT_LEN)
    key = derive_encryption_key(new_password, salt)
    file_path = await resolve_editable_file_path(doc, folder_service)
    old_key = derive_encryption_key(old_password, bytes(row["encryption_salt"]))
    blob = file_path.read_bytes()
    plaintext = decrypt_ciphertext_bytes(blob, old_key)
    new_blob = encrypt_plaintext_bytes(plaintext, key)
    with open(file_path, "wb") as f:
        f.write(new_blob)
    phash = _password_hasher.hash(new_password)
    from ds_db.database_manager.database_helpers import execute

    await execute(
        """
        UPDATE document_metadata
        SET encryption_salt = $2,
            password_hash = $3,
            file_size = $4,
            updated_at = CURRENT_TIMESTAMP
        WHERE document_id = $1
        """,
        document_id,
        salt,
        phash,
        len(new_blob),
        rls_context={"user_id": "", "user_role": "admin"},
    )
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

    from ds_db.database_manager.database_helpers import fetch_one, execute

    row = await fetch_one(
        """
        SELECT encryption_salt, password_hash
        FROM document_metadata
        WHERE document_id = $1
        """,
        document_id,
        rls_context={"user_id": "", "user_role": "admin"},
    )
    if not row:
        raise ValueError("Document not found")
    try:
        _password_hasher.verify(row["password_hash"], password)
    except argon2.exceptions.VerifyMismatchError:
        raise ValueError("Incorrect password") from None

    salt = bytes(row["encryption_salt"])
    key = derive_encryption_key(password, salt)
    file_path = await resolve_editable_file_path(doc, folder_service)
    blob = file_path.read_bytes()
    plaintext = decrypt_ciphertext_bytes(blob, key).decode("utf-8")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(plaintext)

    size = len(plaintext.encode("utf-8"))
    await execute(
        """
        UPDATE document_metadata
        SET is_encrypted = FALSE,
            encryption_version = NULL,
            encryption_salt = NULL,
            password_hash = NULL,
            exempt_from_vectorization = FALSE,
            file_size = $2,
            updated_at = CURRENT_TIMESTAMP
        WHERE document_id = $1
        """,
        document_id,
        size,
        rls_context={"user_id": "", "user_role": "admin"},
    )
    await lock_document(document_id, user_id)

    from ds_models.api_models import ProcessingStatus

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
        import asyncio

        from ds_services.collab_reprocess_helper import schedule_reprocess_after_save

        asyncio.create_task(schedule_reprocess_after_save(document_id, user_id))
    except Exception as e:
        logger.warning("Failed to queue reprocess after decrypt for %s: %s", document_id, e)
