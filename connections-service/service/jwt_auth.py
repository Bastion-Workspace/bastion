"""
JWT generation for data connector REST auth (type=jwt in definition.auth).

Credentials are supplied decrypted by the backend; this module only signs tokens.
"""

from __future__ import annotations

import base64
import binascii
import logging
import time
from typing import Any, Dict, Optional, Tuple

import jwt
from jwt.exceptions import PyJWTError

logger = logging.getLogger(__name__)

_SUPPORTED_ALGS = frozenset({"HS256", "HS384", "HS512"})


def _decode_secret(secret_raw: str, encoding: str) -> bytes:
    enc = (encoding or "utf8").lower()
    if enc == "hex":
        return bytes.fromhex(secret_raw.strip())
    if enc == "base64":
        return base64.b64decode(secret_raw)
    if enc in ("utf8", "utf-8", "text"):
        return secret_raw.encode("utf-8")
    raise ValueError(f"Unsupported secret_encoding: {encoding}")


def _resolve_key_and_secret(
    auth_config: Dict[str, Any], credentials: Dict[str, Any]
) -> Tuple[Optional[str], str]:
    compound_field = auth_config.get("compound_key_field")
    if compound_field:
        raw = credentials.get(compound_field)
        if raw is None or str(raw).strip() == "":
            raise ValueError(f"Missing credential: {compound_field}")
        sep = auth_config.get("compound_key_separator", ":")
        parts = str(raw).split(sep, 1)
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise ValueError(
                f"compound_key_field {compound_field!r} must contain id and secret separated by {sep!r}"
            )
        return parts[0].strip(), parts[1].strip()

    kid_field = auth_config.get("kid_field")
    secret_field = auth_config.get("secret_field") or "secret"
    secret_raw = credentials.get(secret_field)
    if secret_raw is None or str(secret_raw).strip() == "":
        raise ValueError(f"Missing credential: {secret_field}")
    key_id: Optional[str] = None
    if kid_field:
        kid_val = credentials.get(kid_field)
        if kid_val is not None and str(kid_val).strip() != "":
            key_id = str(kid_val).strip()
    return key_id, str(secret_raw).strip()


def generate_jwt_token(auth_config: Dict[str, Any], credentials: Dict[str, Any]) -> Optional[str]:
    """
    Build a short-lived JWT from connector auth config and decrypted credentials.

    Returns None if signing fails (caller should omit auth or fail the request).
    """
    try:
        algorithm = (auth_config.get("algorithm") or "HS256").upper()
        if algorithm not in _SUPPORTED_ALGS:
            logger.warning("JWT auth: unsupported algorithm %s", algorithm)
            return None

        key_id, secret_raw = _resolve_key_and_secret(auth_config, credentials)
        secret_encoding = auth_config.get("secret_encoding") or "utf8"
        signing_key = _decode_secret(secret_raw, secret_encoding)

        now = int(time.time())
        exp_seconds = int(auth_config.get("exp_seconds") or 300)
        if exp_seconds <= 0:
            exp_seconds = 300

        static_claims = auth_config.get("claims") or {}
        if not isinstance(static_claims, dict):
            static_claims = {}

        payload: Dict[str, Any] = dict(static_claims)
        payload.setdefault("iat", now)
        payload.setdefault("exp", now + exp_seconds)

        extra_headers: Optional[Dict[str, str]] = {"kid": key_id} if key_id else None

        token = jwt.encode(
            payload,
            signing_key,
            algorithm=algorithm,
            headers=extra_headers,
        )
        if isinstance(token, bytes):
            token = token.decode("utf-8")
        return str(token)
    except (ValueError, binascii.Error, PyJWTError) as e:
        logger.warning("JWT auth: token generation failed: %s", e)
        return None
    except Exception as e:
        logger.exception("JWT auth: unexpected error: %s", e)
        return None
