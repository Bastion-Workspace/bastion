"""
Ed25519 signing and verification for Bastion-to-Bastion federation (BFP).
Uses PyNaCl (libsodium) detached signatures over raw message bytes.
"""

from __future__ import annotations

import base64
from typing import Tuple

import nacl.signing
from nacl.exceptions import BadSignatureError


def _b64_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def generate_keypair() -> Tuple[str, str]:
    """
    Generate a new Ed25519 signing keypair.

    Returns:
        (private_key_b64, public_key_b64) — URL-safe base64 without padding,
        suitable for JSON and HTTP headers. Private key is the 32-byte seed.
    """
    signing_key = nacl.signing.SigningKey.generate()
    seed = bytes(signing_key)  # 32-byte seed
    verify_key = signing_key.verify_key.encode()
    return _b64_encode(seed), _b64_encode(verify_key)


def get_public_key_from_private(private_key_b64: str) -> str:
    """Derive the public verify key (base64) from a stored private key seed (base64)."""
    seed = _b64_decode(private_key_b64.strip())
    signing_key = nacl.signing.SigningKey(seed)
    return _b64_encode(signing_key.verify_key.encode())


def sign_payload(private_key_b64: str, payload: bytes) -> str:
    """
    Sign payload bytes with Ed25519 (detached signature).

    Returns:
        URL-safe base64 signature (no padding), for X-Bastion-Signature header.
    """
    seed = _b64_decode(private_key_b64.strip())
    signing_key = nacl.signing.SigningKey(seed)
    sig = signing_key.sign(payload).signature
    return _b64_encode(sig)


def verify_signature(public_key_b64: str, payload: bytes, signature_b64: str) -> None:
    """
    Verify a detached Ed25519 signature. Raises BadSignatureError on failure.
    """
    vk_bytes = _b64_decode(public_key_b64.strip())
    verify_key = nacl.signing.VerifyKey(vk_bytes)
    sig = _b64_decode(signature_b64.strip())
    verify_key.verify(payload, sig)


