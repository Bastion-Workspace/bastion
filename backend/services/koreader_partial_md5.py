"""KOReader-compatible partial file digest (matches frontend.util.partialMD5 in KOReader)."""

from __future__ import annotations

import hashlib

# Parity reference (must match frontend `koreaderPartialMd5` / KOReader util.partialMD5):
# 10240 bytes where data[i] = i % 256 → digest a52dff8366b1473d2e13edd2415def67


def koreader_partial_md5(data: bytes) -> str:
    """
    Sample bytes at exponentially spaced offsets (1024 * 4^i for i=-1..10),
    hashing 1024 bytes at each offset until read fails. Used as document id for KoSync.
    """
    h = hashlib.md5()
    step = 1024
    size = 1024
    for i in range(-1, 11):
        shift = 2 * i
        if shift >= 0:
            offset = step << shift
        else:
            offset = step >> (-shift)
        if offset < 0:
            offset = 0
        chunk = data[offset : offset + size]
        if not chunk:
            break
        h.update(chunk)
    return h.hexdigest()
