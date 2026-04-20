#!/usr/bin/env python3
"""
If BBS_ENABLE_SSH is set and the configured host key file is missing, create an
OpenSSH-format ed25519 private key (cryptography, same stack as AsyncSSH).
"""

from __future__ import annotations

import os
import stat
import sys


def _truthy(raw: str | None) -> bool:
    return (raw or "").strip().lower() in ("1", "true", "yes", "on")


def main() -> int:
    if not _truthy(os.environ.get("BBS_ENABLE_SSH")):
        return 0
    path = (os.environ.get("BBS_SSH_HOST_KEY") or "/keys/ssh_host_ed25519_key").strip()
    if not path or os.path.isfile(path):
        return 0
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True, mode=0o755)

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    key = ed25519.Ed25519PrivateKey.generate()
    data = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.OpenSSH,
        encryption_algorithm=serialization.NoEncryption(),
    )
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ensure_ssh_host_key: {e}", file=sys.stderr)
        raise SystemExit(1) from e
