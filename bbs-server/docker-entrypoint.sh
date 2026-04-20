#!/bin/sh
set -e
# Run as root: create SSH host key when SSH is enabled and key path is empty.
python3 /app/scripts/ensure_ssh_host_key.py
key="${BBS_SSH_HOST_KEY:-/keys/ssh_host_ed25519_key}"
if [ -f "$key" ]; then
  chown bbsuser:appgroup "$key" 2>/dev/null || true
  chmod 600 "$key" 2>/dev/null || true
fi
if [ -d /keys ]; then
  chown -R bbsuser:appgroup /keys 2>/dev/null || true
fi
if command -v runuser >/dev/null 2>&1; then
  exec runuser -u bbsuser -g appgroup -- "$@"
fi
exec setpriv --reuid "$(id -u bbsuser)" --regid "$(id -g bbsuser)" --init-groups -- "$@"
