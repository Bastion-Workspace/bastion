"""
BBS server configuration from environment variables.
"""

import os


def _env_bool(name: str, default: str = "0") -> bool:
    v = (os.getenv(name, default) or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int((os.getenv(name, str(default)) or "").strip())
    except ValueError:
        return default


def _clean_service_key(raw: str) -> str:
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1].strip()
    return s


class Settings:
    SERVICE_NAME: str = os.getenv("SERVICE_NAME", "bbs-server")
    BACKEND_URL: str = os.getenv("BACKEND_URL", "http://backend:8000").rstrip("/")
    # Strip whitespace / accidental quotes so value matches backend exactly.
    INTERNAL_SERVICE_KEY: str = _clean_service_key(os.getenv("INTERNAL_SERVICE_KEY", ""))
    # Telnet off unless explicitly enabled (cleartext). SSH defaults off in
    # process env; docker-compose sets BBS_ENABLE_SSH when deploying bbs-server.
    BBS_ENABLE_TELNET: bool = _env_bool("BBS_ENABLE_TELNET", "0")
    BBS_ENABLE_SSH: bool = _env_bool("BBS_ENABLE_SSH", "0")
    BBS_TELNET_PORT: int = int(os.getenv("BBS_TELNET_PORT", "2323"))
    BBS_SSH_PORT: int = int(os.getenv("BBS_SSH_PORT", "2222"))
    BBS_SSH_HOST_KEY: str = os.getenv("BBS_SSH_HOST_KEY", "/keys/ssh_host_ed25519_key").strip()
    BBS_NAME: str = os.getenv("BBS_NAME", "Bastion BBS")
    BBS_MAX_CONNECTIONS: int = int(os.getenv("BBS_MAX_CONNECTIONS", "20"))
    BBS_IDLE_TIMEOUT: int = int(os.getenv("BBS_IDLE_TIMEOUT", "600"))
    BBS_IDLE_WARN_SECONDS: int = int(os.getenv("BBS_IDLE_WARN_SECONDS", "60"))
    # 0 = disabled. After N seconds with no input, blank the screen (Esc only resumes); never in wallpaper pane.
    BBS_SCREEN_BLANK_AFTER_SECONDS: int = max(0, _env_int("BBS_SCREEN_BLANK_AFTER_SECONDS", 180))
    BBS_THEME: str = os.getenv("BBS_THEME", "green").lower()
    BBS_MOTD: str = os.getenv("BBS_MOTD", "")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    EXTERNAL_CHAT_TIMEOUT: float = float(os.getenv("EXTERNAL_CHAT_TIMEOUT", "300"))
    HTTP_TIMEOUT: float = float(os.getenv("BBS_HTTP_TIMEOUT", "60"))
    # Longer timeout for loading/saving large documents in the telnet editor.
    BBS_EDITOR_HTTP_TIMEOUT: float = float(os.getenv("BBS_EDITOR_HTTP_TIMEOUT", "120"))
    # Refuse to open the BBS editor for bodies larger than this (bytes).
    BBS_EDITOR_MAX_BYTES: int = int(os.getenv("BBS_EDITOR_MAX_BYTES", str(2 * 1024 * 1024)))

    @classmethod
    def validate(cls) -> None:
        if not cls.INTERNAL_SERVICE_KEY:
            raise RuntimeError("INTERNAL_SERVICE_KEY is required for BBS server")
        if not cls.BACKEND_URL:
            raise RuntimeError("BACKEND_URL is required")
        if not cls.BBS_ENABLE_TELNET and not cls.BBS_ENABLE_SSH:
            raise RuntimeError("At least one of BBS_ENABLE_TELNET or BBS_ENABLE_SSH must be true")
        if cls.BBS_ENABLE_SSH:
            if not cls.BBS_SSH_HOST_KEY:
                raise RuntimeError("BBS_ENABLE_SSH requires BBS_SSH_HOST_KEY to be set")
            if not os.path.isfile(cls.BBS_SSH_HOST_KEY):
                raise RuntimeError(
                    "BBS_ENABLE_SSH requires the host key file to exist (missing: "
                    f"{cls.BBS_SSH_HOST_KEY}). Ensure docker-entrypoint ran or create the key manually."
                )


settings = Settings()
