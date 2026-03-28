"""
Fetch a default Piper voice on first run so local TTS works out of the box.
Models persist under PIPER_MODEL_PATH (e.g. Docker volume voice-models:/app/models).
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Tuple

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

# rhasspy/piper-voices — https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/arctic/medium
_BOOTSTRAP_VOICE_STEM = "en_US-arctic-medium"
_HF_REPO_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
_HF_VOICE_SUBPATH = "en/en_US/arctic/medium"
_CONNECT_TIMEOUT = 30.0
_READ_TIMEOUT = 600.0


def _bootstrap_enabled() -> bool:
    raw = os.getenv("PIPER_BOOTSTRAP", "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _bootstrap_voice_stem() -> str:
    return os.getenv("PIPER_BOOTSTRAP_VOICE", _BOOTSTRAP_VOICE_STEM).strip() or _BOOTSTRAP_VOICE_STEM


def _voice_onnx_exists(stem: str) -> bool:
    base = Path(settings.PIPER_MODEL_PATH)
    if not base.exists():
        return False
    for p in base.rglob(f"{stem}.onnx"):
        if p.is_file():
            return True
    return False


def _bootstrap_urls(stem: str) -> Tuple[str, str]:
    if stem != _BOOTSTRAP_VOICE_STEM:
        raise ValueError(
            f"PIPER_BOOTSTRAP_VOICE={stem!r} is not supported; "
            f"only {_BOOTSTRAP_VOICE_STEM!r} has a bundled Hugging Face path. "
            "Set PIPER_BOOTSTRAP=0 and install models manually."
        )
    prefix = f"{_HF_REPO_BASE}/{_HF_VOICE_SUBPATH}/{stem}"
    return (f"{prefix}.onnx", f"{prefix}.onnx.json")


async def _download_to_path(client: httpx.AsyncClient, url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=dest.parent, prefix=f".{dest.name}.", suffix=".part"
    )
    try:
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        async with client.stream("GET", url, follow_redirects=True) as response:
            response.raise_for_status()
            with open(tmp_path, "wb") as out:
                async for chunk in response.aiter_bytes():
                    if chunk:
                        out.write(chunk)
        os.replace(tmp_path, dest)
    finally:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        except OSError:
            pass


async def ensure_piper_bootstrap_voice() -> None:
    """
    If enabled and the bootstrap voice is missing, download .onnx + .json from Hugging Face.
    Safe to call on every startup (idempotent).
    """
    if not _bootstrap_enabled():
        logger.info("Piper bootstrap disabled (PIPER_BOOTSTRAP)")
        return

    stem = _bootstrap_voice_stem()
    if _voice_onnx_exists(stem):
        logger.info("Piper bootstrap skipped: %s.onnx already present", stem)
        return

    try:
        onnx_url, json_url = _bootstrap_urls(stem)
    except ValueError as e:
        logger.warning("%s", e)
        return

    dest_dir = Path(settings.PIPER_MODEL_PATH)
    onnx_dest = dest_dir / f"{stem}.onnx"
    json_dest = dest_dir / f"{stem}.onnx.json"

    timeout = httpx.Timeout(
        connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT, write=_READ_TIMEOUT, pool=30.0
    )
    logger.info(
        "Piper bootstrap: downloading %s from Hugging Face into %s",
        stem,
        dest_dir,
    )
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            await _download_to_path(client, onnx_url, onnx_dest)
            await _download_to_path(client, json_url, json_dest)
    except Exception as e:
        logger.warning(
            "Piper bootstrap failed (%s). Local Piper TTS may be unavailable until "
            "models are placed under %s",
            e,
            settings.PIPER_MODEL_PATH,
        )
        for p in (onnx_dest, json_dest):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        return

    logger.info("Piper bootstrap complete: %s", onnx_dest)
