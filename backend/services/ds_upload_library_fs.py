"""
Async read/write for paths under the document library root (document-service UPLOAD_DIR).

Backend containers typically do not mount ./uploads; all library file bytes go through DS gRPC.
Paths may be absolute (/app/uploads/...) or already relative (Users/alice/Org/inbox.org).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

from clients.document_service_client import get_document_service_client
from config import settings

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def rel_from_uploads(path: PathLike) -> str:
    p = Path(path)
    s = str(p).replace("\\", "/")
    root = str(Path(settings.UPLOAD_DIR)).rstrip("/").replace("\\", "/")
    if s.startswith(root):
        return s[len(root) :].lstrip("/")
    return s.lstrip("/")


async def read_text(user_id: str, path: PathLike, *, encoding: str = "utf-8") -> str:
    rel = rel_from_uploads(path)
    dsc = get_document_service_client()
    await dsc.initialize(required=True)
    ok, data, err = await dsc.read_upload_relative_file_json(user_id, {"rel_path": rel})
    if not ok:
        raise FileNotFoundError(err or rel)
    return data.get("content", "") if data else ""


async def write_text(user_id: str, path: PathLike, content: str, *, encoding: str = "utf-8") -> None:
    rel = rel_from_uploads(path)
    dsc = get_document_service_client()
    await dsc.initialize(required=True)
    ok, _data, err = await dsc.write_upload_relative_file_json(
        user_id, {"rel_path": rel, "content": content}
    )
    if not ok:
        raise OSError(err or "write failed")


async def delete_file(user_id: str, path: PathLike) -> None:
    """Remove a single file under the library root (not directories)."""
    rel = rel_from_uploads(path)
    dsc = get_document_service_client()
    await dsc.initialize(required=True)
    ok, _data, err = await dsc.write_upload_relative_file_json(
        user_id, {"rel_path": rel, "delete": True}
    )
    if not ok:
        raise FileNotFoundError(err or rel)


async def append_text(user_id: str, path: PathLike, suffix: str, *, encoding: str = "utf-8") -> None:
    try:
        cur = await read_text(user_id, path, encoding=encoding)
    except FileNotFoundError:
        cur = ""
    await write_text(user_id, path, cur + suffix, encoding=encoding)


async def is_dir(user_id: str, path: PathLike) -> bool:
    rel = rel_from_uploads(path)
    dsc = get_document_service_client()
    await dsc.initialize(required=True)
    ok, _data, _err = await dsc.list_upload_relative_dir_json(user_id, {"rel_path": rel})
    return bool(ok)


async def exists(user_id: str, path: PathLike) -> bool:
    if await is_dir(user_id, path):
        return True
    try:
        await read_text(user_id, path)
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        logger.debug("exists check failed for %s: %s", path, e)
        return False


async def list_dir_names(user_id: str, path: PathLike) -> List[str]:
    rel = rel_from_uploads(path)
    dsc = get_document_service_client()
    await dsc.initialize(required=True)
    ok, data, _err = await dsc.list_upload_relative_dir_json(user_id, {"rel_path": rel})
    if not ok or not data:
        return []
    return list(data.get("entries") or [])


async def walk_org_files(user_id: str, username: str, *, include_archives: bool = False) -> List[Path]:
    """
    Return logical Paths under UPLOAD_DIR for each *.org file under Users/{username}/,
    using BFS via document-service ListUploadRelativeFile. Skips .versions directories.
    """
    base_rel = f"Users/{username}"
    out: List[Path] = []
    queue: List[str] = [base_rel]
    seen_dirs: set[str] = set()
    while queue:
        rel = queue.pop(0)
        norm = rel.replace("\\", "/")
        if "/.versions" in norm or norm.endswith("/.versions"):
            continue
        if norm in seen_dirs:
            continue
        seen_dirs.add(norm)
        try:
            names = await list_dir_names(user_id, Path(settings.UPLOAD_DIR) / rel)
        except Exception:
            continue
        for name in names:
            if name.startswith(".") or name.endswith("~"):
                continue
            child_rel = f"{rel}/{name}" if rel else name
            if name.lower().endswith(".org"):
                if not include_archives and name.endswith("_archive.org"):
                    continue
                try:
                    body = await read_text(user_id, Path(settings.UPLOAD_DIR) / child_rel)
                except FileNotFoundError:
                    continue
                except OSError:
                    continue
                if not body.strip() and name.lower() != "inbox.org":
                    continue
                out.append(Path(settings.UPLOAD_DIR) / child_rel)
            else:
                sub = f"{rel}/{name}" if rel else name
                if await is_dir(user_id, Path(settings.UPLOAD_DIR) / sub):
                    queue.append(sub)
    out.sort(key=lambda p: str(p))
    return out
