"""
Help Docs Sync Service - Vectorize help documentation into a dedicated Qdrant collection.

On startup, diffs backend/help_docs (recursive) against a SHA256 manifest (stored under
UPLOAD_DIR so it persists across container restarts) and re-indexes only new or changed files.
Deleted files are removed from the collection. Topic ids match the help API: path relative to
help_docs with forward slashes (e.g. getting-started/01-welcome).
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client.models import PointStruct

from config import settings
from clients.vector_service_client import get_vector_service_client
from services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)

HELP_DOCS_COLLECTION = "help_docs"
MANIFEST_FILENAME = ".sync_manifest.json"

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_HELP_DOCS_DIR = _BACKEND_DIR / "help_docs"
if not _HELP_DOCS_DIR.exists():
    _HELP_DOCS_DIR = Path("/app/help_docs")


def _persistent_manifest_path() -> Path:
    """Manifest on the uploads volume so it survives container recreation."""
    return Path(settings.UPLOAD_DIR) / "help_docs_sync_manifest.json"


def _legacy_manifest_path() -> Path:
    """Previous location (image layer only); still read for one-time migration."""
    return _HELP_DOCS_DIR / MANIFEST_FILENAME


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Split frontmatter and body. Returns (frontmatter_dict, body_str)."""
    if not content.strip().startswith("---"):
        return {}, content.strip()
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content.strip()
    frontmatter_str = parts[1].strip()
    body = parts[2].strip()
    fm = {}
    for line in frontmatter_str.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            fm[key.strip().lower()] = value.strip()
    return fm, body


def _topic_id_from_path(rel_path: Path) -> str:
    """Topic id = path relative to help_docs with forward slashes, no .md suffix."""
    return str(rel_path.with_suffix("")).replace("\\", "/")


def _chunk_by_sections(body: str, title: str) -> List[str]:
    """Split markdown body by ## headers. Each section becomes one chunk with title prepended."""
    chunks = []
    parts = re.split(r"\n(?=##\s)", body.strip())
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        if part.startswith("## "):
            chunk_text = f"# {title}\n\n{part}"
        else:
            chunk_text = f"# {title}\n\n{part}" if title else part
        chunks.append(chunk_text)
    if not chunks and body.strip():
        chunks.append(f"# {title}\n\n{body.strip()}" if title else body.strip())
    return chunks


def _point_id(topic_id: str, chunk_index: int) -> int:
    """Deterministic point ID for Qdrant (unsigned 63-bit)."""
    raw = hashlib.md5(f"help:{topic_id}:{chunk_index}".encode()).hexdigest()
    return int(raw, 16) % (2**63)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HelpDocsSyncService:
    """Syncs help_docs markdown files into the help_docs Qdrant collection."""

    def __init__(self) -> None:
        self._vector_store: Optional[VectorStoreService] = None
        self._vector_client = None

    async def _get_vector_store(self) -> VectorStoreService:
        if self._vector_store is None:
            self._vector_store = VectorStoreService()
            await self._vector_store.initialize()
        return self._vector_store

    async def _get_vector_client(self):
        if self._vector_client is None:
            self._vector_client = await get_vector_service_client(required=False)
        return self._vector_client

    def _read_manifest(self) -> Dict[str, str]:
        for path in (_persistent_manifest_path(), _legacy_manifest_path()):
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception as e:
                logger.warning("Could not read help docs manifest %s: %s", path, e)
        return {}

    def _write_manifest(self, manifest: Dict[str, str]) -> None:
        path = _persistent_manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    async def sync(self) -> None:
        """Diff help_docs against manifest, re-index changed/new, remove deleted."""
        if not _HELP_DOCS_DIR.exists():
            logger.warning("Help docs directory not found: %s", _HELP_DOCS_DIR)
            return

        manifest = self._read_manifest()
        vector_store = await self._get_vector_store()
        vector_client = await self._get_vector_client()
        if not vector_client:
            logger.warning("Vector service unavailable, skipping help docs sync")
            return

        await vector_store.ensure_collection_exists(HELP_DOCS_COLLECTION)

        current_files: Dict[str, str] = {}
        for path in sorted(_HELP_DOCS_DIR.rglob("*.md")):
            if path.name.startswith("."):
                continue
            try:
                rel = path.relative_to(_HELP_DOCS_DIR)
                if rel.parts[0].startswith("."):
                    continue
                topic_id = _topic_id_from_path(rel)
                current_files[topic_id] = _file_sha256(path)
            except Exception as e:
                logger.warning("Skipping help file %s: %s", path, e)

        for topic_id in list(manifest.keys()):
            if topic_id not in current_files:
                await vector_store.delete_points_by_filter(
                    document_id=topic_id,
                    collection_name=HELP_DOCS_COLLECTION,
                )
                del manifest[topic_id]
                logger.info("Removed help topic from vector store: %s", topic_id)

        for topic_id, file_hash in current_files.items():
            if manifest.get(topic_id) == file_hash:
                continue
            path = (_HELP_DOCS_DIR / topic_id).with_suffix(".md")
            if not path.exists() or not path.is_file():
                continue
            await self._index_topic(path, topic_id, file_hash, vector_store, vector_client)
            manifest[topic_id] = file_hash

        self._write_manifest(manifest)
        logger.info("Help docs sync complete: %d topics", len(manifest))

    async def _index_topic(
        self,
        path: Path,
        topic_id: str,
        file_hash: str,
        vector_store: VectorStoreService,
        vector_client: Any,
    ) -> None:
        """Delete existing points for topic, chunk content, embed, and upsert."""
        await vector_store.delete_points_by_filter(
            document_id=topic_id,
            collection_name=HELP_DOCS_COLLECTION,
        )
        raw = path.read_text(encoding="utf-8")
        fm, body = _parse_frontmatter(raw)
        title = fm.get("title", path.stem.replace("-", " ").replace("_", " ").title())
        chunks = _chunk_by_sections(body, title)
        if not chunks:
            logger.debug("No content to embed for topic %s", topic_id)
            return

        texts = chunks
        try:
            embeddings = await vector_client.generate_embeddings(texts)
        except Exception as e:
            logger.warning("Failed to embed help topic %s: %s", topic_id, e)
            return

        if len(embeddings) != len(chunks):
            logger.warning("Embedding count mismatch for %s", topic_id)
            return

        points = []
        for i, (text, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = _point_id(topic_id, i)
            payload = {
                "document_id": topic_id,
                "topic_id": topic_id,
                "title": title,
                "document_title": title,
                "chunk_index": i,
                "content": text,
                "file_hash": file_hash,
                "source": "help_docs",
            }
            points.append(
                PointStruct(id=point_id, vector=embedding, payload=payload)
            )

        success = await vector_store.insert_points(
            points=points,
            collection_name=HELP_DOCS_COLLECTION,
        )
        if success:
            logger.info("Indexed help topic %s (%d chunks)", topic_id, len(points))
        else:
            logger.warning("Failed to insert points for help topic %s", topic_id)
