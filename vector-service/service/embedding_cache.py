"""
Embedding Cache - Hash-based caching with TTL and optional SQLite L2 persistence.
"""

import hashlib
import logging
import os
import sqlite3
import struct
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_HEALTHCHECK_HASH = "__bastion_embedding_cache_healthcheck__"


def _sqlite_parent_writable(db_path: str) -> bool:
    try:
        parent = Path(db_path).parent
        parent.mkdir(parents=True, exist_ok=True)
        return os.access(parent, os.W_OK)
    except OSError:
        return False


def _open_sqlite_with_journal_fallback(db_path: str) -> Optional[sqlite3.Connection]:
    """
    Open SQLite for embedding cache. Tries WAL, then DELETE journal mode.
    Named volumes may contain a DB owned by another UID; WAL sidecars can then
    fail with 'readonly database' even when the main file opens.
    """
    if not _sqlite_parent_writable(db_path):
        logger.warning(
            "Embedding cache path parent is not writable: %s (uid=%s gid=%s)",
            Path(db_path).parent,
            os.getuid(),
            os.getgid(),
        )
        return None

    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                content_hash TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        conn.commit()
    except sqlite3.Error:
        conn.close()
        raise

    for journal in ("wal", "delete"):
        try:
            conn.execute(f"PRAGMA journal_mode={journal}")
            conn.commit()
            conn.execute(
                "INSERT OR REPLACE INTO embedding_cache (content_hash, embedding, created_at) VALUES (?, ?, ?)",
                (_HEALTHCHECK_HASH, b"", 0.0),
            )
            conn.execute(
                "DELETE FROM embedding_cache WHERE content_hash = ?",
                (_HEALTHCHECK_HASH,),
            )
            conn.commit()
            logger.info(
                "Embedding cache SQLite journal_mode=%s path=%s",
                journal.upper(),
                db_path,
            )
            return conn
        except sqlite3.OperationalError as e:
            logger.warning(
                "Embedding cache SQLite journal_mode=%s failed at %s: %s",
                journal.upper(),
                db_path,
                e,
            )
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
            continue

    conn.close()
    return None


class EmbeddingCache:
    """Hash-based embedding cache with TTL; optional on-disk SQLite for L2."""

    def __init__(self, ttl_seconds: int = 10800, db_path: Optional[str] = None):
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, tuple[List[float], float]] = {}
        self.hits = 0
        self.misses = 0
        self._enabled = True
        raw = (db_path or "").strip()
        self._sqlite_requested_path: Optional[str] = raw if raw else None
        self._db_path: Optional[str] = raw if raw else None
        self._db: Optional[sqlite3.Connection] = None
        self._db_lock = threading.Lock()

    async def initialize(self):
        """Initialize in-memory cache and optional SQLite L2 store."""
        if self._sqlite_requested_path:
            configured = self._sqlite_requested_path
            tmp_fallback = os.path.join(
                tempfile.gettempdir(),
                f"bastion_vector_embedding_cache_{os.getuid()}.db",
            )
            try:
                with self._db_lock:
                    self._db = _open_sqlite_with_journal_fallback(configured)
                    used_path = configured
                    if self._db is None and tmp_fallback != configured:
                        self._db = _open_sqlite_with_journal_fallback(tmp_fallback)
                        used_path = tmp_fallback
                    if self._db is None:
                        raise sqlite3.OperationalError(
                            "SQLite L2 not writable at configured path or /tmp fallback"
                        )
                    self._db_path = used_path
                if used_path != configured:
                    logger.warning(
                        "Embedding cache L2 moved from %s to %s (named volume often "
                        "retains files from a previous container UID; remove volume "
                        "`vector-embedding-cache` or chown -R <uid>:<gid> on the volume "
                        "when changing BASTION_RUNTIME_* / compose user:)",
                        configured,
                        used_path,
                    )
                logger.info(
                    "Embedding cache initialized: TTL=%ss, SQLite L2=%s",
                    self.ttl_seconds,
                    used_path,
                )
            except Exception as e:
                logger.warning(
                    "Embedding cache SQLite L2 disabled (%s); using memory only: %s",
                    configured,
                    e,
                )
                self._db_path = None
                self._sqlite_requested_path = None
                self._db = None
        else:
            logger.info(f"Embedding cache initialized with {self.ttl_seconds}s TTL (memory only)")

    @staticmethod
    def _pack_embedding(embedding: List[float]) -> bytes:
        return struct.pack(f"{len(embedding)}f", *embedding)

    @staticmethod
    def _unpack_embedding(blob: bytes) -> List[float]:
        if not blob:
            return []
        n = len(blob) // struct.calcsize("f")
        return list(struct.unpack(f"{n}f", blob))

    def hash_text(self, text: str, model: str = "") -> str:
        """Generate stable hash for text content and model (avoids cross-provider cache hits)."""
        key = f"{model}:{text}" if model else text
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _get_sqlite(self, content_hash: str) -> Optional[List[float]]:
        if not self._db:
            return None
        now = time.time()
        with self._db_lock:
            cur = self._db.execute(
                "SELECT embedding, created_at FROM embedding_cache WHERE content_hash = ?",
                (content_hash,),
            )
            row = cur.fetchone()
        if not row:
            return None
        blob, created_at = row
        if (now - float(created_at)) >= self.ttl_seconds:
            with self._db_lock:
                self._db.execute(
                    "DELETE FROM embedding_cache WHERE content_hash = ?",
                    (content_hash,),
                )
                self._db.commit()
            return None
        return self._unpack_embedding(blob)

    def _tmp_sqlite_fallback_path(self) -> str:
        return os.path.join(
            tempfile.gettempdir(),
            f"bastion_vector_embedding_cache_{os.getuid()}.db",
        )

    def _set_sqlite(self, content_hash: str, embedding: List[float]) -> None:
        if not self._db:
            return
        blob = self._pack_embedding(embedding)
        ts = time.time()
        try:
            with self._db_lock:
                self._db.execute(
                    """
                    INSERT OR REPLACE INTO embedding_cache (content_hash, embedding, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (content_hash, blob, ts),
                )
                self._db.commit()
        except sqlite3.OperationalError as e:
            err = str(e).lower()
            if "readonly" not in err and "read-only" not in err:
                raise
            tmp_fb = self._tmp_sqlite_fallback_path()
            with self._db_lock:
                try:
                    if self._db:
                        self._db.close()
                except sqlite3.Error:
                    pass
                self._db = None
                if self._db_path and self._db_path != tmp_fb:
                    self._db = _open_sqlite_with_journal_fallback(tmp_fb)
                    if self._db:
                        self._db_path = tmp_fb
                        logger.warning(
                            "Embedding cache L2 moved to %s after readonly write on %s "
                            "(named volume `vector-embedding-cache` may need "
                            "`docker volume rm` or chown to uid=%s gid=%s)",
                            tmp_fb,
                            self._sqlite_requested_path or "(unknown)",
                            os.getuid(),
                            os.getgid(),
                        )
                        try:
                            self._db.execute(
                                """
                                INSERT OR REPLACE INTO embedding_cache (content_hash, embedding, created_at)
                                VALUES (?, ?, ?)
                                """,
                                (content_hash, blob, ts),
                            )
                            self._db.commit()
                            return
                        except sqlite3.OperationalError as e2:
                            logger.error(
                                "Embedding cache SQLite still not writable at %s: %s",
                                tmp_fb,
                                e2,
                            )
                            try:
                                self._db.close()
                            except sqlite3.Error:
                                pass
                            self._db = None
                            return
                logger.error(
                    "Embedding cache SQLite write failed (readonly): %s. "
                    "If using Docker named volume vector-embedding-cache, recreate it or "
                    "match volume file ownership to uid=%s gid=%s.",
                    e,
                    os.getuid(),
                    os.getgid(),
                )

    async def get(self, content_hash: str) -> Optional[List[float]]:
        """Get embedding from cache if not expired (L1 memory, then L2 SQLite)."""
        if not self._enabled:
            self.misses += 1
            return None

        if content_hash in self.cache:
            embedding, timestamp = self.cache[content_hash]
            age = time.time() - timestamp
            if age < self.ttl_seconds:
                self.hits += 1
                logger.debug(f"Cache hit (memory): {content_hash[:16]}... (age: {age:.1f}s)")
                return embedding
            del self.cache[content_hash]
            logger.debug(f"Cache expired (memory): {content_hash[:16]}... (age: {age:.1f}s)")

        if self._db:
            emb = self._get_sqlite(content_hash)
            if emb is not None:
                self.cache[content_hash] = (emb, time.time())
                self.hits += 1
                logger.debug(f"Cache hit (sqlite): {content_hash[:16]}...")
                return emb

        self.misses += 1
        return None

    async def set(self, content_hash: str, embedding: List[float]):
        """Store embedding in L1 and optional L2."""
        if self._enabled:
            ts = time.time()
            self.cache[content_hash] = (embedding, ts)
            logger.debug(f"Cached embedding (memory): {content_hash[:16]}...")
            self._set_sqlite(content_hash, embedding)

    async def clear(self, content_hash: Optional[str] = None) -> int:
        """Clear cache (all or specific hash) in memory and SQLite."""
        if content_hash:
            removed = 0
            if content_hash in self.cache:
                del self.cache[content_hash]
                removed = 1
                logger.info(f"Cleared cache entry (memory): {content_hash[:16]}...")
            if self._db:
                with self._db_lock:
                    cur = self._db.execute(
                        "DELETE FROM embedding_cache WHERE content_hash = ?",
                        (content_hash,),
                    )
                    self._db.commit()
                    rc = cur.rowcount or 0
                if rc:
                    removed = max(removed, rc)
                    logger.info(f"Cleared cache entry (sqlite): {content_hash[:16]}...")
            return removed
        count = len(self.cache)
        self.cache.clear()
        logger.info(f"Cleared entire memory cache ({count} entries)")
        if self._db:
            with self._db_lock:
                cur = self._db.execute("SELECT COUNT(*) FROM embedding_cache")
                n = int(cur.fetchone()[0] or 0)
                self._db.execute("DELETE FROM embedding_cache")
                self._db.commit()
            logger.info(f"Cleared entire sqlite cache ({n} entries)")
            return count + n
        return count

    def get_stats(self) -> Dict:
        """Get cache statistics (hits/misses are process-local)."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        size = len(self.cache)
        if self._db:
            try:
                with self._db_lock:
                    cur = self._db.execute("SELECT COUNT(*) FROM embedding_cache")
                    row = cur.fetchone()
                    if row:
                        size = max(size, int(row[0] or 0))
            except Exception as e:
                logger.debug("Could not count sqlite cache rows: %s", e)

        return {
            "size": size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl_seconds,
            "enabled": self._enabled,
            "sqlite_path": self._db_path,
        }

    async def cleanup_expired(self) -> int:
        """Remove expired entries from memory and SQLite."""
        now = time.time()
        expired_keys = [
            key
            for key, (_, timestamp) in self.cache.items()
            if (now - timestamp) >= self.ttl_seconds
        ]
        for key in expired_keys:
            del self.cache[key]
        removed = len(expired_keys)
        if self._db:
            cutoff = now - self.ttl_seconds
            try:
                with self._db_lock:
                    cur = self._db.execute(
                        "DELETE FROM embedding_cache WHERE created_at < ?", (cutoff,)
                    )
                    self._db.commit()
                    removed += cur.rowcount or 0
            except Exception as e:
                logger.warning("SQLite cache cleanup failed: %s", e)
        if removed:
            logger.info(f"Cleaned up {removed} expired cache entries")
        return removed

    def disable(self):
        """Disable cache"""
        self._enabled = False
        logger.info("Embedding cache disabled")

    def enable(self):
        """Enable cache"""
        self._enabled = True
        logger.info("Embedding cache enabled")
