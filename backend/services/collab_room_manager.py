"""
Per-document Yjs rooms (pycrdt-websocket YRoom): lifecycle, DB snapshots, disk flush.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from pycrdt import Doc, Text
from pycrdt.websocket import YRoom

from services.collab_persist import (
    flush_collaborative_document,
    load_collab_state_row,
    read_document_plaintext_for_collab,
    save_collab_state_row,
)
from utils.collab_websocket_adapter import FastAPIWebsocketAdapter

logger = logging.getLogger(__name__)

CONTENT_KEY = "content"
_GRACE_SECONDS = 30.0
_PERIODIC_DB_SECONDS = 30.0
_YDOC_SNAPSHOT_MIN_BYTES = 1


def _encode_doc_snapshot(ydoc: Doc) -> bytes:
    if hasattr(ydoc, "get_update"):
        return ydoc.get_update()
    raise RuntimeError("pycrdt Doc has no get_update(); upgrade pycrdt")


def _plaintext_from_ydoc(ydoc: Doc) -> str:
    try:
        t = ydoc[CONTENT_KEY]
        if t is None:
            ydoc[CONTENT_KEY] = Text()
            t = ydoc[CONTENT_KEY]
        out = str(t)
        return out if out is not None else ""
    except Exception:
        return ""


async def _init_yroom_document(ydoc: Doc, document_id: str) -> None:
    blob = await load_collab_state_row(document_id)
    if blob and len(blob) >= _YDOC_SNAPSHOT_MIN_BYTES:
        try:
            ydoc.apply_update(blob)
            ydoc[CONTENT_KEY] = Text()
            return
        except Exception as e:
            logger.warning("collab: failed to apply stored ydoc for %s: %s", document_id, e)
    text = await read_document_plaintext_for_collab(document_id)
    with ydoc.transaction():
        ydoc[CONTENT_KEY] = Text(text)


class CollabRoomManager:
    """Singleton: one YRoom per document when collaborators are connected."""

    def __init__(self) -> None:
        self._guard = asyncio.Lock()
        self._rooms: Dict[str, YRoom] = {}
        self._start_tasks: Dict[str, asyncio.Task] = {}
        self._periodic_tasks: Dict[str, asyncio.Task] = {}
        self._grace_tasks: Dict[str, asyncio.Task] = {}

    async def get_or_create_room(self, document_id: str) -> YRoom:
        async with self._guard:
            existing = self._rooms.get(document_id)
            if existing is not None:
                return existing
        room = YRoom(ready=False)
        await _init_yroom_document(room.ydoc, document_id)
        room.ready = True
        start_task = asyncio.create_task(room.start())
        await room.started.wait()
        async with self._guard:
            if document_id in self._rooms:
                try:
                    await room.stop()
                except RuntimeError:
                    pass
                except Exception:
                    pass
                if not start_task.done():
                    start_task.cancel()
                    try:
                        await start_task
                    except (asyncio.CancelledError, Exception):
                        pass
                return self._rooms[document_id]
            self._rooms[document_id] = room
            self._start_tasks[document_id] = start_task
            self._periodic_tasks[document_id] = asyncio.create_task(
                self._periodic_db_persist(document_id)
            )
        logger.info("collab: opened room for document %s", document_id)
        return room

    async def _periodic_db_persist(self, document_id: str) -> None:
        try:
            while True:
                await asyncio.sleep(_PERIODIC_DB_SECONDS)
                async with self._guard:
                    room = self._rooms.get(document_id)
                if room is None:
                    return
                if not room.clients:
                    continue
                try:
                    blob = _encode_doc_snapshot(room.ydoc)
                    await save_collab_state_row(document_id, blob)
                except Exception as e:
                    logger.warning("collab: periodic DB save failed for %s: %s", document_id, e)
        except asyncio.CancelledError:
            return

    def _cancel_grace(self, document_id: str) -> None:
        t = self._grace_tasks.pop(document_id, None)
        if t and not t.done():
            t.cancel()

    async def _grace_close(self, document_id: str) -> None:
        try:
            await asyncio.sleep(_GRACE_SECONDS)
            async with self._guard:
                room = self._rooms.get(document_id)
                if room is None or room.clients:
                    return
            await self._destroy_room(document_id)
        except asyncio.CancelledError:
            return

    async def _destroy_room(self, document_id: str) -> None:
        async with self._guard:
            room = self._rooms.pop(document_id, None)
            per_task = self._periodic_tasks.pop(document_id, None)
            st_task = self._start_tasks.pop(document_id, None)
            self._grace_tasks.pop(document_id, None)
        if per_task and not per_task.done():
            per_task.cancel()
            try:
                await per_task
            except asyncio.CancelledError:
                pass
        if room is None:
            return
        try:
            plain = _plaintext_from_ydoc(room.ydoc)
            blob = _encode_doc_snapshot(room.ydoc)
            await save_collab_state_row(document_id, blob)
            await flush_collaborative_document(document_id, plain)
        except Exception as e:
            logger.error("collab: flush on room close failed for %s: %s", document_id, e)
        try:
            await room.stop()
        except Exception as e:
            logger.debug("collab: room.stop for %s: %s", document_id, e)
        if st_task and not st_task.done():
            st_task.cancel()
            try:
                await st_task
            except (asyncio.CancelledError, Exception):
                pass
        logger.info("collab: destroyed room for document %s", document_id)

    async def schedule_grace_if_empty(self, document_id: str) -> None:
        async with self._guard:
            room = self._rooms.get(document_id)
            if room is None or room.clients:
                return
            if document_id in self._grace_tasks:
                return
            self._grace_tasks[document_id] = asyncio.create_task(self._grace_close(document_id))

    async def serve_connection(self, document_id: str, adapter: FastAPIWebsocketAdapter) -> None:
        self._cancel_grace(document_id)
        room = await self.get_or_create_room(document_id)
        try:
            await room.serve(adapter)
        finally:
            await self.schedule_grace_if_empty(document_id)

    async def flush_now(self, document_id: str) -> None:
        """Persist current Y.Doc to DB and disk if a room is active."""
        async with self._guard:
            room = self._rooms.get(document_id)
        if room is None:
            return
        try:
            plain = _plaintext_from_ydoc(room.ydoc)
            blob = _encode_doc_snapshot(room.ydoc)
            await save_collab_state_row(document_id, blob)
            await flush_collaborative_document(document_id, plain)
        except Exception as e:
            logger.error("collab: manual flush failed for %s: %s", document_id, e)
            raise


_collab_manager: Optional[CollabRoomManager] = None


def get_collab_room_manager() -> CollabRoomManager:
    global _collab_manager
    if _collab_manager is None:
        _collab_manager = CollabRoomManager()
    return _collab_manager
