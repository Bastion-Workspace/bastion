"""
Thin facade: document metadata and admin flows go to document-service via gRPC.
Shared document_repository, embedding_manager, and kg_service remain on the backend for colocated embedding and tooling.
Document text processing uses utils.document_processor (initialized from the service container).
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import UploadFile

from clients.document_service_client import get_document_service_client
from models.api_models import (
    BulkCategorizeRequest,
    BulkOperationResponse,
    BulkUploadResponse,
    DocumentCategoriesResponse,
    DocumentFilterRequest,
    DocumentInfo,
    DocumentListResponse,
    DocumentStatus,
    DocumentUpdateRequest,
    DocumentUploadResponse,
    ProcessingStatus,
)

logger = logging.getLogger(__name__)

_DOC_CACHE_MAX = 512
_DOC_CACHE_TTL_SEC = 30.0


class _DocCacheMiss:
    """Sentinel indicating no cache entry for get_document."""

    __slots__ = ()


_DOC_CACHE_MISS = _DocCacheMiss()


class DocumentServiceFacade:
    """gRPC-backed document operations; keeps injected backend collaborators for embeddings and DB helpers."""

    def __init__(self) -> None:
        self.document_repository = None
        self.embedding_manager = None
        self.kg_service = None
        self.websocket_manager = None
        self._dsc = None
        self._doc_cache: Dict[str, tuple[float, Optional[DocumentInfo]]] = {}

    def _dsc_get(self):
        if self._dsc is None:
            self._dsc = get_document_service_client()
        return self._dsc

    def _cache_invalidate(self, document_id: str) -> None:
        self._doc_cache.pop(document_id, None)

    def _cache_get_hit(self, document_id: str) -> Union[_DocCacheMiss, Optional[DocumentInfo]]:
        """Return _DOC_CACHE_MISS if miss; else DocumentInfo or None (cached negative lookup)."""
        now = time.monotonic()
        ent = self._doc_cache.get(document_id)
        if not ent:
            return _DOC_CACHE_MISS
        ts, val = ent
        if now - ts > _DOC_CACHE_TTL_SEC:
            self._doc_cache.pop(document_id, None)
            return _DOC_CACHE_MISS
        return val

    def _cache_set(self, document_id: str, doc: Optional[DocumentInfo]) -> None:
        if len(self._doc_cache) >= _DOC_CACHE_MAX:
            try:
                oldest = next(iter(self._doc_cache))
                self._doc_cache.pop(oldest, None)
            except StopIteration:
                pass
        self._doc_cache[document_id] = (time.monotonic(), doc)

    async def initialize(
        self,
        enable_parallel: bool = True,
        processing_config: Any = None,
        shared_document_repository=None,
        shared_embedding_manager=None,
        shared_kg_service=None,
        websocket_manager=None,
        skip_incomplete_resume: bool = False,
    ) -> None:
        self.document_repository = shared_document_repository
        self.embedding_manager = shared_embedding_manager
        self.kg_service = shared_kg_service
        self.websocket_manager = websocket_manager
        dsc = self._dsc_get()
        await dsc.initialize(required=True)
        if not skip_incomplete_resume:
            try:
                ok, _, err = await dsc.document_mirror_json(
                    "",
                    {"action": "resume_incomplete_processing"},
                    timeout=600.0,
                )
                if not ok:
                    logger.warning("document-service resume_incomplete_processing: %s", err)
            except Exception as e:
                logger.warning("document-service resume_incomplete_processing failed: %s", e)
        logger.info("DocumentServiceFacade initialized")

    async def close(self) -> None:
        logger.debug("DocumentServiceFacade.close (no-op; shared resources owned by container)")

    async def _mirror(self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0) -> Dict[str, Any]:
        t0 = time.monotonic()
        dsc = self._dsc_get()
        await dsc.initialize(required=True)
        ok, data, err = await dsc.document_mirror_json(user_id or "", payload, timeout=timeout)
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug(
            "DS_CALL_LATENCY action=%s elapsed_ms=%.1f ok=%s",
            payload.get("action"),
            elapsed_ms,
            ok,
        )
        if not ok:
            raise RuntimeError(err or "DocumentMirror failed")
        return data or {}

    async def _emit_document_status_update(
        self,
        document_id: str,
        status: str,
        user_id: str = None,
        content_source: str = "embedding",
    ) -> None:
        try:
            if self.websocket_manager and self.document_repository:
                try:
                    document_metadata = await self.document_repository.get_document_metadata(document_id)
                    folder_id = document_metadata.get("folder_id") if document_metadata else None
                    filename = document_metadata.get("filename") if document_metadata else None
                    collection_type = (document_metadata or {}).get("collection_type") or "user"
                except Exception as e:
                    logger.warning("Could not get metadata for document %s: %s", document_id, e)
                    folder_id = None
                    filename = None
                    collection_type = "user"

                effective_user_id = user_id
                if (
                    collection_type == "team"
                    or not effective_user_id
                    or str(effective_user_id) == "system"
                ):
                    effective_user_id = None

                await self.websocket_manager.send_document_status_update(
                    document_id=document_id,
                    status=status,
                    folder_id=folder_id,
                    user_id=effective_user_id,
                    filename=filename,
                    content_source=content_source,
                )
        except Exception as e:
            logger.error("Failed to emit document status update: %s", e)

    async def get_document(self, document_id: str) -> Optional[DocumentInfo]:
        hit = self._cache_get_hit(document_id)
        if hit is not _DOC_CACHE_MISS:
            return hit
        data = await self._mirror("", {"action": "get_document", "document_id": document_id})
        raw = data.get("document")
        if not raw:
            self._cache_set(document_id, None)
            return None
        doc = DocumentInfo.model_validate(raw)
        self._cache_set(document_id, doc)
        return doc

    async def check_document_exists(self, doc_id: str) -> bool:
        data = await self._mirror("", {"action": "check_document_exists", "doc_id": doc_id})
        return bool(data.get("exists"))

    async def list_documents(self, skip: int = 0, limit: int = 100) -> List[DocumentInfo]:
        data = await self._mirror("", {"action": "list_documents", "skip": skip, "limit": limit})
        docs = data.get("documents") or []
        return [DocumentInfo.model_validate(d) for d in docs]

    async def filter_documents(self, filter_request: DocumentFilterRequest) -> DocumentListResponse:
        payload = {
            "action": "filter_documents",
            "filter_request": filter_request.model_dump(mode="json"),
        }
        data = await self._mirror("", payload, timeout=180.0)
        return DocumentListResponse.model_validate(data["response"])

    async def get_document_status(self, doc_id: str) -> Optional[DocumentStatus]:
        data = await self._mirror("", {"action": "get_document_status", "doc_id": doc_id})
        st = data.get("status")
        if not st:
            return None
        return DocumentStatus.model_validate(st)

    async def update_document_metadata(
        self, document_id: str, update_request: DocumentUpdateRequest
    ) -> bool:
        data = await self._mirror(
            "",
            {
                "action": "update_document_metadata",
                "document_id": document_id,
                "update_request": update_request.model_dump(mode="json"),
            },
        )
        self._cache_invalidate(document_id)
        return bool(data.get("success"))

    async def bulk_categorize_documents(self, bulk_request: BulkCategorizeRequest) -> BulkOperationResponse:
        data = await self._mirror(
            "",
            {
                "action": "bulk_categorize_documents",
                "bulk_request": bulk_request.model_dump(mode="json"),
            },
            timeout=300.0,
        )
        return BulkOperationResponse.model_validate(data["response"])

    async def delete_document_database_only(self, document_id: str) -> bool:
        data = await self._mirror(
            "", {"action": "delete_document_database_only", "document_id": document_id}
        )
        self._cache_invalidate(document_id)
        return bool(data.get("success"))

    async def cleanup_orphaned_embeddings(self) -> int:
        data = await self._mirror("", {"action": "cleanup_orphaned_embeddings"}, timeout=600.0)
        return int(data.get("cleaned_count", 0))

    async def get_duplicate_documents(self) -> Dict[str, List[DocumentInfo]]:
        data = await self._mirror("", {"action": "get_duplicate_documents"}, timeout=300.0)
        out: Dict[str, List[DocumentInfo]] = {}
        for k, v in (data.get("duplicates") or {}).items():
            if isinstance(v, list):
                out[k] = [DocumentInfo.model_validate(x) for x in v]
        return out

    async def get_documents_stats(self) -> Dict[str, Any]:
        data = await self._mirror("", {"action": "get_documents_stats"})
        return dict(data.get("stats") or {})

    async def get_document_categories_overview(self) -> DocumentCategoriesResponse:
        data = await self._mirror("", {"action": "get_document_categories_overview"})
        return DocumentCategoriesResponse.model_validate(data["overview"])

    async def import_from_url(self, url: str, content_type: str = "html") -> DocumentUploadResponse:
        data = await self._mirror(
            "",
            {"action": "import_from_url", "url": url, "content_type": content_type},
            timeout=120.0,
        )
        return DocumentUploadResponse.model_validate(data["upload"])

    async def get_documents_with_hierarchy(self, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        data = await self._mirror(
            "",
            {"action": "get_documents_with_hierarchy", "limit": limit, "offset": offset},
        )
        return dict(data.get("data") or {})

    async def get_zip_hierarchy(self, document_id: str) -> Dict[str, Any]:
        data = await self._mirror("", {"action": "get_zip_hierarchy", "document_id": document_id})
        return dict(data.get("data") or {})

    async def delete_zip_with_children(
        self, parent_document_id: str, delete_children: bool = True
    ) -> Dict[str, Any]:
        data = await self._mirror(
            "",
            {
                "action": "delete_zip_with_children",
                "parent_document_id": parent_document_id,
                "delete_children": delete_children,
            },
            timeout=300.0,
        )
        return dict(data.get("data") or {})

    async def remove_document_exemption(
        self, document_id: str, user_id: str = None, inherit: bool = False
    ) -> bool:
        dsc = self._dsc_get()
        await dsc.initialize(required=True)
        ok, data, err = await dsc.remove_document_exemption_json(
            user_id or "",
            {"document_id": document_id, "user_id": user_id or "", "inherit": inherit},
            timeout=600.0,
        )
        self._cache_invalidate(document_id)
        return bool(ok and data and data.get("success"))

    async def delete_document(self, document_id: str, user_id: str = None) -> bool:
        dsc = self._dsc_get()
        await dsc.initialize(required=True)
        ok, data, err = await dsc.delete_document_json(
            user_id or "",
            {"document_id": document_id, "user_id": user_id},
            timeout=600.0,
        )
        self._cache_invalidate(document_id)
        return bool(ok and data and data.get("success"))

    async def store_text_document(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any],
        filename: str = None,
        user_id: Optional[str] = None,
        collection_type: str = "user",
        folder_id: Optional[str] = None,
        file_path: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> bool:
        dsc = self._dsc_get()
        await dsc.initialize(required=True)
        uid = user_id or ""
        payload: Dict[str, Any] = {
            "doc_id": doc_id,
            "content": content,
            "metadata": metadata,
            "filename": filename,
            "user_id": uid,
            "collection_type": collection_type,
            "folder_id": folder_id,
            "file_path": file_path,
        }
        if team_id is not None:
            payload["team_id"] = team_id
        ok, data, err = await dsc.store_text_document_json(
            uid,
            payload,
            timeout=600.0,
        )
        self._cache_invalidate(doc_id)
        return bool(ok and data and data.get("success"))

    async def upload_and_process(
        self,
        file: UploadFile,
        doc_type: str = None,
        user_id: str = None,
        folder_id: str = None,
        team_id: str = None,
        folder_service=None,
        exempt_from_vectorization_override: Optional[bool] = None,
    ) -> DocumentUploadResponse:
        dsc = self._dsc_get()
        await dsc.initialize(required=True)
        collection_type = "team" if team_id else ("user" if user_id else "global")
        return await dsc.upload_via_document_service(
            file,
            doc_type=doc_type,
            user_id=user_id,
            folder_id=folder_id,
            team_id=team_id,
            collection_type=collection_type,
            exempt_from_vectorization=exempt_from_vectorization_override,
        )

    async def upload_multiple_documents(
        self, files: List[UploadFile], enable_parallel: bool = True
    ) -> BulkUploadResponse:
        start = time.time()
        dsc = self._dsc_get()
        await dsc.initialize(required=True)
        sem = asyncio.Semaphore(8 if enable_parallel else 1)

        async def _one(f: UploadFile) -> DocumentUploadResponse:
            async with sem:
                try:
                    return await dsc.upload_via_document_service(
                        f,
                        user_id=None,
                        folder_id=None,
                        team_id=None,
                        collection_type="global",
                    )
                except Exception as e:
                    logger.error("Bulk upload failed for %s: %s", getattr(f, "filename", ""), e)
                    return DocumentUploadResponse(
                        document_id="",
                        filename=getattr(f, "filename", "") or "",
                        status=ProcessingStatus.FAILED,
                        message=str(e),
                    )

        results = await asyncio.gather(*[_one(f) for f in files])
        ok_n = sum(1 for r in results if r.status != ProcessingStatus.FAILED)
        fail_n = len(results) - ok_n
        return BulkUploadResponse(
            total_files=len(files),
            successful_uploads=ok_n,
            failed_uploads=fail_n,
            upload_results=list(results),
            processing_time=time.time() - start,
            message=f"Uploaded {ok_n}/{len(files)} via document-service",
        )

    def _detect_document_type(self, filename: str) -> str:
        if filename and filename.lower().endswith(".metadata.json"):
            return "image_sidecar"
        ext = Path(filename).suffix.lower()
        type_map = {
            ".pdf": "pdf",
            ".txt": "txt",
            ".md": "md",
            ".org": "org",
            ".docx": "docx",
            ".pptx": "pptx",
            ".epub": "epub",
            ".html": "html",
            ".htm": "html",
            ".zip": "zip",
            ".jpg": "image",
            ".jpeg": "image",
            ".png": "image",
            ".gif": "image",
        }
        return type_map.get(ext, "txt")

    async def _process_url_async(
        self,
        document_id: str,
        url: str,
        content_type: str = "html",
        user_id: str = None,
    ) -> None:
        dsc = self._dsc_get()
        await dsc.initialize(required=True)
        ok, _, err = await dsc.document_mirror_json(
            user_id or "",
            {
                "action": "process_url_async",
                "document_id": document_id,
                "url": url,
                "content_type": content_type,
                "user_id": user_id,
            },
            timeout=30.0,
        )
        if not ok:
            logger.error("process_url_async mirror failed: %s", err)
