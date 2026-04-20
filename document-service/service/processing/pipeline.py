"""
Document processing pipeline (vendored backend stack, ds_* imports).
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from starlette.datastructures import UploadFile

from ds_config import settings
from ds_models.api_models import Entity
from ds_processing.document_processor import DocumentProcessor
from ds_services.folder_service import FolderService
from ds_services.parallel_document_service import ParallelDocumentService

from bastion_indexing.policy import APP_CHUNK_INDEX_SCHEMA_VERSION, is_chunk_index_eligible

from service.redis_status_shim import RedisDocumentStatusBridge

logger = logging.getLogger(__name__)


class DocumentProcessingPipeline:
    """Owns ParallelDocumentService + FolderService for upload/reprocess."""

    def __init__(self) -> None:
        self._bridge: Optional[RedisDocumentStatusBridge] = None
        self._parallel: Any = None
        self._folder_service: Any = None

    async def initialize(self, entity_extractor: Any) -> None:
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        self._bridge = RedisDocumentStatusBridge(redis_url)

        self._folder_service = FolderService()
        await self._folder_service.initialize()

        pds = ParallelDocumentService()
        await pds.initialize(
            websocket_manager=self._bridge,
            skip_incomplete_resume=False,
        )
        self._parallel = pds

        dp = DocumentProcessor.get_instance()

        class _InternalNERClient:
            def __init__(self, ex: Any) -> None:
                self._ex = ex
                self._initialized = True

            async def extract_entities(self, text: str, max_length: Optional[int] = None):
                ents = self._ex.extract(text, max_length=max_length)
                return [
                    Entity(
                        name=e.name,
                        entity_type=e.entity_type,
                        confidence=e.confidence,
                        source_chunk="",
                        metadata={"source": "spacy", "context": e.context or ""},
                    )
                    for e in ents
                ]

        dp.document_service_client = _InternalNERClient(entity_extractor)

        from shims.services.service_container import set_document_services

        set_document_services(self._parallel, self._folder_service)
        logger.info("Document processing pipeline initialized")

    async def close(self) -> None:
        if self._bridge:
            await self._bridge.aclose()
            self._bridge = None
        try:
            from shims.services.service_container import clear_document_services

            clear_document_services()
        except Exception:
            pass
        self._parallel = None
        self._folder_service = None

    async def upload_and_process(
        self,
        *,
        filename: str,
        content: bytes,
        doc_type: str,
        user_id: str,
        folder_id: str,
        team_id: str,
        collection_type: str,
        exempt_from_vectorization: bool,
    ) -> Dict[str, Any]:
        from io import BytesIO

        if not self._parallel or not self._folder_service:
            raise RuntimeError("Pipeline not initialized")

        buf = BytesIO(content)
        upload = UploadFile(filename=filename or "upload", file=buf)

        if team_id:
            ct = "team"
        elif user_id:
            ct = "user"
        else:
            ct = "global"
        if collection_type in ("user", "team", "global"):
            ct = collection_type

        ex_override = True if exempt_from_vectorization else None

        result = await self._parallel.upload_and_process(
            upload,
            doc_type or None,
            user_id=user_id or None,
            folder_id=folder_id or None,
            team_id=team_id or None,
            folder_service=self._folder_service,
            exempt_from_vectorization_override=ex_override,
        )
        msg = result.message or ""
        dup = "duplicate" in msg.lower()
        return {
            "document_id": result.document_id,
            "filename": result.filename,
            "status": result.status.value if hasattr(result.status, "value") else str(result.status),
            "message": msg,
            "duplicate": dup,
        }

    async def reprocess_document(
        self, document_id: str, user_id: str, force_reprocess: bool = False
    ) -> Tuple[bool, str]:
        if not self._parallel or not self._folder_service:
            raise RuntimeError("Pipeline not initialized")

        pds = self._parallel
        fs = self._folder_service

        doc_info = await pds.get_document(document_id)
        if not doc_info:
            return False, "Document not found"

        doc_type_val = (
            doc_info.doc_type.value
            if hasattr(doc_info.doc_type, "value")
            else str(doc_info.doc_type)
        )
        zip_flag = getattr(doc_info, "is_zip_container", None)

        if not is_chunk_index_eligible(doc_type_val, zip_flag):
            await pds.document_repository.update_chunk_count(document_id, 0)
            return True, "ineligible_doc_type"

        if not force_reprocess:
            idx_at = getattr(doc_info, "chunk_indexed_at", None)
            idx_hash = getattr(doc_info, "chunk_indexed_file_hash", None) or ""
            idx_ver = int(getattr(doc_info, "chunk_index_schema_version", 0) or 0)
            file_hash = doc_info.file_hash or ""
            if idx_at and idx_hash == file_hash and idx_ver == APP_CHUNK_INDEX_SCHEMA_VERSION:
                has_chunks = await pds.document_repository.document_has_chunks(document_id)
                if not has_chunks:
                    logger.info(
                        "Chunk index state marked fresh but no rows in document_chunks; "
                        "clearing drift state for %s",
                        document_id,
                    )
                    await pds.document_repository.clear_chunk_index_state(document_id)
                else:
                    return True, "already_indexed"
        else:
            await pds.document_repository.clear_chunk_index_state(document_id)

        collection_type = getattr(doc_info, "collection_type", None) or "user"
        team_id = getattr(doc_info, "team_id", None)
        if team_id is not None and not isinstance(team_id, str):
            team_id = str(team_id)
        path_user_id = getattr(doc_info, "user_id", None) or user_id

        if force_reprocess:
            await pds.document_repository.reset_processing_retry_fields_for_reprocess(
                document_id, user_id=path_user_id
            )

        file_path = None
        try:
            folder_path = await fs.get_document_file_path(
                filename=doc_info.filename,
                folder_id=getattr(doc_info, "folder_id", None),
                user_id=path_user_id,
                collection_type=collection_type,
                team_id=team_id,
            )
            if folder_path and Path(folder_path).exists():
                file_path = Path(folder_path)
            else:
                filename_with_id = f"{document_id}_{doc_info.filename}"
                folder_path = await fs.get_document_file_path(
                    filename=filename_with_id,
                    folder_id=getattr(doc_info, "folder_id", None),
                    user_id=path_user_id,
                    collection_type=collection_type,
                    team_id=team_id,
                )
                if folder_path and Path(folder_path).exists():
                    file_path = Path(folder_path)
        except Exception as e:
            logger.warning("reprocess path resolution failed: %s", e)

        if not file_path or not file_path.exists():
            upload_dir = Path(settings.UPLOAD_DIR)
            for potential_file in upload_dir.glob(f"{document_id}_*"):
                file_path = potential_file
                break

        if not file_path or not file_path.exists():
            return False, "File not found on disk"

        doc_type = pds._detect_document_type(doc_info.filename)
        await pds._process_document_async(document_id, file_path, doc_type, path_user_id)
        return True, ""

    async def get_processing_status(self, document_id: str) -> Tuple[bool, str, str]:
        if not self._parallel:
            raise RuntimeError("Pipeline not initialized")
        doc = await self._parallel.document_repository.get_by_id(document_id)
        if not doc:
            return False, "", "not found"
        st = getattr(doc, "status", None)
        val = st.value if hasattr(st, "value") else str(st)
        return True, val, ""
