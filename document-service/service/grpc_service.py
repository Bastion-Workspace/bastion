"""
gRPC DocumentService — Phase 1 (NER, upload, reprocess) + Phase 2 (handlers via mixins).
"""

import logging
import sys
from typing import AsyncIterator, Optional

sys.path.insert(0, "/app")

import grpc
from protos import document_service_pb2, document_service_pb2_grpc

from ds_handlers.document_edit_handlers_mixin import DocumentEditHandlersMixin
from ds_handlers.document_handlers_mixin import DocumentHandlersMixin
from ds_handlers.ds_entity_search_mixin import DsEntitySearchMixin
from ds_handlers.file_creation_handlers_mixin import FileCreationHandlersMixin
from ds_handlers.grpc_handler_base import DocumentGrpcHandlerBase
from ds_handlers.phase2_handlers import Phase2HandlersMixin
from service.entity_extractor import EntityExtractor
from service.processing.pipeline import DocumentProcessingPipeline

logger = logging.getLogger(__name__)

SERVICE_VERSION = "1.0.0"


class DocumentServiceImplementation(
    Phase2HandlersMixin,
    DocumentEditHandlersMixin,
    FileCreationHandlersMixin,
    DsEntitySearchMixin,
    DocumentHandlersMixin,
    DocumentGrpcHandlerBase,
    document_service_pb2_grpc.DocumentServiceServicer,
):
    """NER + processing pipeline + document / folder / file gRPC surface."""

    def __init__(self) -> None:
        DocumentGrpcHandlerBase.__init__(self)
        self.entity_extractor = EntityExtractor()
        self._initialized = False
        self._pipeline: Optional[DocumentProcessingPipeline] = None
        self._neo4j_maint_task = None
        self._neo4j_maint_stop = None
        self._vector_maint_task = None
        self._vector_maint_stop = None

    async def initialize(self) -> None:
        """Load spaCy and optionally the processing stack (also wires service_container for DS)."""
        try:
            await self.entity_extractor.initialize()
            self._initialized = True
            logger.info("Document Service NER initialized")
        except Exception as e:
            logger.error("Failed to initialize Document Service NER: %s", e)
            raise

        try:
            import os

            if os.getenv("DATABASE_URL"):
                self._pipeline = DocumentProcessingPipeline()
                await self._pipeline.initialize(self.entity_extractor)
                logger.info("Document processing pipeline initialized")
            else:
                logger.info("DATABASE_URL not set; upload/reprocess RPCs disabled")
        except Exception as e:
            logger.error("Failed to initialize processing pipeline: %s", e)
            self._pipeline = None

        if self._pipeline and getattr(self._pipeline, "_parallel", None):
            kg = getattr(self._pipeline._parallel, "kg_service", None)
            from ds_services.neo4j_maintenance import spawn_neo4j_maintenance_task

            self._neo4j_maint_task, self._neo4j_maint_stop = spawn_neo4j_maintenance_task(kg)

            emb = getattr(self._pipeline._parallel, "embedding_manager", None)
            from ds_services.vector_maintenance import spawn_vector_maintenance_task

            self._vector_maint_task, self._vector_maint_stop = spawn_vector_maintenance_task(
                emb
            )

    async def shutdown_neo4j_maintenance(self) -> None:
        from ds_services.neo4j_maintenance import cancel_neo4j_maintenance_task
        from ds_services.vector_maintenance import cancel_vector_maintenance_task

        await cancel_neo4j_maintenance_task(self._neo4j_maint_task, self._neo4j_maint_stop)
        self._neo4j_maint_task = None
        self._neo4j_maint_stop = None

        await cancel_vector_maintenance_task(self._vector_maint_task, self._vector_maint_stop)
        self._vector_maint_task = None
        self._vector_maint_stop = None

    async def ExtractEntities(
        self,
        request: document_service_pb2.ExtractEntitiesRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.ExtractEntitiesResponse:
        """Extract entities from text using spaCy."""
        try:
            text = request.text or ""
            max_length = request.max_length if request.max_length > 0 else None
            entities = self.entity_extractor.extract(text, max_length=max_length)
            entity_messages = [
                document_service_pb2.ExtractedEntity(
                    name=e.name,
                    entity_type=e.entity_type,
                    confidence=e.confidence,
                    context=e.context,
                )
                for e in entities
            ]
            return document_service_pb2.ExtractEntitiesResponse(
                entities=entity_messages,
                success=True,
            )
        except Exception as e:
            logger.exception("ExtractEntities failed")
            return document_service_pb2.ExtractEntitiesResponse(
                success=False,
                error=str(e),
            )

    async def HealthCheck(
        self,
        request: document_service_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.HealthCheckResponse:
        return document_service_pb2.HealthCheckResponse(
            status="healthy" if self.entity_extractor.is_loaded else "degraded",
            service_version=SERVICE_VERSION,
            gliner_loaded=self.entity_extractor.is_loaded,
        )

    async def UploadAndProcess(
        self,
        request_iterator: AsyncIterator[document_service_pb2.UploadChunk],
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.UploadResponse:
        if not self._pipeline:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            return document_service_pb2.UploadResponse(
                success=False,
                error="Processing pipeline not configured (set DATABASE_URL)",
            )
        metadata = None
        parts: list[bytes] = []
        try:
            async for chunk in request_iterator:
                if chunk.metadata and chunk.metadata.filename:
                    metadata = chunk.metadata
                if chunk.data:
                    parts.append(chunk.data)
            if not metadata or not metadata.filename:
                return document_service_pb2.UploadResponse(
                    success=False,
                    error="First chunk must include UploadMetadata with filename",
                )
            content = b"".join(parts)
            if not content:
                return document_service_pb2.UploadResponse(
                    success=False,
                    error="Empty upload",
                )
            result = await self._pipeline.upload_and_process(
                filename=metadata.filename,
                content=content,
                doc_type=(metadata.doc_type or "").strip(),
                user_id=(metadata.user_id or "").strip(),
                folder_id=(metadata.folder_id or "").strip(),
                team_id=(metadata.team_id or "").strip(),
                collection_type=(metadata.collection_type or "").strip(),
                exempt_from_vectorization=bool(metadata.exempt_from_vectorization),
            )
            return document_service_pb2.UploadResponse(
                success=True,
                document_id=result.get("document_id", ""),
                filename=result.get("filename", metadata.filename),
                status=result.get("status", ""),
                message=result.get("message", ""),
                duplicate=bool(result.get("duplicate")),
            )
        except Exception as e:
            logger.exception("UploadAndProcess failed")
            return document_service_pb2.UploadResponse(success=False, error=str(e))

    async def ReprocessDocument(
        self,
        request: document_service_pb2.ReprocessRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.ReprocessResponse:
        if not self._pipeline:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            return document_service_pb2.ReprocessResponse(
                success=False,
                error="Processing pipeline not configured",
            )
        try:
            force = bool(getattr(request, "force_reprocess", False))
            if not force:
                try:
                    for k, v in context.invocation_metadata():
                        lk = k.lower() if isinstance(k, str) else k.decode().lower()
                        if lk != "x-force-reprocess":
                            continue
                        val = v.decode().lower() if isinstance(v, (bytes, bytearray)) else str(v).lower()
                        if val in ("1", "true", "yes"):
                            force = True
                            break
                except Exception:
                    pass
            ok, err = await self._pipeline.reprocess_document(
                request.document_id,
                request.user_id or "",
                force_reprocess=force,
            )
            return document_service_pb2.ReprocessResponse(
                success=ok,
                document_id=request.document_id,
                error=err or "",
            )
        except Exception as e:
            logger.exception("ReprocessDocument failed")
            return document_service_pb2.ReprocessResponse(
                success=False,
                document_id=request.document_id,
                error=str(e),
            )

    async def GetProcessingStatus(
        self,
        request: document_service_pb2.ProcessingStatusRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.ProcessingStatusResponse:
        if not self._pipeline:
            return document_service_pb2.ProcessingStatusResponse(
                found=False,
                message="pipeline disabled",
            )
        try:
            found, status, msg = await self._pipeline.get_processing_status(request.document_id)
            return document_service_pb2.ProcessingStatusResponse(
                found=found,
                status=status,
                message=msg,
            )
        except Exception as e:
            return document_service_pb2.ProcessingStatusResponse(
                found=False,
                message=str(e),
            )

    async def DocumentMirror(
        self,
        request: document_service_pb2.JsonToolRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.JsonToolResponse:
        """JSON dispatch for backend DocumentServiceFacade (metadata, URL reprocess, ZIP ops)."""
        from ds_handlers.document_mirror_rpc import handle_document_mirror_request

        return await handle_document_mirror_request(request)
