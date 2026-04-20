"""
Document Service gRPC Client

Entity extraction (spaCy), upload/reprocess, and Phase 2 file/folder I/O via gRPC.
"""

import json
import logging
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

import grpc

from config import get_settings
from protos import document_service_pb2, document_service_pb2_grpc
from protos import tool_service_pb2

from models.api_models import DocumentUploadResponse, Entity, ProcessingStatus
from starlette.datastructures import UploadFile

logger = logging.getLogger(__name__)

_document_service_client_singleton: Optional["DocumentServiceClient"] = None


def get_document_service_client() -> "DocumentServiceClient":
    global _document_service_client_singleton
    if _document_service_client_singleton is None:
        _document_service_client_singleton = DocumentServiceClient()
    return _document_service_client_singleton


class DocumentServiceClient:
    """Client for Document Service entity extraction via gRPC."""

    def __init__(self, service_url: Optional[str] = None) -> None:
        self.settings = get_settings()
        self.service_url = service_url or self.settings.DOCUMENT_SERVICE_URL
        self.channel: Optional[grpc.aio.Channel] = None
        self.stub: Optional[document_service_pb2_grpc.DocumentServiceStub] = None
        self._initialized = False

    async def initialize(self, required: bool = False) -> None:
        """Initialize the gRPC channel and stub."""
        if self._initialized:
            return
        try:
            logger.debug("Connecting to Document Service at %s", self.service_url)
            # Keepalive at 5 min to avoid GOAWAY "too_many_pings" (ENHANCE_YOUR_CALM) from server/proxy
            options = [
                ("grpc.max_send_message_length", 100 * 1024 * 1024),
                ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                ("grpc.keepalive_time_ms", 300000),  # 5 min
                ("grpc.keepalive_timeout_ms", 20000),
                ("grpc.keepalive_permit_without_calls", 1),
            ]
            self.channel = grpc.aio.insecure_channel(self.service_url, options=options)
            self.stub = document_service_pb2_grpc.DocumentServiceStub(self.channel)
            health_request = document_service_pb2.HealthCheckRequest()
            response = await self.stub.HealthCheck(health_request, timeout=10.0)
            if response.status in ("healthy", "degraded"):
                logger.info(
                    "Connected to Document Service v%s (status=%s, ner_loaded=%s)",
                    response.service_version,
                    response.status,
                    response.gliner_loaded,
                )
                self._initialized = True
            else:
                logger.warning("Document Service health check returned: %s", response.status)
        except Exception as e:
            logger.error("Failed to connect to Document Service: %s", e)
            if required:
                raise
            logger.warning("Document Service unavailable - entity extraction will fail until connected")

    async def close(self) -> None:
        """Close the gRPC channel."""
        if self.channel:
            await self.channel.close()
            self._initialized = False
            logger.info("Document Service client closed")

    async def extract_entities(
        self, text: str, max_length: Optional[int] = None
    ) -> List[Entity]:
        """
        Extract entities from text using the document-service (spaCy).

        Args:
            text: Raw document text.
            max_length: Optional max character length (service default if 0 or None).

        Returns:
            List of Entity models (name, entity_type, confidence, source_chunk, metadata).
        """
        if not self._initialized:
            try:
                await self.initialize(required=True)
            except Exception as e:
                logger.error("Cannot extract entities: Document Service unavailable: %s", e)
                raise RuntimeError("Document Service is not available") from e
        try:
            request = document_service_pb2.ExtractEntitiesRequest(
                text=text,
                max_length=max_length or 0,
            )
            response = await self.stub.ExtractEntities(request, timeout=120.0)
            if not response.success:
                raise RuntimeError(response.error or "ExtractEntities failed")
            return [
                Entity(
                    name=e.name,
                    entity_type=e.entity_type,
                    confidence=e.confidence,
                    source_chunk="",
                    metadata={"source": "spacy", "context": e.context or ""},
                )
                for e in response.entities
            ]
        except grpc.RpcError as e:
            logger.error("Document Service RPC error: %s", e)
            raise RuntimeError(f"Document Service error: {e}") from e

    async def upload_via_document_service(
        self,
        file: UploadFile,
        *,
        doc_type: Optional[str] = None,
        user_id: Optional[str] = None,
        folder_id: Optional[str] = None,
        team_id: Optional[str] = None,
        collection_type: str = "",
        exempt_from_vectorization: Optional[bool] = None,
    ) -> DocumentUploadResponse:
        """Stream upload to document-service UploadAndProcess RPC."""
        if not self._initialized:
            await self.initialize(required=True)

        fn = file.filename or ""
        if exempt_from_vectorization is None:
            exempt = bool(fn.lower().endswith(".json"))
        else:
            exempt = bool(exempt_from_vectorization)

        meta = document_service_pb2.UploadMetadata(
            filename=fn,
            doc_type=doc_type or "",
            user_id=user_id or "",
            folder_id=folder_id or "",
            team_id=team_id or "",
            collection_type=collection_type or "",
            exempt_from_vectorization=exempt,
        )

        async def _chunks() -> AsyncIterator[document_service_pb2.UploadChunk]:
            await file.seek(0)
            chunk_size = 256 * 1024
            first = True
            while True:
                blob = await file.read(chunk_size)
                if first:
                    yield document_service_pb2.UploadChunk(metadata=meta, data=blob)
                    first = False
                    if not blob:
                        break
                else:
                    if not blob:
                        break
                    yield document_service_pb2.UploadChunk(data=blob)

        response = await self.stub.UploadAndProcess(
            _chunks(),
            timeout=3600.0,
        )
        if not response.success:
            raise RuntimeError(response.error or "UploadAndProcess failed")
        try:
            pst = ProcessingStatus(response.status)
        except ValueError:
            pst = ProcessingStatus.PROCESSING
        return DocumentUploadResponse(
            document_id=response.document_id,
            filename=response.filename or fn,
            status=pst,
            message=response.message or "",
        )

    async def reprocess_via_document_service(
        self,
        document_id: str,
        user_id: Optional[str] = None,
        *,
        force_reprocess: bool = False,
    ) -> None:
        """Run full reprocess pipeline in document-service."""
        if not self._initialized:
            await self.initialize(required=True)
        req = document_service_pb2.ReprocessRequest(
            document_id=document_id,
            user_id=user_id or "",
        )
        if hasattr(req, "force_reprocess"):
            setattr(req, "force_reprocess", bool(force_reprocess))
        metadata = ()
        if force_reprocess:
            metadata = (("x-force-reprocess", "true"),)
        resp = await self.stub.ReprocessDocument(
            req, timeout=3600.0, metadata=metadata
        )
        if not resp.success:
            raise RuntimeError(resp.error or "ReprocessDocument failed")

    async def json_tool_call(
        self,
        unary: Callable,
        *,
        user_id: str,
        payload: Dict[str, Any],
        timeout: float = 120.0,
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """Call a DocumentService JsonToolRequest/JsonToolResponse RPC."""
        if not self._initialized:
            await self.initialize(required=True)
        req = document_service_pb2.JsonToolRequest(
            user_id=user_id or "",
            payload_json=json.dumps(payload),
        )
        resp = await unary(req, timeout=timeout)
        if not resp.success:
            return False, None, resp.error or "request failed"
        try:
            return True, json.loads(resp.result_json or "{}"), None
        except json.JSONDecodeError as e:
            return False, None, str(e)

    async def download_document_stream(
        self,
        document_id: str,
        user_id: str,
        *,
        role: str = "",
        timeout: float = 3600.0,
    ) -> AsyncIterator[document_service_pb2.FileDownloadChunk]:
        if not self._initialized:
            await self.initialize(required=True)
        req = document_service_pb2.DownloadDocumentRequest(
            document_id=document_id,
            user_id=user_id or "",
            role=role or "",
        )
        async for chunk in self.stub.DownloadDocument(req, timeout=timeout):
            yield chunk

    async def generate_library_zip_stream(
        self, user_id: str, *, timeout: float = 3600.0
    ) -> AsyncIterator[document_service_pb2.FileDownloadChunk]:
        if not self._initialized:
            await self.initialize(required=True)
        req = document_service_pb2.GenerateLibraryZipRequest(user_id=user_id or "")
        async for chunk in self.stub.GenerateLibraryZip(req, timeout=timeout):
            yield chunk

    async def flush_collaborative_document(
        self,
        document_id: str,
        new_content: str,
        *,
        user_id: str = "",
        role: str = "",
        timeout: float = 120.0,
    ) -> None:
        if not self._initialized:
            await self.initialize(required=True)
        req = document_service_pb2.FlushCollaborativeDocumentRequest(
            document_id=document_id,
            user_id=user_id or "",
            new_content=new_content or "",
            role=role or "",
        )
        resp = await self.stub.FlushCollaborativeDocument(req, timeout=timeout)
        if not resp.success:
            raise RuntimeError(resp.error or "FlushCollaborativeDocument failed")

    async def get_folder_tree_grpc(
        self,
        user_id: str,
        *,
        timeout: float = 120.0,
    ) -> tool_service_pb2.GetFolderTreeResponse:
        if not self._initialized:
            await self.initialize(required=True)
        req = tool_service_pb2.GetFolderTreeRequest(user_id=user_id or "")
        return await self.stub.GetFolderTree(req, timeout=timeout)

    # --- JsonToolRequest RPC wrappers (Phase 2, unconditional DS) ---

    async def place_file_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(self.stub.PlaceFile, user_id=user_id, payload=payload, timeout=timeout)

    async def move_file_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(self.stub.MoveFile, user_id=user_id, payload=payload, timeout=timeout)

    async def delete_file_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(self.stub.DeleteFile, user_id=user_id, payload=payload, timeout=timeout)

    async def rename_file_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(self.stub.RenameFile, user_id=user_id, payload=payload, timeout=timeout)

    async def create_folder_structure_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(
            self.stub.CreateFolderStructure, user_id=user_id, payload=payload, timeout=timeout
        )

    async def create_folder_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(self.stub.CreateFolder, user_id=user_id, payload=payload, timeout=timeout)

    async def update_folder_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(self.stub.UpdateFolder, user_id=user_id, payload=payload, timeout=timeout)

    async def delete_folder_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(self.stub.DeleteFolder, user_id=user_id, payload=payload, timeout=timeout)

    async def move_folder_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(self.stub.MoveFolder, user_id=user_id, payload=payload, timeout=timeout)

    async def get_folder_contents_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(self.stub.GetFolderContents, user_id=user_id, payload=payload, timeout=timeout)

    async def get_folder_contents_batch_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 180.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(
            self.stub.GetFolderContentsBatch, user_id=user_id, payload=payload, timeout=timeout
        )

    async def update_folder_metadata_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(
            self.stub.UpdateFolderMetadata, user_id=user_id, payload=payload, timeout=timeout
        )

    async def exempt_folder_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(self.stub.ExemptFolder, user_id=user_id, payload=payload, timeout=timeout)

    async def remove_folder_exemption_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(
            self.stub.RemoveFolderExemption, user_id=user_id, payload=payload, timeout=timeout
        )

    async def override_folder_exemption_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(
            self.stub.OverrideFolderExemption, user_id=user_id, payload=payload, timeout=timeout
        )

    async def exempt_document_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(
            self.stub.ExemptDocument, user_id=user_id, payload=payload, timeout=timeout
        )

    async def remove_document_exemption_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(
            self.stub.RemoveDocumentExemption, user_id=user_id, payload=payload, timeout=timeout
        )

    async def get_document_versions_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(
            self.stub.GetDocumentVersions, user_id=user_id, payload=payload, timeout=timeout
        )

    async def get_version_content_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(
            self.stub.GetVersionContent, user_id=user_id, payload=payload, timeout=timeout
        )

    async def diff_versions_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(self.stub.DiffVersions, user_id=user_id, payload=payload, timeout=timeout)

    async def rollback_to_version_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(
            self.stub.RollbackToVersion, user_id=user_id, payload=payload, timeout=timeout
        )

    async def encrypt_document_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(self.stub.EncryptDocument, user_id=user_id, payload=payload, timeout=timeout)

    async def create_decrypt_session_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(
            self.stub.CreateDecryptSession, user_id=user_id, payload=payload, timeout=timeout
        )

    async def try_decrypt_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(self.stub.TryDecrypt, user_id=user_id, payload=payload, timeout=timeout)

    async def encryption_heartbeat_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(
            self.stub.EncryptionHeartbeat, user_id=user_id, payload=payload, timeout=timeout
        )

    async def encryption_lock_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(self.stub.EncryptionLock, user_id=user_id, payload=payload, timeout=timeout)

    async def change_encryption_password_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(
            self.stub.ChangeEncryptionPassword, user_id=user_id, payload=payload, timeout=timeout
        )

    async def remove_encryption_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(
            self.stub.RemoveEncryption, user_id=user_id, payload=payload, timeout=timeout
        )

    async def write_encrypted_content_from_session_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(
            self.stub.WriteEncryptedContentFromSession,
            user_id=user_id,
            payload=payload,
            timeout=timeout,
        )

    async def read_upload_relative_file_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(
            self.stub.ReadUploadRelativeFile, user_id=user_id, payload=payload, timeout=timeout
        )

    async def write_upload_relative_file_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(
            self.stub.WriteUploadRelativeFile, user_id=user_id, payload=payload, timeout=timeout
        )

    async def list_upload_relative_dir_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(
            self.stub.ListUploadRelativeDir, user_id=user_id, payload=payload, timeout=timeout
        )

    async def delete_document_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(self.stub.DeleteDocument, user_id=user_id, payload=payload, timeout=timeout)

    async def store_text_document_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 120.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(
            self.stub.StoreTextDocument, user_id=user_id, payload=payload, timeout=timeout
        )

    async def scan_and_recover_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 600.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        return await self.json_tool_call(
            self.stub.ScanAndRecoverFiles, user_id=user_id, payload=payload, timeout=timeout
        )

    async def document_mirror_json(
        self, user_id: str, payload: Dict[str, Any], *, timeout: float = 600.0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """Dispatch to document-service DocumentMirror (metadata, URL reprocess, ZIP helpers)."""
        return await self.json_tool_call(
            self.stub.DocumentMirror, user_id=user_id, payload=payload, timeout=timeout
        )

    async def get_document_grpc(
        self,
        document_id: str,
        user_id: str,
        *,
        timeout: float = 120.0,
    ) -> tool_service_pb2.DocumentResponse:
        if not self._initialized:
            await self.initialize(required=True)
        req = tool_service_pb2.DocumentRequest(document_id=document_id, user_id=user_id or "")
        return await self.stub.GetDocument(req, timeout=timeout)

    async def get_document_content_grpc(
        self,
        document_id: str,
        user_id: str,
        *,
        timeout: float = 120.0,
    ) -> tool_service_pb2.DocumentContentResponse:
        if not self._initialized:
            await self.initialize(required=True)
        req = tool_service_pb2.DocumentRequest(document_id=document_id, user_id=user_id or "")
        return await self.stub.GetDocumentContent(req, timeout=timeout)

    async def update_document_content_grpc(
        self,
        document_id: str,
        user_id: str,
        content: str,
        *,
        append: bool = False,
        write_initiator: str | None = None,
        timeout: float = 120.0,
    ) -> tool_service_pb2.UpdateDocumentContentResponse:
        if not self._initialized:
            await self.initialize(required=True)
        req = tool_service_pb2.UpdateDocumentContentRequest(
            user_id=user_id or "",
            document_id=document_id,
            content=content or "",
            append=append,
        )
        if write_initiator:
            req.write_initiator = write_initiator
        return await self.stub.UpdateDocumentContent(req, timeout=timeout)

    async def update_document_metadata_grpc(
        self,
        request: tool_service_pb2.UpdateDocumentMetadataRequest,
        *,
        timeout: float = 120.0,
    ) -> tool_service_pb2.UpdateDocumentMetadataResponse:
        if not self._initialized:
            await self.initialize(required=True)
        return await self.stub.UpdateDocumentMetadata(request, timeout=timeout)
