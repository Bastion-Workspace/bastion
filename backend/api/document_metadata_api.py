"""
Document Metadata API endpoints.
CRUD for document metadata sidecar files and LLM summary generation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import grpc
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field

from models.api_models import AuthenticatedUserResponse
from services.service_container import get_service_container
from services.settings_service import settings_service
from utils.auth_middleware import get_current_user
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["document-metadata"], prefix="")

try:
    from protos import orchestrator_pb2, orchestrator_pb2_grpc
    ORCHESTRATOR_GRPC_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_GRPC_AVAILABLE = False
    orchestrator_pb2 = None
    orchestrator_pb2_grpc = None

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg",
    ".tiff", ".tif", ".heic", ".heif", ".ico",
}


async def _get_document_service():
    container = await get_service_container()
    return container.document_service


async def _get_folder_service():
    container = await get_service_container()
    return container.folder_service


async def check_document_access(
    doc_id: str,
    current_user: AuthenticatedUserResponse,
    permission: str = "read",
) -> Optional[Any]:
    """Check if user has access to document; return document info or None."""
    document_service = await _get_document_service()
    try:
        doc_info = await document_service.document_repository.get_by_id(doc_id)
        if not doc_info:
            return None
        collection_type = getattr(doc_info, "collection_type", "user")
        user_id = getattr(doc_info, "user_id", None)
        if collection_type == "global":
            if permission in ("write", "admin") and current_user.role != "admin":
                return None
        elif collection_type == "user":
            if user_id != current_user.user_id:
                return None
        return doc_info
    except Exception as e:
        logger.error(f"Error checking document access: {e}", exc_info=True)
        return None


def _is_image_document(filename: str) -> bool:
    if not filename:
        return False
    ext = Path(filename).suffix.lower()
    return ext in IMAGE_EXTENSIONS


class DocumentMetadataRequest(BaseModel):
    """Request model for document metadata sidecar."""
    schema_version: str = Field(default="1.0", description="Schema version")
    document_filename: Optional[str] = Field(None, description="Document filename (optional, derived from doc)")
    summary: Optional[str] = Field(None, description="Summary of the document")
    description: Optional[str] = Field(None, description="Description")
    key_topics: Optional[list] = Field(default_factory=list, description="Key topics")
    tags: Optional[list] = Field(default_factory=list, description="Tags")
    notes: Optional[str] = Field(None, description="Personal notes")
    author: Optional[str] = Field(None, description="Author override")
    custom_fields: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom key-value fields")
    llm_metadata: Optional[Dict[str, Any]] = Field(None, description="LLM generation tracking")


@router.get("/api/documents/{document_id}/doc-metadata")
async def get_document_metadata(
    document_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Get document metadata from sidecar file if it exists; otherwise return empty template."""
    document_service = await _get_document_service()
    folder_service = await _get_folder_service()

    doc_info = await check_document_access(document_id, current_user, "read")
    if not doc_info:
        raise HTTPException(status_code=404, detail="Document not found or access denied")

    filename = getattr(doc_info, "filename", "")
    if _is_image_document(filename):
        raise HTTPException(
            status_code=400,
            detail="Use image-metadata API for image documents",
        )

    user_id = getattr(doc_info, "user_id", None)
    folder_id = getattr(doc_info, "folder_id", None)
    collection_type = getattr(doc_info, "collection_type", "user")
    file_path_str = await folder_service.get_document_file_path(
        filename=filename,
        folder_id=folder_id,
        user_id=user_id,
        collection_type=collection_type,
    )
    doc_path = Path(file_path_str)
    metadata_path = doc_path.parent / f"{doc_path.stem}.metadata.json"

    if not metadata_path.exists():
        return {
            "exists": False,
            "metadata": {
                "schema_version": "1.0",
                "schema_type": "document",
                "document_filename": filename,
                "summary": "",
                "description": "",
                "key_topics": [],
                "tags": [],
                "notes": "",
                "author": "",
                "custom_fields": {},
            },
        }

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read doc metadata: {e}")
        raise HTTPException(status_code=500, detail="Failed to read metadata file")

    return {"exists": True, "metadata": metadata}


@router.post("/api/documents/{document_id}/doc-metadata")
async def create_or_update_document_metadata(
    request: Request,
    document_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Create or update document metadata sidecar file. File watcher will process it."""
    raw_body = await request.json()
    document_service = await _get_document_service()
    folder_service = await _get_folder_service()

    doc_info = await check_document_access(document_id, current_user, "write")
    if not doc_info:
        raise HTTPException(status_code=404, detail="Document not found or access denied")

    filename = getattr(doc_info, "filename", "")
    if _is_image_document(filename):
        raise HTTPException(
            status_code=400,
            detail="Use image-metadata API for image documents",
        )

    user_id = getattr(doc_info, "user_id", None)
    folder_id = getattr(doc_info, "folder_id", None)
    collection_type = getattr(doc_info, "collection_type", "user")
    file_path_str = await folder_service.get_document_file_path(
        filename=filename,
        folder_id=folder_id,
        user_id=user_id,
        collection_type=collection_type,
    )
    doc_path = Path(file_path_str)

    if not doc_path.exists():
        raise HTTPException(status_code=404, detail="Document file not found on disk")

    metadata_path = doc_path.parent / f"{doc_path.stem}.metadata.json"
    metadata_json = {
        "schema_version": raw_body.get("schema_version", "1.0"),
        "schema_type": "document",
        "document_filename": raw_body.get("document_filename") or filename,
        "summary": raw_body.get("summary") or "",
        "description": raw_body.get("description") or "",
        "key_topics": raw_body.get("key_topics") or [],
        "tags": raw_body.get("tags") or [],
        "notes": raw_body.get("notes") or "",
        "author": raw_body.get("author") or "",
        "custom_fields": raw_body.get("custom_fields") or {},
    }
    if raw_body.get("llm_metadata") is not None:
        metadata_json["llm_metadata"] = raw_body["llm_metadata"]

    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_json, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to write doc metadata: {e}")
        raise HTTPException(status_code=500, detail="Failed to write metadata file")

    return {
        "status": "success",
        "message": "Document metadata saved; file watcher will sync to DB.",
        "metadata_path": str(metadata_path),
    }


@router.delete("/api/documents/{document_id}/doc-metadata")
async def delete_document_metadata(
    document_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Delete document metadata sidecar file."""
    document_service = await _get_document_service()
    folder_service = await _get_folder_service()

    doc_info = await check_document_access(document_id, current_user, "write")
    if not doc_info:
        raise HTTPException(status_code=404, detail="Document not found or access denied")

    filename = getattr(doc_info, "filename", "")
    if _is_image_document(filename):
        raise HTTPException(
            status_code=400,
            detail="Use image-metadata API for image documents",
        )

    user_id = getattr(doc_info, "user_id", None)
    folder_id = getattr(doc_info, "folder_id", None)
    collection_type = getattr(doc_info, "collection_type", "user")
    file_path_str = await folder_service.get_document_file_path(
        filename=filename,
        folder_id=folder_id,
        user_id=user_id,
        collection_type=collection_type,
    )
    doc_path = Path(file_path_str)
    metadata_path = doc_path.parent / f"{doc_path.stem}.metadata.json"

    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Metadata file not found")

    try:
        metadata_path.unlink()
    except Exception as e:
        logger.error(f"Failed to delete doc metadata: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete metadata file")

    return {"status": "success", "message": "Document metadata deleted"}


@router.post("/api/documents/{document_id}/generate-summary")
async def generate_document_summary(
    document_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """
    Generate an LLM summary for a document. Does not save; returns summary for user to review and save via doc-metadata.
    """
    if not ORCHESTRATOR_GRPC_AVAILABLE:
        raise HTTPException(status_code=503, detail="Orchestrator gRPC not available")

    document_service = await _get_document_service()
    folder_service = await _get_folder_service()

    doc_info = await check_document_access(document_id, current_user, "read")
    if not doc_info:
        raise HTTPException(status_code=404, detail="Document not found or access denied")

    filename = getattr(doc_info, "filename", "")
    if _is_image_document(filename):
        raise HTTPException(
            status_code=400,
            detail="Use describe-image for image documents",
        )

    user_id = getattr(doc_info, "user_id", None)
    folder_id = getattr(doc_info, "folder_id", None)
    collection_type = getattr(doc_info, "collection_type", "user")
    file_path_str = await folder_service.get_document_file_path(
        filename=filename,
        folder_id=folder_id,
        user_id=user_id,
        collection_type=collection_type,
    )
    file_path = Path(file_path_str)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Document file not found")

    extension = file_path.suffix.lower()
    doc_type_map = {
        ".pdf": "pdf", ".txt": "txt", ".md": "md", ".docx": "docx", ".doc": "docx",
        ".epub": "epub", ".html": "html", ".htm": "html", ".eml": "eml",
        ".zip": "zip", ".srt": "srt",
    }
    doc_type = doc_type_map.get(extension)
    if not doc_type or doc_type == "image":
        raise HTTPException(
            status_code=400,
            detail="Unsupported document type for summarization",
        )

    max_chars = 50000
    content = ""
    try:
        from utils.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        await processor.initialize()
        result = await processor.process_document(
            str(file_path), doc_type, document_id
        )
        if result.chunks:
            parts = [c.content for c in result.chunks]
            content = "\n\n".join(parts)
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n[... truncated ...]"
        else:
            content = ""
    except Exception as e:
        logger.warning(f"Document processor failed for {filename}, falling back to raw read: {e}")
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(max_chars * 2)
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n[... truncated ...]"
        except Exception as read_err:
            logger.error(f"Failed to read document for summary: {read_err}")
            raise HTTPException(status_code=500, detail="Failed to read document content")

    if not content or not content.strip():
        raise HTTPException(
            status_code=400,
            detail="Document has no extractable text to summarize",
        )

    model_used = await settings_service.get_effective_image_analysis_model(
        current_user.user_id
    )
    from services.grpc_context_gatherer import get_context_gatherer
    context_gatherer = get_context_gatherer()
    grpc_request = await context_gatherer.build_chat_request(
        query="Summarize this document.",
        user_id=current_user.user_id,
        conversation_id=f"doc-summary-{document_id}",
        session_id="generate-summary",
        request_context={
            "document_content": content,
            "document_analysis_model": model_used,
        },
        state=None,
        agent_type="document_description",
        routing_reason="document_summary",
    )

    try:
        async with grpc.aio.insecure_channel("llm-orchestrator:50051") as channel:
            stub = orchestrator_pb2_grpc.OrchestratorServiceStub(channel)
            accumulated = ""
            metadata_received = {}
            error_message = None
            async for chunk in stub.StreamChat(grpc_request):
                if chunk.type == "content" and chunk.message:
                    accumulated += chunk.message
                if chunk.metadata:
                    metadata_received.update(dict(chunk.metadata))
                if chunk.type == "error" and chunk.message:
                    error_message = chunk.message
                    break
                if chunk.type == "complete":
                    if chunk.metadata:
                        metadata_received.update(dict(chunk.metadata))
                    break
            if error_message:
                raise HTTPException(status_code=502, detail=error_message)
            summary = metadata_received.get("description") or accumulated
            summary = summary.strip() if summary else ""
            return {
                "success": True,
                "summary": summary,
                "model_used": metadata_received.get("model_used") or model_used,
                "message": "Review and save via POST /api/documents/{id}/doc-metadata",
            }
    except grpc.RpcError as e:
        logger.error(f"Orchestrator gRPC error: {e}")
        raise HTTPException(status_code=502, detail=f"Orchestrator error: {e.details()}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"generate-summary failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
