"""
Image Metadata API endpoints
CRUD operations for image metadata sidecar files
"""

import base64
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import grpc
from fastapi import APIRouter, HTTPException, Depends, Body, Request
from pydantic import BaseModel, Field

from models.api_models import AuthenticatedUserResponse
from services.service_container import get_service_container
from services.user_settings_kv_service import get_user_setting
from services.capabilities_service import capabilities_service
from services.settings_service import settings_service
from services.grpc_context_gatherer import get_context_gatherer
from utils.auth_middleware import get_current_user, require_admin
from config import settings
from clients.tool_service_client import get_tool_service_client
from services.database_manager.database_helpers import fetch_all, fetch_one, execute
from services.face_encoding_service import get_face_encoding_service
from models.object_detection_models import (
    ObjectDetectionRequest,
    AnnotateObjectRequest,
    AddExampleRequest,
    UpdateDetectedObjectRequest,
)

try:
    from protos import orchestrator_pb2, orchestrator_pb2_grpc
    ORCHESTRATOR_GRPC_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_GRPC_AVAILABLE = False
    orchestrator_pb2 = None
    orchestrator_pb2_grpc = None

logger = logging.getLogger(__name__)

router = APIRouter(tags=["image-metadata"], prefix="")

# Test endpoint to verify router is registered
@router.get("/api/test-image-metadata")
async def test_image_metadata_route():
    """Test endpoint to verify image metadata router is registered"""
    return {"status": "ok", "message": "Image metadata router is working"}


class ImageMetadataRequest(BaseModel):
    """Request model for image metadata with universal + type-specific fields"""
    # Universal fields (all image types)
    type: str = Field(..., description="Image type: comic, artwork, meme, screenshot, medical, documentation, maps, photo, other")
    title: Optional[str] = Field(None, description="Image title (optional)")
    content: Optional[str] = Field(None, description="Description/transcript of image contents (optional)")
    author: Optional[str] = Field(None, description="Author/creator name")
    date: Optional[str] = Field(None, description="Date in YYYY-MM-DD format")
    series: Optional[str] = Field(None, description="Series or collection name")
    tags: Optional[list] = Field(default_factory=list, description="List of tags")
    llm_metadata: Optional[Dict[str, Any]] = Field(None, description="LLM description generation tracking (model, timestamp, confidence)")
    faces: Optional[list] = Field(None, description="Detected faces with bbox and identity")
    detected_objects: Optional[list] = Field(None, alias="objects", description="Detected objects with bbox and label")

    # Type-specific optional fields
    location: Optional[str] = Field(None, description="Geographic location or venue (for photos/maps)")
    event: Optional[str] = Field(None, description="Event name (for photos)")
    medium: Optional[str] = Field(None, description="Art medium (for artwork - e.g., 'Oil on canvas')")
    dimensions: Optional[str] = Field(None, description="Physical dimensions (for artwork - e.g., '24x36 inches')")
    body_part: Optional[str] = Field(None, description="Body part (for medical images - e.g., 'chest', 'skull')")
    modality: Optional[str] = Field(None, description="Medical imaging modality (for medical - e.g., 'X-ray', 'MRI', 'CT')")
    map_type: Optional[str] = Field(None, description="Map type (for maps - e.g., 'topographic', 'political')")
    coordinates: Optional[str] = Field(None, description="Geographic coordinates (for maps/photos - e.g., '40.7128,-74.0060')")
    application: Optional[str] = Field(None, description="Application name (for screenshots - e.g., 'VS Code', 'Chrome')")
    platform: Optional[str] = Field(None, description="Platform/OS (for screenshots - e.g., 'Windows 11', 'macOS')")


class FaceTagRequest(BaseModel):
    """Request model for tagging a detected face"""
    identity_name: str = Field(..., description="Name of the person (e.g., 'Steve McQueen')")
    face_encoding: list = Field(..., description="128-dimensional face encoding vector")


async def _get_document_service():
    """Get document service from service container"""
    container = await get_service_container()
    return container.document_service


async def _get_folder_service():
    """Get folder service from service container"""
    container = await get_service_container()
    return container.folder_service


async def check_document_access(doc_id: str, current_user: AuthenticatedUserResponse, permission: str = "read") -> Optional[Any]:
    """
    Check if user has access to document and return document info
    
    Permission: "read", "write", "admin"
    """
    document_service = await _get_document_service()
    try:
        # Get document from repository
        doc_info = await document_service.document_repository.get_by_id(doc_id)
        if not doc_info:
            logger.warning(f"❌ Document not found: {doc_id}")
            return None
        
        logger.info(f"🔍 Document found: {doc_id}, collection_type: {getattr(doc_info, 'collection_type', 'unknown')}, user_id: {getattr(doc_info, 'user_id', None)}")
        
        # Check permissions
        collection_type = getattr(doc_info, 'collection_type', 'user')
        user_id = getattr(doc_info, 'user_id', None)
        
        if collection_type == 'global':
            # Global documents: read for all, write/admin for admins only
            if permission in ['write', 'admin']:
                if current_user.role != 'admin':
                    logger.warning(f"❌ Permission denied: User {current_user.username} (role: {current_user.role}) cannot {permission} global document")
                    return None
                logger.info(f"✅ Admin access granted for global document")
        elif collection_type == 'user':
            # User documents: owner can read/write, others need explicit sharing
            if user_id != current_user.user_id:
                logger.warning(f"❌ Permission denied: User {current_user.user_id} does not own document (owner: {user_id})")
                # Check if shared (simplified - could be enhanced)
                return None
            logger.info(f"✅ Owner access granted for user document")
        
        return doc_info
    except Exception as e:
        logger.error(f"❌ Error checking document access: {e}", exc_info=True)
        return None


def _normalize_objects_for_metadata(objects: list) -> list:
    """Ensure each object has a single canonical label: class_name = user_tag when user_tag is set."""
    if not objects or not isinstance(objects, list):
        return objects
    out = []
    for obj in objects:
        if not isinstance(obj, dict):
            out.append(obj)
            continue
        copy = dict(obj)
        user_tag = (copy.get("user_tag") or "").strip()
        if user_tag:
            copy["class_name"] = user_tag
        out.append(copy)
    return out


@router.post("/api/documents/{document_id}/image-metadata")
async def create_image_metadata(
    request: Request,
    document_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Create or update image metadata sidecar file

    Creates a .metadata.json file alongside the image file.
    File watcher will detect and process it automatically.
    """
    raw_body = await request.json()
    metadata = ImageMetadataRequest.model_validate(raw_body)
    logger.info("POST /api/documents/%s/image-metadata - Route handler called", document_id)
    logger.info("User: %s, Role: %s", current_user.username, current_user.role)
    logger.info("Metadata: type=%s, title=%s...", metadata.type, (metadata.title or "")[:50])
    document_service = await _get_document_service()
    folder_service = await _get_folder_service()
    
    try:
        logger.info(f"📸 Creating image metadata for document_id: {document_id}")
        
        # Check document access
        logger.info(f"🔍 Calling check_document_access for {document_id}")
        doc_info = await check_document_access(document_id, current_user, "write")
        if not doc_info:
            logger.error(f"❌ Document access denied or not found: {document_id}")
            raise HTTPException(status_code=404, detail="Document not found or access denied")
        
        logger.info(f"✅ Document found: {doc_info.filename}, collection_type: {getattr(doc_info, 'collection_type', 'unknown')}")
        
        # Check if document is an image OR if it's a metadata sidecar (we need to find the actual image)
        filename = getattr(doc_info, 'filename', '')
        
        # If this is a .metadata.json file, we need to find the corresponding image
        if filename.endswith('.metadata.json'):
            # Extract image filename from sidecar name
            image_filename = filename[:-14]  # Remove ".metadata.json"
            logger.info(f"📄 Detected metadata sidecar, looking for image: {image_filename}")
            
            # Find the image document by filename
            user_id = getattr(doc_info, 'user_id', None)
            collection_type = getattr(doc_info, 'collection_type', 'user')
            folder_id = getattr(doc_info, 'folder_id', None)
            
            # Use document repository to find the image document
            image_doc = await document_service.document_repository.find_by_filename_and_context(
                filename=image_filename,
                user_id=user_id,
                collection_type=collection_type,
                folder_id=folder_id
            )
            
            if not image_doc:
                logger.warning(f"❌ Image document not found for: {image_filename}")
                raise HTTPException(status_code=404, detail=f"Image file not found: {image_filename}")
            
            # Use the image document for the rest of the operation
            doc_info = image_doc
            filename = image_filename
            logger.info(f"✅ Found image document: {image_doc.document_id}")
        
        # Verify it's an image file
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.tiff', '.tif', '.heic', '.heif']
        if not any(filename.lower().endswith(ext) for ext in image_extensions):
            logger.warning(f"❌ Document is not an image file: {filename}")
            raise HTTPException(status_code=400, detail="Document is not an image file")
        
        # Check permissions: admin for global, owner for personal
        collection_type = getattr(doc_info, 'collection_type', 'user')
        user_id = getattr(doc_info, 'user_id', None)
        
        if collection_type == 'global':
            if current_user.role != 'admin':
                raise HTTPException(status_code=403, detail="Only admins can edit metadata for global images")
        elif collection_type == 'user':
            if user_id != current_user.user_id:
                raise HTTPException(status_code=403, detail="You can only edit metadata for your own images")
        
        # Get image file path
        folder_id = getattr(doc_info, 'folder_id', None)
        file_path_str = await folder_service.get_document_file_path(
            filename=filename,
            folder_id=folder_id,
            user_id=user_id,
            collection_type=collection_type
        )
        image_path = Path(file_path_str)
        
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image file not found")
        
        # Create metadata sidecar path
        # Use stem (filename without extension) to match actual sidecar naming: image.jpg -> image.metadata.json
        metadata_path = image_path.parent / f"{image_path.stem}.metadata.json"
        
        # Build metadata JSON with universal + type-specific schema
        metadata_json = {
            "schema_version": "1.0",
            "image_filename": filename,
            "type": metadata.type,
            "title": metadata.title or "",
            "content": metadata.content or "",
            "author": metadata.author,
            "date": metadata.date,
            "series": metadata.series,
            "tags": metadata.tags or []
        }
        if metadata.llm_metadata is not None:
            metadata_json["llm_metadata"] = metadata.llm_metadata
        if "faces" in raw_body and raw_body["faces"] is not None:
            metadata_json["faces"] = raw_body["faces"]
        elif metadata.faces is not None:
            metadata_json["faces"] = metadata.faces
        if "objects" in raw_body and raw_body["objects"] is not None:
            raw_objects = raw_body["objects"]
            metadata_json["objects"] = _normalize_objects_for_metadata(raw_objects)
        elif metadata.objects is not None:
            metadata_json["objects"] = _normalize_objects_for_metadata(metadata.objects)
            logger.info("image-metadata POST including objects count=%s", len(metadata.objects) if isinstance(metadata.objects, list) else 0)
        else:
            logger.info("image-metadata POST received metadata.objects=None (key missing or null in request body)")

        # Add type-specific fields if provided
        if metadata.location:
            metadata_json["location"] = metadata.location
        if metadata.event:
            metadata_json["event"] = metadata.event
        if metadata.medium:
            metadata_json["medium"] = metadata.medium
        if metadata.dimensions:
            metadata_json["dimensions"] = metadata.dimensions
        if metadata.body_part:
            metadata_json["body_part"] = metadata.body_part
        if metadata.modality:
            metadata_json["modality"] = metadata.modality
        if metadata.map_type:
            metadata_json["map_type"] = metadata.map_type
        if metadata.coordinates:
            metadata_json["coordinates"] = metadata.coordinates
        if metadata.application:
            metadata_json["application"] = metadata.application
        if metadata.platform:
            metadata_json["platform"] = metadata.platform
        
        # Write metadata file
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_json, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Created image metadata sidecar: {metadata_path}")
        
        # File watcher will detect and process the .metadata.json file automatically
        return {
            "status": "success",
            "message": "Image metadata created successfully",
            "metadata_path": str(metadata_path)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to create image metadata: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create image metadata: {str(e)}")


@router.get("/api/documents/{document_id}/image-metadata")
async def get_image_metadata(
    document_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get image metadata from sidecar file if it exists"""
    document_service = await _get_document_service()
    folder_service = await _get_folder_service()
    
    try:
        logger.info(f"📸 Getting image metadata for document_id: {document_id}")
        
        # Check document access
        doc_info = await check_document_access(document_id, current_user, "read")
        if not doc_info:
            logger.warning(f"❌ Document access denied or not found: {document_id}")
            raise HTTPException(status_code=404, detail="Document not found or access denied")
        
        logger.info(f"✅ Document found: {doc_info.filename}")
        
        # Check if document is an image OR if it's a metadata sidecar
        filename = getattr(doc_info, 'filename', '')
        
        # If this is a .metadata.json file, we need to find the corresponding image
        if filename.endswith('.metadata.json'):
            # Extract image filename from sidecar name
            image_filename = filename[:-14]  # Remove ".metadata.json"
            logger.info(f"📄 Detected metadata sidecar, looking for image: {image_filename}")
            
            # Find the image document by filename
            user_id = getattr(doc_info, 'user_id', None)
            collection_type = getattr(doc_info, 'collection_type', 'user')
            folder_id = getattr(doc_info, 'folder_id', None)
            
            # Use document repository to find the image document
            image_doc = await document_service.document_repository.find_by_filename_and_context(
                filename=image_filename,
                user_id=user_id,
                collection_type=collection_type,
                folder_id=folder_id
            )
            
            if not image_doc:
                logger.warning(f"❌ Image document not found for: {image_filename}")
                raise HTTPException(status_code=404, detail=f"Image file not found: {image_filename}")
            
            # Use the image document for the rest of the operation
            doc_info = image_doc
            filename = image_filename
            logger.info(f"✅ Found image document: {image_doc.document_id}")
        
        # Get image file path
        user_id = getattr(doc_info, 'user_id', None)
        folder_id = getattr(doc_info, 'folder_id', None)
        collection_type = getattr(doc_info, 'collection_type', 'user')
        
        file_path_str = await folder_service.get_document_file_path(
            filename=filename,
            folder_id=folder_id,
            user_id=user_id,
            collection_type=collection_type
        )
        image_path = Path(file_path_str)
        
        logger.info(f"📁 Image path: {image_path}")
        
        # Check for metadata sidecar
        # Use stem (filename without extension) to match actual sidecar naming: image.jpg -> image.metadata.json
        metadata_path = image_path.parent / f"{image_path.stem}.metadata.json"
        logger.info(f"🔍 Looking for metadata at: {metadata_path}")
        logger.info(f"📂 Metadata file exists: {metadata_path.exists()}")
        
        # Debug: List files in the directory
        if not metadata_path.exists():
            try:
                files_in_dir = list(image_path.parent.glob("*.metadata.json"))
                logger.info(f"📋 Metadata files in directory: {[f.name for f in files_in_dir]}")
                all_files = list(image_path.parent.glob(f"{image_path.stem}*"))
                logger.info(f"📋 All files with same basename: {[f.name for f in all_files]}")
            except Exception as e:
                logger.warning(f"⚠️ Could not list directory contents: {e}")
            
            # Return empty template
            return {
                "exists": False,
                "metadata": {
                    "type": "other",
                    "title": "",
                    "content": "",
                    "author": "",
                    "date": "",
                    "series": "",
                    "tags": []
                }
            }
        
        # Read and return metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return {
            "exists": True,
            "metadata": metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get image metadata: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get image metadata: {str(e)}")


@router.put("/api/documents/{document_id}/image-metadata")
async def update_image_metadata(
    document_id: str,
    metadata: ImageMetadataRequest = Body(...),
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Update existing image metadata sidecar file"""
    # Same logic as create - just overwrites existing file
    return await create_image_metadata(document_id, metadata, current_user)


@router.delete("/api/documents/{document_id}/image-metadata")
async def delete_image_metadata(
    document_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Delete image metadata sidecar file"""
    document_service = await _get_document_service()
    folder_service = await _get_folder_service()
    
    try:
        # Check document access
        doc_info = await check_document_access(document_id, current_user, "write")
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check permissions: admin for global, owner for personal
        collection_type = getattr(doc_info, 'collection_type', 'user')
        user_id = getattr(doc_info, 'user_id', None)
        
        if collection_type == 'global':
            if current_user.role != 'admin':
                raise HTTPException(status_code=403, detail="Only admins can delete metadata for global images")
        elif collection_type == 'user':
            if user_id != current_user.user_id:
                raise HTTPException(status_code=403, detail="You can only delete metadata for your own images")
        
        # Get image file path
        filename = getattr(doc_info, 'filename', '')
        folder_id = getattr(doc_info, 'folder_id', None)
        file_path_str = await folder_service.get_document_file_path(
            filename=filename,
            folder_id=folder_id,
            user_id=user_id,
            collection_type=collection_type
        )
        image_path = Path(file_path_str)
        
        # Delete metadata sidecar
        # Use stem (filename without extension) to match actual sidecar naming: image.jpg -> image.metadata.json
        metadata_path = image_path.parent / f"{image_path.stem}.metadata.json"
        
        if metadata_path.exists():
            metadata_path.unlink()
            logger.info(f"✅ Deleted image metadata sidecar: {metadata_path}")
            
            # TODO: Optionally delete the document record and embeddings for the metadata
            # For now, just delete the file - the document record can remain
            
            return {
                "status": "success",
                "message": "Image metadata deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Metadata file not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete image metadata: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete image metadata: {str(e)}")


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


@router.post("/api/documents/{document_id}/describe-image")
async def describe_image_llm(
    document_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Generate LLM description for an image document.
    Does not save to metadata; UI shows diff/merge and user saves.
    """
    if not ORCHESTRATOR_GRPC_AVAILABLE:
        raise HTTPException(status_code=503, detail="Orchestrator gRPC not available")

    user_dict = {"user_id": current_user.user_id, "role": getattr(current_user, "role", "")}
    has_cap = await capabilities_service.user_has_feature(user_dict, "feature.image.llm_description")
    if not has_cap:
        raise HTTPException(status_code=403, detail="Image LLM description not enabled for your account")

    document_service = await _get_document_service()
    folder_service = await _get_folder_service()

    doc_info = await check_document_access(document_id, current_user, "read")
    if not doc_info:
        raise HTTPException(status_code=404, detail="Document not found or access denied")

    filename = getattr(doc_info, "filename", "")
    if not filename:
        raise HTTPException(status_code=400, detail="Document has no filename")
    path_lower = filename.lower()
    if not any(path_lower.endswith(ext) for ext in IMAGE_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Document is not an image file")

    user_id = getattr(doc_info, "user_id", None)
    folder_id = getattr(doc_info, "folder_id", None)
    collection_type = getattr(doc_info, "collection_type", "user")
    file_path_str = await folder_service.get_document_file_path(
        filename=filename,
        folder_id=folder_id,
        user_id=user_id,
        collection_type=collection_type
    )
    image_path = Path(file_path_str)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to read image: {e}")
        raise HTTPException(status_code=500, detail="Failed to read image file")

    model_used = await settings_service.get_effective_image_analysis_model(current_user.user_id)
    request_context = {
        "image_base64": image_base64,
        "image_analysis_model": model_used,
    }
    query = "Describe this image."
    conversation_id = f"image-desc-{document_id}"
    context_gatherer = get_context_gatherer()
    grpc_request = await context_gatherer.build_chat_request(
        query=query,
        user_id=current_user.user_id,
        conversation_id=conversation_id,
        session_id="describe-image",
        request_context=request_context,
        state=None,
        agent_type="image_description",
        routing_reason="image_description",
    )

    options = [
        ("grpc.max_send_message_length", 100 * 1024 * 1024),
        ("grpc.max_receive_message_length", 100 * 1024 * 1024),
    ]
    try:
        async with grpc.aio.insecure_channel("llm-orchestrator:50051", options=options) as channel:
            stub = orchestrator_pb2_grpc.OrchestratorServiceStub(channel)
            accumulated_content = ""
            metadata_received = {}
            error_message = None
            async for chunk in stub.StreamChat(grpc_request):
                if chunk.type == "content" and chunk.message:
                    accumulated_content += chunk.message
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

            description = metadata_received.get("description") or accumulated_content
            detected_text = metadata_received.get("detected_text", "")
            confidence = metadata_received.get("confidence", "0")
            try:
                confidence_float = float(confidence)
            except (TypeError, ValueError):
                confidence_float = 0.92
            model_from_meta = metadata_received.get("model_used") or model_used

            return {
                "success": True,
                "description": description.strip() if description else "",
                "detected_text": (detected_text or "").strip(),
                "model_used": model_from_meta,
                "confidence": confidence_float,
            }
    except grpc.RpcError as e:
        logger.error(f"Orchestrator gRPC error: {e}")
        raise HTTPException(status_code=502, detail=f"Orchestrator error: {e.details()}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"describe_image_llm failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _check_vision_features_enabled(user_id: str, user_role: str = None) -> bool:
    """Check if user has vision features enabled
    
    Args:
        user_id: User ID
        user_role: User role (if admin, always returns True)
    """
    # Admins always have access
    if user_role == "admin":
        return True
    
    try:
        value = await get_user_setting(user_id, "enable_vision_features")
        return value == "true"
    except Exception as e:
        logger.warning(f"Failed to check vision features setting: {e}")
        return False


async def _generate_metadata_suggestions(
    document_id: str,
    faces: list,
    filename: str,
    user_id: str,
    collection_type: str
) -> Optional[Dict[str, Any]]:
    """
    Generate metadata suggestions based on detected faces
    
    Args:
        document_id: Document ID
        faces: List of detected faces with identity information
        filename: Image filename
        user_id: User ID
        collection_type: Collection type ('user' or 'global')
        
    Returns:
        Dict with suggested metadata fields, or None if no suggestions
    """
    try:
        # Get document info directly (we already have access from analyze_faces)
        folder_service = await _get_folder_service()
        document_service = await _get_document_service()
        doc_info = await document_service.document_repository.get_by_id(document_id)
        if not doc_info:
            return None
        
        folder_id = getattr(doc_info, 'folder_id', None)
        file_path_str = await folder_service.get_document_file_path(
            filename=filename,
            folder_id=folder_id,
            user_id=user_id,
            collection_type=collection_type
        )
        image_path = Path(file_path_str)
        # Use stem (filename without extension) to match actual sidecar naming: image.jpg -> image.metadata.json
        metadata_path = image_path.parent / f"{image_path.stem}.metadata.json"
        
        # Load existing metadata if it exists
        existing_metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    existing_metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load existing metadata for suggestions: {e}")
        
        # Generate suggestions based on detected faces
        suggestions = {}
        
        # Suggest type="photo" if faces detected and type not already set
        if existing_metadata.get("type") in [None, "", "other"]:
            suggestions["type"] = "photo"
        
        # Suggest title based on filename if no title exists
        if not existing_metadata.get("title"):
            # Use filename without extension as title suggestion
            title_suggestion = Path(filename).stem
            # Capitalize first letter of each word
            title_suggestion = " ".join(word.capitalize() for word in title_suggestion.replace("_", " ").replace("-", " ").split())
            suggestions["title"] = title_suggestion
        
        # Suggest content/description based on detected identities
        identified_faces = [face for face in faces if face.get("identity_name")]
        suggested_identities = [face.get("suggested_identity") for face in faces if face.get("suggested_identity")]
        
        if not existing_metadata.get("content"):
            content_parts = []
            
            if identified_faces:
                identity_names = [face["identity_name"] for face in identified_faces]
                if len(identity_names) == 1:
                    content_parts.append(f"Photo of {identity_names[0]}")
                else:
                    content_parts.append(f"Photo of {', '.join(identity_names[:-1])} and {identity_names[-1]}")
            elif suggested_identities:
                # Use suggested identities if no confirmed ones
                if len(suggested_identities) == 1:
                    content_parts.append(f"Photo possibly showing {suggested_identities[0]}")
                else:
                    content_parts.append(f"Photo possibly showing {', '.join(suggested_identities)}")
            else:
                # Generic description based on number of faces
                face_count = len(faces)
                if face_count == 1:
                    content_parts.append("Photo with one person")
                else:
                    content_parts.append(f"Photo with {face_count} people")
            
            if content_parts:
                suggestions["content"] = ". ".join(content_parts) + "."
        
        # Suggest tags based on identified faces (if not already in tags)
        existing_tags = existing_metadata.get("tags", [])
        if not isinstance(existing_tags, list):
            existing_tags = []
        
        existing_tags_lower = [tag.lower().strip() if isinstance(tag, str) else str(tag).lower().strip() for tag in existing_tags]
        
        suggested_tags = []
        # Add confirmed identities as tags
        for face in identified_faces:
            identity_name = face.get("identity_name")
            if identity_name and identity_name.lower().strip() not in existing_tags_lower:
                suggested_tags.append(identity_name)
        
        # Add high-confidence suggested identities as tags
        for face in faces:
            suggested_identity = face.get("suggested_identity")
            suggested_confidence = face.get("suggested_confidence", 0)
            if suggested_identity and suggested_confidence >= 80.0:
                if suggested_identity.lower().strip() not in existing_tags_lower:
                    suggested_tags.append(suggested_identity)
        
        if suggested_tags:
            suggestions["tags"] = suggested_tags
        
        # Only return suggestions if we have any
        if suggestions:
            return {
                "suggested_metadata": suggestions,
                "reason": "Faces detected in image",
                "can_apply": True  # Frontend can use this to offer "Apply suggestions" button
            }
        
        return None
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to generate metadata suggestions: {e}")
        return None


async def _store_detected_faces(document_id: str, faces: list):
    """Store detected faces in database"""
    try:
        # Delete existing faces for this document (re-analysis replaces old results)
        await execute(
            "DELETE FROM detected_faces WHERE document_id = $1",
            document_id
        )
        
        # Insert new faces
        for face in faces:
            await execute(
                """
                INSERT INTO detected_faces 
                (document_id, bbox_x, bbox_y, bbox_width, bbox_height, face_encoding, confidence)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                document_id,
                face["bbox_x"],
                face["bbox_y"],
                face["bbox_width"],
                face["bbox_height"],
                face["face_encoding"],
                face.get("confidence", 1.0)
            )
        logger.info(f"✅ Stored {len(faces)} detected faces for document {document_id}")
    except Exception as e:
        logger.error(f"❌ Failed to store detected faces: {e}")
        raise


async def _fetch_detected_faces(document_id: str) -> list:
    """Fetch detected faces for a document"""
    try:
        rows = await fetch_all(
            """
            SELECT id, bbox_x, bbox_y, bbox_width, bbox_height, 
                   face_encoding, identity_name, identity_confirmed, confidence, 
                   tagged_by, tagged_at
            FROM detected_faces
            WHERE document_id = $1
            ORDER BY id
            """,
            document_id
        )
        return [
            {
                "id": row["id"],
                "bbox_x": row["bbox_x"],
                "bbox_y": row["bbox_y"],
                "bbox_width": row["bbox_width"],
                "bbox_height": row["bbox_height"],
                "face_encoding": row["face_encoding"],
                "identity_name": row["identity_name"],
                "identity_confirmed": row["identity_confirmed"],
                "confidence": row["confidence"],
                "tagged_by": row["tagged_by"],
                "tagged_at": str(row["tagged_at"]) if row["tagged_at"] else None
            }
            for row in rows
        ]
    except Exception as e:
        logger.error(f"❌ Failed to fetch detected faces: {e}")
        return []


async def _update_face_identity(face_id: int, identity_name: str, user_id: str):
    """Update face identity in detected_faces table"""
    try:
        await execute(
            """
            UPDATE detected_faces
            SET identity_name = $1,
                identity_confirmed = true,
                tagged_by = $2,
                tagged_at = NOW()
            WHERE id = $3
            """,
            identity_name,
            user_id,
            face_id
        )
        logger.info(f"✅ Updated face {face_id} with identity '{identity_name}'")
    except Exception as e:
        logger.error(f"❌ Failed to update face identity: {e}")
        raise


async def _get_face_encoding(face_id: int) -> Optional[list]:
    """Get face encoding from database"""
    try:
        row = await fetch_one(
            "SELECT face_encoding FROM detected_faces WHERE id = $1",
            face_id
        )
        if row:
            return row["face_encoding"]
        return None
    except Exception as e:
        logger.error(f"❌ Failed to get face encoding: {e}")
        return None


async def _update_known_identity(identity_name: str, face_encoding: list, created_by: Optional[str] = None, source_document_id: Optional[str] = None):
    """
    Update or create known identity using Qdrant vector storage
    Each face encoding is stored separately - more samples = better matching!
    """
    try:
        # Get face encoding service
        face_service = await get_face_encoding_service()
        
        # Add encoding to Qdrant (global and per-user collection when created_by is set)
        point_id = await face_service.add_face_encoding(
            identity_name=identity_name,
            face_encoding=face_encoding,
            source_document_id=source_document_id or "unknown",
            metadata={"created_by": created_by} if created_by else {},
            user_id=created_by,
        )
        
        # Update or create metadata record in PostgreSQL
        existing = await fetch_one(
            "SELECT id, sample_count FROM known_identities WHERE identity_name = $1",
            identity_name
        )
        
        if existing:
            # Increment sample count
            sample_count = existing["sample_count"]
            await execute(
                """
                UPDATE known_identities
                SET sample_count = $1,
                    updated_at = NOW()
                WHERE identity_name = $2
                """,
                sample_count + 1,
                identity_name
            )
            logger.info(f"✅ Updated known identity '{identity_name}' (sample_count: {sample_count + 1}, point: {point_id})")
        else:
            # Create new identity metadata record
            # face_encoding column gets empty array (actual encodings in Qdrant)
            await execute(
                """
                INSERT INTO known_identities (identity_name, face_encoding, sample_count, created_by, created_at, updated_at)
                VALUES ($1, $2, 1, $3, NOW(), NOW())
                """,
                identity_name,
                [],  # Empty array for PostgreSQL (actual encodings in Qdrant)
                created_by
            )
            logger.info(f"✅ Created new known identity '{identity_name}' (point: {point_id})")
            
    except Exception as e:
        logger.error(f"❌ Failed to update known identity: {e}")
        raise


async def _find_similar_faces(face_encoding: list, threshold: float = 0.82, exclude_document: Optional[str] = None) -> list:
    """Find similar faces across all images using face encoding distance"""
    try:
        try:
            import face_recognition
        except ImportError:
            logger.warning("face_recognition library not available for similarity search")
            return []
        
        query_encoding = np.array(face_encoding)
        
        # Get all detected faces (excluding the current document if specified)
        if exclude_document:
            rows = await fetch_all(
                """
                SELECT df.id, df.document_id, df.face_encoding, df.identity_name,
                       dm.title, dm.filename
                FROM detected_faces df
                JOIN document_metadata dm ON df.document_id = dm.document_id
                WHERE df.document_id != $1
                """,
                exclude_document
            )
        else:
            rows = await fetch_all(
                """
                SELECT df.id, df.document_id, df.face_encoding, df.identity_name,
                       dm.title, dm.filename
                FROM detected_faces df
                JOIN document_metadata dm ON df.document_id = dm.document_id
                """
            )
        
        similar_faces = []
        for row in rows:
            stored_encoding = np.array(row["face_encoding"])
            
            # Calculate face distance (lower = more similar)
            distance = face_recognition.face_distance([stored_encoding], query_encoding)[0]
            
            # Convert distance to confidence percentage
            # face_recognition uses 0.6 as typical threshold for "same person"
            confidence = max(0, (1 - (distance / 0.6)) * 100)
            
            if confidence >= (threshold * 100):
                similar_faces.append({
                    "face_id": row["id"],
                    "document_id": row["document_id"],
                    "title": row["title"],
                    "filename": row["filename"],
                    "identity_name": row["identity_name"],
                    "confidence": round(confidence, 1)
                })
        
        # Sort by confidence descending
        similar_faces.sort(key=lambda x: x["confidence"], reverse=True)
        
        return similar_faces
        
    except Exception as e:
        logger.error(f"❌ Failed to find similar faces: {e}")
        return []


async def _merge_identity_suggestions(
    stored_faces: list,
    identified_faces: list
) -> list:
    """
    Merge identity suggestions from Tools Service with stored faces
    
    Matches identified faces to stored faces by bounding box proximity
    and adds suggested_identity and suggested_confidence to stored_faces.
    
    Args:
        stored_faces: List of faces from database (with IDs)
        identified_faces: List of identified faces from Tools Service (with identity_name, confidence, bbox)
        
    Returns:
        List of stored faces with identity suggestions added
    """
    try:
        # Match identified faces to stored faces by bounding box proximity
        for stored_face in stored_faces:
            # Skip already tagged faces
            if stored_face.get("identity_name"):
                continue
            
            stored_bbox = {
                "x": stored_face.get("bbox_x", 0),
                "y": stored_face.get("bbox_y", 0),
                "width": stored_face.get("bbox_width", 0),
                "height": stored_face.get("bbox_height", 0)
            }
            
            # Find best matching identified face by bounding box overlap
            best_match = None
            best_overlap = 0.0
            
            for identified_face in identified_faces:
                identified_bbox = {
                    "x": identified_face.get("bbox_x", 0),
                    "y": identified_face.get("bbox_y", 0),
                    "width": identified_face.get("bbox_width", 0),
                    "height": identified_face.get("bbox_height", 0)
                }
                
                # Calculate bounding box overlap (IoU - Intersection over Union)
                overlap = _calculate_bbox_overlap(stored_bbox, identified_bbox)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = identified_face
            
            # If we found a good match (overlap > 0.5), add suggestion
            if best_match and best_overlap > 0.5:
                stored_face["suggested_identity"] = best_match.get("identity_name", "")
                stored_face["suggested_confidence"] = int(best_match.get("confidence", 0.0) * 100)
                logger.info(f"✨ Matched stored face to '{best_match.get('identity_name')}' ({best_match.get('confidence', 0.0):.1%} confidence, {best_overlap:.1%} overlap)")
        
        return stored_faces
        
    except Exception as e:
        logger.error(f"❌ Failed to merge identity suggestions: {e}")
        return stored_faces


def _calculate_bbox_overlap(bbox1: Dict[str, int], bbox2: Dict[str, int]) -> float:
    """
    Calculate Intersection over Union (IoU) for two bounding boxes
    
    Args:
        bbox1: First bounding box {x, y, width, height}
        bbox2: Second bounding box {x, y, width, height}
        
    Returns:
        IoU value between 0.0 and 1.0
    """
    # Calculate intersection rectangle
    x1 = max(bbox1["x"], bbox2["x"])
    y1 = max(bbox1["y"], bbox2["y"])
    x2 = min(bbox1["x"] + bbox1["width"], bbox2["x"] + bbox2["width"])
    y2 = min(bbox1["y"] + bbox1["height"], bbox2["y"] + bbox2["height"])
    
    # No intersection
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    bbox1_area = bbox1["width"] * bbox1["height"]
    bbox2_area = bbox2["width"] * bbox2["height"]
    union_area = bbox1_area + bbox2_area - intersection_area
    
    # Avoid division by zero
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


@router.post("/api/documents/{document_id}/analyze-faces")
async def analyze_image_faces(
    document_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Manually trigger face detection on an image
    Requires user setting 'enable_vision_features' = true
    """
    try:
        # 1. Check user opt-in (admins bypass this check)
        vision_enabled = await _check_vision_features_enabled(current_user.user_id, current_user.role)
        if not vision_enabled:
            raise HTTPException(status_code=403, detail="Vision features disabled for user. Enable in settings.")
        
        # 2. Check document access
        doc_info = await check_document_access(document_id, current_user, "read")
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found or access denied")
        
        # 3. Get image file path
        folder_service = await _get_folder_service()
        filename = getattr(doc_info, 'filename', '')
        user_id = getattr(doc_info, 'user_id', None)
        folder_id = getattr(doc_info, 'folder_id', None)
        collection_type = getattr(doc_info, 'collection_type', 'user')
        
        file_path_str = await folder_service.get_document_file_path(
            filename=filename,
            folder_id=folder_id,
            user_id=user_id,
            collection_type=collection_type
        )
        image_path = Path(file_path_str)
        
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image file not found")
        
        # 4. Call Tools Service for face detection (graceful degradation)
        try:
            tool_client = await get_tool_service_client()
            detection_result = await tool_client.detect_faces(
                attachment_path=str(image_path),
                user_id=user_id or current_user.user_id
            )
            
            if not detection_result.get("success"):
                error_msg = detection_result.get("error", "Face detection failed")
                logger.error(f"Face detection failed: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "faces": []
                }
            
            # Convert Tools Service response format to expected format
            faces_list = []
            for face in detection_result.get("faces", []):
                faces_list.append({
                    "bbox_x": face.get("bbox_x", 0),
                    "bbox_y": face.get("bbox_y", 0),
                    "bbox_width": face.get("bbox_width", 0),
                    "bbox_height": face.get("bbox_height", 0),
                    "face_encoding": face.get("face_encoding", []),
                    "confidence": 1.0
                })
            
            result = {
                "faces": faces_list,
                "processing_time_seconds": 0,  # Tools Service doesn't return this
                "image_width": detection_result.get("image_width"),
                "image_height": detection_result.get("image_height")
            }
            
        except Exception as e:
            logger.error(f"Tools Service unavailable: {e}")
            return {
                "success": False,
                "error": "Face analysis service unavailable",
                "faces": []
            }
        
        # 5. Store detected faces in database
        await _store_detected_faces(document_id, result["faces"])
        
        # 6. Reload faces from database to get IDs
        stored_faces = await _fetch_detected_faces(document_id)
        
        # 7. Match faces against known identities for auto-suggestions using Tools Service
        # High-confidence matches are suggested but NOT auto-synced until user confirms
        try:
            identification_result = await tool_client.identify_faces(
                attachment_path=str(image_path),
                user_id=user_id or current_user.user_id,
                confidence_threshold=0.82  # Align with L2 < 0.6 same-person rule (cosine >= 0.82)
            )
            
            # Merge identity suggestions with stored faces by matching bounding boxes
            faces_with_suggestions = await _merge_identity_suggestions(
                stored_faces=stored_faces,
                identified_faces=identification_result.get("identified_faces", []) if identification_result.get("success") else []
            )
        except Exception as e:
            logger.warning(f"Face identification failed, continuing without suggestions: {e}")
            # Continue without identity suggestions if identification fails
            faces_with_suggestions = stored_faces
        
        # Convert to response format (exclude face_encoding - too large for API response)
        faces_response = []
        for face in faces_with_suggestions:
            face_data = {
                "id": face["id"],
                "bbox_x": face["bbox_x"],
                "bbox_y": face["bbox_y"],
                "bbox_width": face["bbox_width"],
                "bbox_height": face["bbox_height"],
                "identity_name": face["identity_name"],
                "identity_confirmed": face["identity_confirmed"],
                "confidence": face.get("confidence", 1.0)
            }
            
            # Add suggested identity if found
            if face.get("suggested_identity"):
                face_data["suggested_identity"] = face["suggested_identity"]
                face_data["suggested_confidence"] = face.get("suggested_confidence", 0)
                logger.info(f"   📤 Returning suggestion: {face_data['suggested_identity']} ({face_data['suggested_confidence']}%)")
            
            faces_response.append(face_data)
        
        # 9. Generate metadata suggestions when faces are detected
        metadata_suggestions = None
        if len(faces_response) > 0:
            metadata_suggestions = await _generate_metadata_suggestions(
                document_id=document_id,
                faces=faces_response,
                filename=filename,
                user_id=user_id or current_user.user_id,
                collection_type=collection_type
            )
        
        response = {
            "success": True,
            "faces": faces_response,
            "processing_time": result["processing_time_seconds"],
            "image_width": result["image_width"],
            "image_height": result["image_height"]
        }
        
        # Add metadata suggestions if available
        if metadata_suggestions:
            response["metadata_suggestions"] = metadata_suggestions
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to analyze faces: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze faces: {str(e)}")


@router.put("/api/documents/{document_id}/faces/{face_id}/tag")
async def tag_detected_face(
    document_id: str,
    face_id: int,
    request: FaceTagRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    User tags a detected face with identity name
    Updates known_identities table for future matching
    """
    try:
        # Check document access
        doc_info = await check_document_access(document_id, current_user, "write")
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found or access denied")
        
        # Verify face belongs to this document and get encoding
        face_row = await fetch_one(
            "SELECT id, face_encoding FROM detected_faces WHERE id = $1 AND document_id = $2",
            face_id,
            document_id
        )
        if not face_row:
            raise HTTPException(status_code=404, detail="Face not found in this document")
        
        # Get face encoding from database if not provided
        face_encoding = request.face_encoding
        if not face_encoding or len(face_encoding) == 0:
            face_encoding = face_row.get("face_encoding")
            if not face_encoding:
                raise HTTPException(status_code=400, detail="Face encoding not found")
        
        # Update detected_faces table
        await _update_face_identity(face_id, request.identity_name, current_user.user_id)
        
        # Update or create known_identities entry with Qdrant storage
        await _update_known_identity(
            identity_name=request.identity_name,
            face_encoding=face_encoding,
            created_by=current_user.user_id,
            source_document_id=document_id
        )
        
        # Auto-sync identity to metadata tags (makes it searchable)
        tags_auto_synced = False
        try:
            collection_type = getattr(doc_info, 'collection_type', 'user')
            user_id = getattr(doc_info, 'user_id', None) or current_user.user_id
            
            from services.image_sidecar_service import get_image_sidecar_service
            sidecar_service = await get_image_sidecar_service()
            
            sync_result = await sidecar_service.sync_identity_to_tags(
                document_id=document_id,
                identity_name=request.identity_name,
                user_id=user_id,
                collection_type=collection_type
            )
            
            tags_auto_synced = sync_result.get("success", False)
            
            if tags_auto_synced:
                logger.info(f"✅ Auto-synced tagged identity '{request.identity_name}' to metadata tags")
            else:
                logger.warning(f"⚠️ Failed to sync identity to tags: {sync_result.get('error')}")
        except Exception as sync_error:
            logger.warning(f"⚠️ Failed to auto-sync identity to tags: {sync_error}")
            # Continue - this is not critical, just a convenience feature
        
        # Find similar faces for suggestions (currently disabled - we use auto-matching instead)
        similar = []
        
        return {
            "success": True,
            "similar_faces_found": len(similar),
            "suggestions": similar[:5],  # Top 5 matches
            "tags_auto_synced": tags_auto_synced  # Indicates tags were automatically added
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to tag face: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to tag face: {str(e)}")


@router.get("/api/documents/{document_id}/faces")
async def get_detected_faces(
    document_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get all detected faces for an image"""
    try:
        # Check document access
        doc_info = await check_document_access(document_id, current_user, "read")
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found or access denied")
        
        # Query detected_faces table
        faces = await _fetch_detected_faces(document_id)
        
        # Strip face_encoding from response (too large for API, only needed internally)
        faces_response = []
        for face in faces:
            face_copy = face.copy()
            face_copy.pop("face_encoding", None)  # Remove encoding
            faces_response.append(face_copy)
        
        return {
            "success": True,
            "faces": faces_response
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get detected faces: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get detected faces: {str(e)}")


# ========== Object Detection Endpoints ==========


@router.post("/api/documents/{document_id}/detect-objects")
async def detect_objects(
    document_id: str,
    request: ObjectDetectionRequest = Body(default=None),
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Run object detection (YOLO + optional CLIP + user-defined matching) on an image."""
    try:
        doc_info = await check_document_access(document_id, current_user, "read")
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found or access denied")

        folder_service = await _get_folder_service()
        filename = getattr(doc_info, "filename", "")
        user_id = getattr(doc_info, "user_id", None) or current_user.user_id
        folder_id = getattr(doc_info, "folder_id", None)
        collection_type = getattr(doc_info, "collection_type", "user")

        file_path_str = await folder_service.get_document_file_path(
            filename=filename,
            folder_id=folder_id,
            user_id=user_id,
            collection_type=collection_type,
        )
        image_path = Path(file_path_str)
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image file not found")

        from services.object_detection_service import get_object_detection_service

        obj_service = await get_object_detection_service()
        opts = request or ObjectDetectionRequest()
        result = await obj_service.detect_objects_in_image(
            document_id=document_id,
            image_path=str(image_path),
            user_id=user_id,
            class_filter=opts.class_filter,
            confidence_threshold=opts.confidence_threshold,
            semantic_descriptions=opts.semantic_descriptions,
            match_user_annotations=opts.match_user_annotations,
        )

        if result.get("error"):
            return {
                "success": False,
                "error": result["error"],
                "objects": [],
            }

        await obj_service.process_detection_results(document_id, result["objects"], user_id=user_id)
        stored = await obj_service.get_detected_objects(document_id)

        return {
            "success": True,
            "objects": stored,
            "image_width": result.get("image_width"),
            "image_height": result.get("image_height"),
            "processing_time_seconds": result.get("processing_time_seconds"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Object detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/documents/{document_id}/annotate-object")
async def create_object_annotation(
    document_id: str,
    request: AnnotateObjectRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Create a user-defined object annotation (draw bbox + description); store embeddings in Qdrant."""
    try:
        doc_info = await check_document_access(document_id, current_user, "write")
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found or access denied")

        folder_service = await _get_folder_service()
        filename = getattr(doc_info, "filename", "")
        user_id = getattr(doc_info, "user_id", None) or current_user.user_id
        folder_id = getattr(doc_info, "folder_id", None)
        collection_type = getattr(doc_info, "collection_type", "user")

        file_path_str = await folder_service.get_document_file_path(
            filename=filename,
            folder_id=folder_id,
            user_id=user_id,
            collection_type=collection_type,
        )
        image_path = Path(file_path_str)
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image file not found")

        from clients.image_vision_client import get_image_vision_client
        from services.object_encoding_service import get_object_encoding_service

        vision_client = await get_image_vision_client()
        await vision_client.initialize(required=False)
        if not vision_client._initialized:
            raise HTTPException(status_code=503, detail="Image Vision Service unavailable")

        bbox = {
            "x": request.bbox.x,
            "y": request.bbox.y,
            "width": request.bbox.width,
            "height": request.bbox.height,
        }
        features = await vision_client.extract_object_features(
            image_path=str(image_path),
            bbox=bbox,
            description=request.description or request.object_name,
        )
        if not features or not features.get("combined_embedding"):
            raise HTTPException(status_code=500, detail="Failed to extract object features")

        obj_enc = await get_object_encoding_service()

        # If user already has an annotation with this name, add this bbox as another example
        existing = await fetch_one(
            "SELECT id, user_id, object_name, description, source_document_id, bbox_x, bbox_y, bbox_width, bbox_height, created_at, example_count FROM user_object_annotations WHERE user_id = $1 AND object_name = $2",
            current_user.user_id,
            request.object_name.strip(),
        )
        if existing:
            annotation_id = existing["id"]
            point_id = await obj_enc.add_annotation_example(
                annotation_id=str(annotation_id),
                user_id=current_user.user_id,
                object_name=existing["object_name"],
                combined_embedding=features["combined_embedding"],
                source_document_id=document_id,
            )
            await execute(
                """
                INSERT INTO object_annotation_examples
                (annotation_id, source_document_id, bbox_x, bbox_y, bbox_width, bbox_height, combined_embedding_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                annotation_id,
                document_id,
                request.bbox.x,
                request.bbox.y,
                request.bbox.width,
                request.bbox.height,
                point_id,
            )
            await execute(
                "UPDATE user_object_annotations SET example_count = example_count + 1, updated_at = CURRENT_TIMESTAMP WHERE id = $1",
                annotation_id,
            )
            row = await fetch_one(
                "SELECT id, user_id, object_name, description, source_document_id, bbox_x, bbox_y, bbox_width, bbox_height, created_at, example_count FROM user_object_annotations WHERE id = $1",
                annotation_id,
            )
            return {
                "success": True,
                "annotation_id": annotation_id,
                "annotation": dict(row),
                "added_as_example": True,
            }

        row = await fetch_one(
            """
            INSERT INTO user_object_annotations
            (user_id, object_name, description, source_document_id, bbox_x, bbox_y, bbox_width, bbox_height,
             visual_embedding_id, combined_embedding_id, example_count)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 1)
            RETURNING id, user_id, object_name, description, source_document_id, bbox_x, bbox_y, bbox_width, bbox_height, created_at
            """,
            current_user.user_id,
            request.object_name.strip(),
            request.description or "",
            document_id,
            request.bbox.x,
            request.bbox.y,
            request.bbox.width,
            request.bbox.height,
            "",
            "",
        )
        if not row:
            raise HTTPException(status_code=500, detail="Failed to create annotation row")

        annotation_id = row["id"]
        point_id = await obj_enc.store_object_annotation(
            annotation_id=str(annotation_id),
            user_id=current_user.user_id,
            object_name=request.object_name.strip(),
            combined_embedding=features["combined_embedding"],
            visual_embedding=features.get("visual_embedding"),
            semantic_embedding=features.get("semantic_embedding"),
            source_document_id=document_id,
        )
        await execute(
            "UPDATE user_object_annotations SET visual_embedding_id = $1, combined_embedding_id = $2 WHERE id = $3",
            point_id,
            point_id,
            annotation_id,
        )

        return {
            "success": True,
            "annotation_id": annotation_id,
            "annotation": dict(row),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create object annotation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/annotations/{annotation_id}/add-example")
async def add_annotation_example(
    annotation_id: int,
    request: AddExampleRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Add another example region to an existing user-defined object annotation."""
    try:
        ann = await fetch_one(
            "SELECT id, user_id, object_name FROM user_object_annotations WHERE id = $1 AND user_id = $2",
            annotation_id,
            current_user.user_id,
        )
        if not ann:
            raise HTTPException(status_code=404, detail="Annotation not found or access denied")

        doc_info = await check_document_access(request.document_id, current_user, "read")
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found or access denied")

        folder_service = await _get_folder_service()
        filename = getattr(doc_info, "filename", "")
        user_id = getattr(doc_info, "user_id", None) or current_user.user_id
        folder_id = getattr(doc_info, "folder_id", None)
        collection_type = getattr(doc_info, "collection_type", "user")
        file_path_str = await folder_service.get_document_file_path(
            filename=filename,
            folder_id=folder_id,
            user_id=user_id,
            collection_type=collection_type,
        )
        image_path = Path(file_path_str)
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image file not found")

        from clients.image_vision_client import get_image_vision_client
        from services.object_encoding_service import get_object_encoding_service

        vision_client = await get_image_vision_client()
        await vision_client.initialize(required=False)
        if not vision_client._initialized:
            raise HTTPException(status_code=503, detail="Image Vision Service unavailable")

        bbox = {"x": request.bbox.x, "y": request.bbox.y, "width": request.bbox.width, "height": request.bbox.height}
        features = await vision_client.extract_object_features(
            image_path=str(image_path),
            bbox=bbox,
            description=ann["object_name"],
        )
        if not features or not features.get("combined_embedding"):
            raise HTTPException(status_code=500, detail="Failed to extract object features")

        obj_enc = await get_object_encoding_service()
        point_id = await obj_enc.add_annotation_example(
            annotation_id=str(annotation_id),
            user_id=current_user.user_id,
            object_name=ann["object_name"],
            combined_embedding=features["combined_embedding"],
            source_document_id=request.document_id,
        )
        await execute(
            """
            INSERT INTO object_annotation_examples
            (annotation_id, source_document_id, bbox_x, bbox_y, bbox_width, bbox_height, combined_embedding_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            annotation_id,
            request.document_id,
            request.bbox.x,
            request.bbox.y,
            request.bbox.width,
            request.bbox.height,
            point_id,
        )
        await execute(
            "UPDATE user_object_annotations SET example_count = example_count + 1, updated_at = CURRENT_TIMESTAMP WHERE id = $1",
            annotation_id,
        )

        return {"success": True, "annotation_id": annotation_id, "message": "Example added"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add annotation example failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/documents/{document_id}/object-annotations")
async def get_object_annotations_for_document(
    document_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """List user-defined object annotations created from this document (source_document_id)."""
    try:
        doc_info = await check_document_access(document_id, current_user, "read")
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found or access denied")

        rows = await fetch_all(
            """
            SELECT id, user_id, object_name, description, source_document_id,
                   bbox_x, bbox_y, bbox_width, bbox_height, example_count, created_at
            FROM user_object_annotations
            WHERE source_document_id = $1 AND user_id = $2
            ORDER BY created_at DESC
            """,
            document_id,
            current_user.user_id,
        )
        annotations = [dict(r) for r in rows] if rows else []
        return {"success": True, "annotations": annotations}
    except HTTPException:
        raise
    except Exception as e:
        if "does not exist" in str(e).lower() or "relation" in str(e).lower():
            logger.warning("Object annotations table not found; run migration 036_add_object_detection.sql")
            return {"success": True, "annotations": []}
        logger.error(f"Get object annotations failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/documents/{document_id}/objects")
async def get_detected_objects(
    document_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Get all detected objects for an image, including user-defined annotations."""
    try:
        doc_info = await check_document_access(document_id, current_user, "read")
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found or access denied")

        from services.object_detection_service import get_object_detection_service

        obj_service = await get_object_detection_service()
        objects = await obj_service.get_detected_objects(document_id)

        # Include user-defined annotations for this document so they show in bounding box overlays
        try:
            ann_rows = await fetch_all(
                """
                SELECT id, object_name, bbox_x, bbox_y, bbox_width, bbox_height
                FROM user_object_annotations
                WHERE source_document_id = $1 AND user_id = $2
                ORDER BY id
                """,
                document_id,
                current_user.user_id,
            )
            for r in ann_rows or []:
                d = dict(r)
                objects.append({
                    "id": "ann-%s" % d["id"],
                    "document_id": document_id,
                    "bbox_x": d["bbox_x"],
                    "bbox_y": d["bbox_y"],
                    "bbox_width": d["bbox_width"],
                    "bbox_height": d["bbox_height"],
                    "user_tag": d["object_name"],
                    "class_name": d["object_name"],
                    "detection_method": "user_defined",
                    "confidence": None,
                })
        except Exception:
            pass

        return {"success": True, "objects": objects}
    except HTTPException:
        raise
    except Exception as e:
        if "does not exist" in str(e).lower() or "relation" in str(e).lower():
            logger.warning("Detected objects table not found; run migration 036_add_object_detection.sql")
            return {"success": True, "objects": []}
        logger.error(f"Get detected objects failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/detected-objects/{object_id}/confirm")
async def confirm_object_detection(
    object_id: int,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Mark a detected object as confirmed by the user."""
    try:
        row = await fetch_one(
            "SELECT id, document_id FROM detected_objects WHERE id = $1",
            object_id,
        )
        if not row:
            raise HTTPException(status_code=404, detail="Object not found")
        doc_info = await check_document_access(row["document_id"], current_user, "write")
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found or access denied")

        await execute(
            "UPDATE detected_objects SET confirmed = TRUE, confirmed_by = $1, confirmed_at = CURRENT_TIMESTAMP WHERE id = $2",
            current_user.user_id,
            object_id,
        )
        return {"success": True, "object_id": object_id, "confirmed": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Confirm object failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/api/detected-objects/{object_id}")
async def update_detected_object(
    object_id: int,
    body: UpdateDetectedObjectRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    """Update a detected object: set refined tag (e.g. 'Car' -> 'BMW i3') and/or reject (hide)."""
    try:
        row = await fetch_one(
            "SELECT id, document_id FROM detected_objects WHERE id = $1",
            object_id,
        )
        if not row:
            raise HTTPException(status_code=404, detail="Object not found")
        doc_info = await check_document_access(row["document_id"], current_user, "write")
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found or access denied")

        updates = []
        params = []
        if body.user_tag is not None:
            tag_value = body.user_tag.strip() if body.user_tag else None
            updates.append("user_tag = $%d" % (len(params) + 1))
            params.append(tag_value)
            if tag_value:
                updates.append("class_name = $%d" % (len(params) + 1))
                params.append(tag_value)
        if body.rejected is not None:
            updates.append("rejected = $%d" % (len(params) + 1))
            params.append(body.rejected)
        if not updates:
            return {"success": True, "object_id": object_id, "message": "No updates"}
        params.append(object_id)
        await execute(
            "UPDATE detected_objects SET " + ", ".join(updates) + " WHERE id = $%d" % len(params),
            *params,
        )

        # When user_tag is set, extract CLIP embedding for the same YOLO bbox and store in object_features
        # so this custom identity can be matched visually in other images.
        if body.user_tag and body.user_tag.strip():
            try:
                full_row = await fetch_one(
                    """SELECT id, document_id, bbox_x, bbox_y, bbox_width, bbox_height, user_tag
                       FROM detected_objects WHERE id = $1""",
                    object_id,
                )
                if not full_row:
                    pass
                else:
                    folder_service = await _get_folder_service()
                    filename = getattr(doc_info, "filename", "")
                    user_id_attr = getattr(doc_info, "user_id", None) or current_user.user_id
                    folder_id = getattr(doc_info, "folder_id", None)
                    collection_type = getattr(doc_info, "collection_type", "user")
                    file_path_str = await folder_service.get_document_file_path(
                        filename=filename,
                        folder_id=folder_id,
                        user_id=user_id_attr,
                        collection_type=collection_type,
                    )
                    image_path = Path(file_path_str)
                    if image_path.exists():
                        from clients.image_vision_client import get_image_vision_client
                        from services.object_encoding_service import get_object_encoding_service

                        vision_client = await get_image_vision_client()
                        await vision_client.initialize(required=False)
                        if vision_client._initialized:
                            bbox = {
                                "x": full_row["bbox_x"],
                                "y": full_row["bbox_y"],
                                "width": full_row["bbox_width"],
                                "height": full_row["bbox_height"],
                            }
                            features = await vision_client.extract_object_features(
                                image_path=str(image_path),
                                bbox=bbox,
                                description=body.user_tag.strip(),
                            )
                            if features and features.get("combined_embedding"):
                                obj_enc = await get_object_encoding_service()
                                tag = body.user_tag.strip()
                                existing = await fetch_one(
                                    "SELECT id FROM user_object_annotations WHERE user_id = $1 AND object_name = $2",
                                    current_user.user_id,
                                    tag,
                                )
                                if existing:
                                    still_exists = await fetch_one(
                                        "SELECT id FROM user_object_annotations WHERE id = $1",
                                        existing["id"],
                                    )
                                    if not still_exists:
                                        existing = None
                                if existing:
                                    point_id = await obj_enc.add_annotation_example(
                                        annotation_id=str(existing["id"]),
                                        user_id=current_user.user_id,
                                        object_name=tag,
                                        combined_embedding=features["combined_embedding"],
                                        source_document_id=full_row["document_id"],
                                    )
                                    await execute(
                                        """
                                        INSERT INTO object_annotation_examples
                                        (annotation_id, source_document_id, bbox_x, bbox_y, bbox_width, bbox_height, combined_embedding_id)
                                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                                        """,
                                        existing["id"],
                                        full_row["document_id"],
                                        full_row["bbox_x"],
                                        full_row["bbox_y"],
                                        full_row["bbox_width"],
                                        full_row["bbox_height"],
                                        point_id,
                                    )
                                    await execute(
                                        "UPDATE detected_objects SET annotation_id = $1 WHERE id = $2",
                                        existing["id"],
                                        object_id,
                                    )
                                else:
                                    ann_row = await fetch_one(
                                        """
                                        INSERT INTO user_object_annotations
                                        (user_id, object_name, description, source_document_id, bbox_x, bbox_y, bbox_width, bbox_height,
                                         visual_embedding_id, combined_embedding_id, example_count)
                                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 1)
                                        RETURNING id
                                        """,
                                        current_user.user_id,
                                        tag,
                                        "",
                                        full_row["document_id"],
                                        full_row["bbox_x"],
                                        full_row["bbox_y"],
                                        full_row["bbox_width"],
                                        full_row["bbox_height"],
                                        "",
                                        "",
                                    )
                                    if ann_row:
                                        point_id = await obj_enc.store_object_annotation(
                                            annotation_id=str(ann_row["id"]),
                                            user_id=current_user.user_id,
                                            object_name=tag,
                                            combined_embedding=features["combined_embedding"],
                                            visual_embedding=features.get("visual_embedding"),
                                            semantic_embedding=features.get("semantic_embedding"),
                                            source_document_id=full_row["document_id"],
                                        )
                                        await execute(
                                            "UPDATE user_object_annotations SET visual_embedding_id = $1, combined_embedding_id = $2 WHERE id = $3",
                                            point_id,
                                            point_id,
                                            ann_row["id"],
                                        )
                                        await execute(
                                            "UPDATE detected_objects SET annotation_id = $1 WHERE id = $2",
                                            ann_row["id"],
                                            object_id,
                                        )
            except Exception as enc_err:
                logger.warning("CLIP encoding for YOLO user_tag skipped: %s", enc_err)

        return {"success": True, "object_id": object_id, "user_tag": body.user_tag, "rejected": body.rejected}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Update detected object failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/annotations/{annotation_id}")
async def delete_object_annotation(
    annotation_id: int,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """Delete a user-defined object annotation and its Qdrant vectors."""
    try:
        ann = await fetch_one(
            "SELECT id, user_id FROM user_object_annotations WHERE id = $1",
            annotation_id,
        )
        if not ann:
            raise HTTPException(status_code=404, detail="Annotation not found")
        if ann["user_id"] != current_user.user_id and current_user.role != "admin":
            raise HTTPException(status_code=403, detail="Not allowed to delete this annotation")

        from services.object_encoding_service import get_object_encoding_service

        obj_enc = await get_object_encoding_service()
        await obj_enc.delete_annotation(str(annotation_id), user_id=str(ann["user_id"]))
        await execute("DELETE FROM object_annotation_examples WHERE annotation_id = $1", annotation_id)
        await execute("DELETE FROM user_object_annotations WHERE id = $1", annotation_id)

        return {"success": True, "annotation_id": annotation_id, "message": "Annotation deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete annotation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/known-identities/{identity_name}")
async def delete_known_identity(
    identity_name: str,
    current_user: AuthenticatedUserResponse = Depends(require_admin)
):
    """
    Delete a known identity and all associated face encodings (admin only)
    Removes identity from both PostgreSQL and Qdrant
    """
    try:
        # Get face encoding service
        face_service = await get_face_encoding_service()
        
        # Delete from Qdrant first
        qdrant_count = await face_service.delete_identity(identity_name)
        
        # Delete from PostgreSQL
        result = await execute(
            "DELETE FROM known_identities WHERE identity_name = $1 RETURNING id",
            identity_name
        )
        
        if result:
            logger.info(f"✅ Deleted known identity: {identity_name} ({qdrant_count} Qdrant vectors)")
            return {
                "success": True,
                "message": f"Deleted identity: {identity_name}",
                "encodings_deleted": qdrant_count
            }
        else:
            raise HTTPException(status_code=404, detail="Identity not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete known identity: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete identity: {str(e)}")


@router.post("/api/vision/cleanup-orphaned-vectors")
async def cleanup_orphaned_face_vectors(
    current_user: AuthenticatedUserResponse = Depends(require_admin)
):
    """
    Clean up orphaned Qdrant vectors for deleted identities (admin only)
    Run this if you've deleted images and want to clean up Qdrant
    """
    try:
        face_service = await get_face_encoding_service()
        cleaned_count = await face_service.cleanup_orphaned_vectors()
        
        return {
            "success": True,
            "vectors_cleaned": cleaned_count,
            "message": f"Cleaned up {cleaned_count} orphaned face encoding vectors"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to cleanup orphaned vectors: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup vectors: {str(e)}")


@router.delete("/api/vision/clear-all-identities")
async def clear_all_face_identities(
    current_user: AuthenticatedUserResponse = Depends(require_admin)
):
    """
    Clear all face detection data (admin only)
    Deletes all detected faces, known identities, and Qdrant face encodings
    """
    try:
        # Get face encoding service
        face_service = await get_face_encoding_service()
        
        # Clear Qdrant face encodings first
        qdrant_count = await face_service.clear_all_encodings()
        
        # Delete PostgreSQL records (trigger will try to delete identities but they're already cleared above)
        detected_result = await execute("DELETE FROM detected_faces RETURNING id")
        detected_count = len(detected_result) if detected_result else 0
        
        identities_result = await execute("DELETE FROM known_identities RETURNING id")
        identities_count = len(identities_result) if identities_result else 0
        
        logger.info(f"✅ Cleared all face detection data: {detected_count} faces, {identities_count} identities, {qdrant_count} Qdrant vectors")
        
        return {
            "success": True,
            "detected_faces_deleted": detected_count,
            "known_identities_deleted": identities_count,
            "qdrant_vectors_deleted": qdrant_count,
            "message": f"Cleared {detected_count} detected faces, {identities_count} known identities, and {qdrant_count} face encoding vectors"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to clear face identities: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear face identities: {str(e)}")


@router.post("/api/documents/{document_id}/add-identity-tag")
async def add_identity_to_metadata_tags(
    document_id: str,
    identity_name: str = Body(..., embed=True),
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Add a detected face identity to the image's metadata.json tags field
    and sync to document_metadata.tags for search
    """
    try:
        # Check document access
        doc_info = await check_document_access(document_id, current_user, "write")
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found or access denied")
        
        # Get collection type and user_id
        collection_type = getattr(doc_info, 'collection_type', 'user')
        user_id = getattr(doc_info, 'user_id', None) or current_user.user_id
        
        # Get image sidecar service
        from services.image_sidecar_service import get_image_sidecar_service
        sidecar_service = await get_image_sidecar_service()
        
        # Sync identity to tags
        result = await sidecar_service.sync_identity_to_tags(
            document_id=document_id,
            identity_name=identity_name,
            user_id=user_id,
            collection_type=collection_type
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to add identity to tags"))
        
        return {
            "success": True,
            "tags": result.get("tags", []),
            "message": result.get("message", f"Added '{identity_name}' to tags")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to add identity to tags: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add identity to tags: {str(e)}")


@router.get("/api/documents/{document_id}/suggest-face-tags")
async def suggest_face_tags_from_metadata(
    document_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
):
    """
    Compare metadata.json tags with untagged detected faces
    Returns suggestions for which faces might match which tags
    """
    try:
        # Check document access
        doc_info = await check_document_access(document_id, current_user, "read")
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found or access denied")
        
        # Get metadata.json tags
        filename = getattr(doc_info, 'filename', '')
        user_id = getattr(doc_info, 'user_id', None)
        folder_id = getattr(doc_info, 'folder_id', None)
        collection_type = getattr(doc_info, 'collection_type', 'user')
        
        # Get file path
        folder_service = await _get_folder_service()
        file_path_str = await folder_service.get_document_file_path(
            filename=filename,
            folder_id=folder_id,
            user_id=user_id,
            collection_type=collection_type
        )
        image_path = Path(file_path_str)
        
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image file not found")
        
        # Load metadata.json
        # Use stem (filename without extension) to match actual sidecar naming: image.jpg -> image.metadata.json
        metadata_path = image_path.parent / f"{image_path.stem}.metadata.json"
        metadata_tags = []
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata_data = json.load(f)
                    metadata_tags = metadata_data.get('tags', [])
                    if not isinstance(metadata_tags, list):
                        metadata_tags = []
            except Exception as e:
                logger.warning(f"Failed to load metadata.json: {e}")
        
        # Get untagged faces
        untagged_faces = await fetch_all(
            "SELECT id, bbox_x, bbox_y, bbox_width, bbox_height FROM detected_faces WHERE document_id = $1 AND identity_name IS NULL",
            document_id
        )
        
        if not untagged_faces or len(untagged_faces) == 0:
            return {
                "success": True,
                "has_suggestions": False,
                "message": "No untagged faces found"
            }
        
        # Check if any metadata tags match known_identities
        suggestions = []
        for tag in metadata_tags:
            if not isinstance(tag, str) or not tag.strip():
                continue
            
            tag_lower = tag.strip().lower()
            
            # Check if this tag exists in known_identities
            known_identity = await fetch_one(
                "SELECT identity_name, sample_count FROM known_identities WHERE LOWER(identity_name) = $1",
                tag_lower
            )
            
            if known_identity:
                suggestions.append({
                    "tag": tag,
                    "identity_name": known_identity["identity_name"],
                    "sample_count": known_identity["sample_count"],
                    "untagged_faces_count": len(untagged_faces)
                })
        
        if not suggestions:
            return {
                "success": True,
                "has_suggestions": False,
                "message": f"Found {len(untagged_faces)} untagged face(s), but no matching tags in metadata"
            }
        
        return {
            "success": True,
            "has_suggestions": True,
            "untagged_faces_count": len(untagged_faces),
            "suggestions": suggestions,
            "message": f"Found {len(untagged_faces)} untagged face(s). You have tags that match known identities: {', '.join([s['tag'] for s in suggestions])}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to suggest face tags: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to suggest face tags: {str(e)}")
