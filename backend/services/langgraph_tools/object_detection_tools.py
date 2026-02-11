"""Object detection tools for LangGraph agents."""
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from services.service_container import get_service_container
from services.object_detection_service import get_object_detection_service
from services.database_manager.database_helpers import fetch_all

logger = logging.getLogger(__name__)

async def _get_document_path(document_id: str, user_id: str) -> Optional[str]:
    try:
        container = await get_service_container()
        doc_info = await container.document_service.document_repository.get_by_id(document_id)
        if not doc_info:
            return None
        doc_user_id = getattr(doc_info, "user_id", None)
        if doc_user_id != user_id:
            return None
        filename = getattr(doc_info, "filename", "")
        folder_id = getattr(doc_info, "folder_id", None)
        collection_type = getattr(doc_info, "collection_type", "user")
        file_path_str = await container.folder_service.get_document_file_path(
            filename=filename, folder_id=folder_id, user_id=doc_user_id, collection_type=collection_type,
        )
        path = Path(file_path_str)
        return str(path) if path.exists() else None
    except Exception as e:
        logger.error("Failed to get document path: %s", e)
        return None


async def get_reference_image_bytes_for_object(object_name: str, user_id: str) -> Optional[tuple]:
    """
    Find an image containing the given object and return its bytes plus document_id.
    Used by image generation to supply a reference image (e.g. Farmall Tractor).
    Returns (bytes, document_id) or None.
    """
    try:
        result = await search_images_by_object(object_name=object_name.strip(), user_id=user_id, limit=1)
        if not result.get("success") or not result.get("document_ids"):
            return None
        document_id = result["document_ids"][0]
        path = await _get_document_path(document_id, user_id)
        if not path:
            return None
        with open(path, "rb") as f:
            data = f.read()
        return (data, document_id)
    except Exception as e:
        logger.error("get_reference_image_bytes_for_object failed: %s", e)
        return None

async def detect_objects_in_image(document_id: str, user_id: str, class_filter: Optional[List[str]] = None, confidence_threshold: float = 0.5, semantic_descriptions: Optional[List[str]] = None) -> Dict[str, Any]:
    image_path = await _get_document_path(document_id, user_id)
    if not image_path:
        return {"success": False, "error": "Document not found or access denied", "objects": []}
    try:
        obj_service = await get_object_detection_service()
        result = await obj_service.detect_objects_in_image(document_id=document_id, image_path=image_path, user_id=user_id, class_filter=class_filter, confidence_threshold=confidence_threshold, semantic_descriptions=semantic_descriptions or [], match_user_annotations=True)
        if result.get("error"):
            return {"success": False, "error": result["error"], "objects": []}
        await obj_service.process_detection_results(document_id, result["objects"], user_id=user_id)
        stored = await obj_service.get_detected_objects(document_id)
        return {"success": True, "objects": stored, "image_width": result.get("image_width"), "image_height": result.get("image_height"), "processing_time_seconds": result.get("processing_time_seconds")}
    except Exception as e:
        logger.error("Object detection failed: %s", e)
        return {"success": False, "error": str(e), "objects": []}

async def search_images_by_object(object_name: str, user_id: str, limit: int = 20) -> Dict[str, Any]:
    """Find document_ids containing this object (class_name, user_tag, or user annotation name)."""
    try:
        rows = await fetch_all(
            """
            SELECT DISTINCT dobj.document_id
            FROM detected_objects dobj
            LEFT JOIN user_object_annotations uoa ON dobj.annotation_id = uoa.id AND uoa.user_id = $2
            WHERE (dobj.class_name = $1 OR dobj.user_tag = $1 OR uoa.object_name = $1)
              AND (dobj.rejected IS NULL OR dobj.rejected = FALSE)
            ORDER BY dobj.document_id
            LIMIT $3
            """,
            object_name,
            user_id,
            limit,
        )
        document_ids = [r["document_id"] for r in rows] if rows else []
        return {"success": True, "object_name": object_name, "document_ids": document_ids, "count": len(document_ids)}
    except Exception as e:
        if "user_tag" in str(e) or "column" in str(e).lower():
            try:
                rows = await fetch_all(
                    "SELECT DISTINCT dobj.document_id FROM detected_objects dobj "
                    "LEFT JOIN user_object_annotations uoa ON dobj.annotation_id = uoa.id AND uoa.user_id = $2 "
                    "WHERE dobj.class_name = $1 OR uoa.object_name = $1 ORDER BY dobj.document_id LIMIT $3",
                    object_name,
                    user_id,
                    limit,
                )
                document_ids = [r["document_id"] for r in rows] if rows else []
                return {"success": True, "object_name": object_name, "document_ids": document_ids, "count": len(document_ids)}
            except Exception as e2:
                logger.error("Search images by object failed: %s", e2)
                return {"success": False, "error": str(e2), "document_ids": [], "count": 0}
        logger.error("Search images by object failed: %s", e)
        return {"success": False, "error": str(e), "document_ids": [], "count": 0}

async def annotate_custom_object(document_id: str, user_id: str, object_name: str, bbox_x: int, bbox_y: int, bbox_width: int, bbox_height: int, description: Optional[str] = None) -> Dict[str, Any]:
    image_path = await _get_document_path(document_id, user_id)
    if not image_path:
        return {"success": False, "error": "Document not found or access denied", "annotation_id": None}
    try:
        from clients.image_vision_client import get_image_vision_client
        from services.object_encoding_service import get_object_encoding_service
        from services.database_manager.database_helpers import fetch_one, execute
        vision_client = await get_image_vision_client()
        await vision_client.initialize(required=False)
        if not vision_client._initialized:
            return {"success": False, "error": "Image Vision Service unavailable", "annotation_id": None}
        bbox = {"x": bbox_x, "y": bbox_y, "width": bbox_width, "height": bbox_height}
        features = await vision_client.extract_object_features(image_path=image_path, bbox=bbox, description=description or object_name)
        if not features or not features.get("combined_embedding"):
            return {"success": False, "error": "Failed to extract object features", "annotation_id": None}
        obj_enc = await get_object_encoding_service()
        name = (object_name or "").strip()
        existing = await fetch_one(
            "SELECT id, object_name FROM user_object_annotations WHERE user_id = $1 AND object_name = $2",
            user_id, name,
        )
        if existing:
            annotation_id = existing["id"]
            point_id = await obj_enc.add_annotation_example(
                annotation_id=str(annotation_id),
                user_id=user_id,
                object_name=existing["object_name"],
                combined_embedding=features["combined_embedding"],
                source_document_id=document_id,
            )
            await execute(
                "INSERT INTO object_annotation_examples (annotation_id, source_document_id, bbox_x, bbox_y, bbox_width, bbox_height, combined_embedding_id) VALUES ($1, $2, $3, $4, $5, $6, $7)",
                annotation_id, document_id, bbox_x, bbox_y, bbox_width, bbox_height, point_id,
            )
            await execute(
                "UPDATE user_object_annotations SET example_count = example_count + 1, updated_at = CURRENT_TIMESTAMP WHERE id = $1",
                annotation_id,
            )
            return {"success": True, "annotation_id": annotation_id, "object_name": existing["object_name"], "added_as_example": True}
        row = await fetch_one(
            "INSERT INTO user_object_annotations (user_id, object_name, description, source_document_id, bbox_x, bbox_y, bbox_width, bbox_height, visual_embedding_id, combined_embedding_id, example_count) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 1) RETURNING id, object_name, created_at",
            user_id, name, description or "", document_id, bbox_x, bbox_y, bbox_width, bbox_height, "", "",
        )
        if not row:
            return {"success": False, "error": "Failed to create annotation row", "annotation_id": None}
        annotation_id = row["id"]
        point_id = await obj_enc.store_object_annotation(annotation_id=str(annotation_id), user_id=user_id, object_name=name, combined_embedding=features["combined_embedding"], visual_embedding=features.get("visual_embedding"), semantic_embedding=features.get("semantic_embedding"), source_document_id=document_id)
        await execute("UPDATE user_object_annotations SET visual_embedding_id = $1, combined_embedding_id = $2 WHERE id = $3", point_id, point_id, annotation_id)
        return {"success": True, "annotation_id": annotation_id, "object_name": row["object_name"]}
    except Exception as e:
        logger.error("Create object annotation failed: %s", e)
        return {"success": False, "error": str(e), "annotation_id": None}
