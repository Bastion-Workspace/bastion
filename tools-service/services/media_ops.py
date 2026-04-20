"""Media-related orchestration for gRPC handlers (image search, face identification)."""

from typing import Any, Dict, List, Optional


async def search_images_for_grpc(
    *,
    query: str,
    image_type: Optional[str],
    date: Optional[str],
    author: Optional[str],
    series: Optional[str],
    limit: int,
    user_id: Optional[str],
    is_random: bool,
    exclude_document_ids: Optional[List[str]],
) -> Dict[str, Any]:
    """Invoke ImageSearchTools and normalize return value for gRPC mapping."""
    from services.langgraph_tools.image_search_tools import ImageSearchTools

    image_search = ImageSearchTools()
    result = await image_search.search_images(
        query=query,
        image_type=image_type,
        date=date,
        author=author,
        series=series,
        limit=limit or 10,
        user_id=user_id,
        is_random=is_random,
        exclude_document_ids=exclude_document_ids,
    )
    if isinstance(result, dict):
        return {
            "format": "structured",
            "images_markdown": result.get("images_markdown", ""),
            "metadata": result.get("metadata", []),
            "structured_images": result.get("images", []),
        }
    text = result if isinstance(result, str) else str(result)
    return {"format": "legacy", "results_text": text}


async def identify_faces_for_grpc(
    *, attachment_path: str, user_id: str, confidence_threshold: float
) -> Dict[str, Any]:
    """Run attachment face pipeline and return data for IdentifyFacesResponse."""
    from services.attachment_processor_service import attachment_processor_service

    _ = confidence_threshold  # reserved for future use; matches handler behavior
    result = await attachment_processor_service.process_image_for_search(
        attachment_path=attachment_path,
        user_id=user_id,
    )
    if result.get("error"):
        return {
            "success": False,
            "error": result.get("error"),
            "face_count": 0,
            "identified_faces": [],
            "identified_count": 0,
        }
    face_count = result.get("face_count", 0)
    detected_identities = result.get("detected_identities", [])
    bounding_boxes = result.get("bounding_boxes", [])
    faces: List[Dict[str, Any]] = []
    for i, identity_name in enumerate(detected_identities):
        if i < len(bounding_boxes):
            bbox = bounding_boxes[i]
            faces.append(
                {
                    "identity_name": identity_name,
                    "confidence": 0.85,
                    "bbox_x": bbox.get("x", 0),
                    "bbox_y": bbox.get("y", 0),
                    "bbox_width": bbox.get("width", 0),
                    "bbox_height": bbox.get("height", 0),
                }
            )
    return {
        "success": True,
        "error": None,
        "face_count": face_count,
        "identified_faces": faces,
        "identified_count": len(detected_identities),
    }


async def generate_images_for_grpc(
    *,
    prompt: str,
    size: str,
    fmt: str,
    seed: Optional[int],
    num_images: int,
    negative_prompt: Optional[str],
    model: Optional[str],
    reference_image_data: Optional[bytes],
    reference_image_url: Optional[str],
    reference_strength: float,
    user_id: Optional[str],
    folder_id: Optional[str],
) -> Dict[str, Any]:
    from services.image_generation_service import get_image_generation_service

    image_service = await get_image_generation_service()
    return await image_service.generate_images(
        prompt=prompt,
        size=size or "1024x1024",
        fmt=fmt or "png",
        seed=seed,
        num_images=num_images or 1,
        negative_prompt=negative_prompt,
        model=model,
        reference_image_data=reference_image_data,
        reference_image_url=reference_image_url,
        reference_strength=reference_strength,
        user_id=user_id,
        folder_id=folder_id,
    )
