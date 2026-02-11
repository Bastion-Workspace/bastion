"""
Attachment detection and analysis nodes for the research workflow.
"""

import logging
from typing import Any, Dict

from orchestrator.agents.research.research_state import ResearchState

logger = logging.getLogger(__name__)


async def detect_attachments_node(agent: Any, state: ResearchState) -> Dict[str, Any]:
    """Detect if user has attached images; set has_attachments and attachments list."""
    shared_memory = state.get("shared_memory", {})
    metadata = state.get("metadata", {})

    attachments = shared_memory.get("attachments", [])
    if not attachments and metadata.get("attachments"):
        attachments = metadata["attachments"]

    image_attachments = [
        att for att in attachments
        if att.get("content_type", "").startswith("image/")
    ]

    if not image_attachments and shared_memory.get("attached_images"):
        attached_images = shared_memory["attached_images"]
        image_attachments = [
            {"content_type": "image/jpeg", "file_path": a.get("storage_path") or a.get("file_path"), "base64_data": a.get("data") or a.get("base64")}
            for a in (attached_images if isinstance(attached_images, list) else [])
            if isinstance(a, dict)
        ]

    logger.info("Detected %d image attachment(s)", len(image_attachments))
    return {
        "has_attachments": len(image_attachments) > 0,
        "attachments": image_attachments,
        "attached_images": shared_memory.get("attached_images", []),
        "metadata": metadata,
        "user_id": state.get("user_id", "system"),
        "shared_memory": shared_memory,
        "messages": state.get("messages", []),
        "query": state.get("query", ""),
    }


async def attachment_analysis_node(agent: Any, state: ResearchState) -> Dict[str, Any]:
    """Run attachment analysis subgraph (vision LLM + face detection)."""
    try:
        from orchestrator.subgraphs.attachment_analysis_subgraph import build_attachment_analysis_subgraph

        subgraph = build_attachment_analysis_subgraph()
        subgraph_state = {
            "query": state.get("query", ""),
            "attachments": state.get("attachments", []),
            "user_id": state.get("user_id", "system"),
            "metadata": state.get("metadata", {}),
            "messages": state.get("messages", []),
            "shared_memory": state.get("shared_memory", {}),
            "primary_attachment": {},
            "attachment_type": "",
            "vision_description": None,
            "face_detection_results": None,
            "detected_identities": [],
            "analysis_summary": "",
            "confidence": 0.0,
            "response": {},
            "task_status": "",
            "error": "",
        }
        result = await subgraph.ainvoke(subgraph_state)
        analysis_summary = result.get("analysis_summary", "")
        response = result.get("response", {})

        return {
            "final_response": analysis_summary,
            "research_complete": True,
            "attachment_analysis_results": response,
            "attachment_analysis": {"result": analysis_summary, "type": "attachment_analysis", **response},
            "attachment_processed": True,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }
    except Exception as e:
        logger.error("Attachment analysis failed: %s", e)
        return {
            "final_response": f"Image analysis failed: {e}",
            "research_complete": True,
            "attachment_analysis": {"error": str(e)},
            "attachment_processed": True,
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }


async def process_attachments_node(agent: Any, state: ResearchState) -> Dict[str, Any]:
    """Process attached images for face identification (legacy path)."""
    try:
        shared_memory = state.get("shared_memory", {})
        attached_images = shared_memory.get("attached_images", [])
        user_id = state.get("user_id", "system")
        query = state.get("query", "")
        query_lower = query.lower()

        result = {
            "attachment_processed": True,
            "attached_images": attached_images,
            "attachment_analysis": None,
            "metadata": state.get("metadata", {}),
            "user_id": user_id,
            "shared_memory": shared_memory,
            "messages": state.get("messages", []),
            "query": query,
        }

        if not attached_images:
            logger.info("No attached images found - skipping attachment processing")
            return result

        identity_keywords = ["who", "person", "identify", "face", "people", "this", "that"]
        wants_identification = any(kw in query_lower for kw in identity_keywords)

        if not wants_identification:
            logger.info("Query doesn't indicate face identification intent - skipping face processing")
            return result

        first_image = attached_images[0]
        attachment_path = first_image.get("storage_path") or first_image.get("file_path")

        if not attachment_path:
            logger.warning("Attached image missing storage_path")
            result["attachment_analysis"] = {"error": "Image path not found"}
            return result

        from orchestrator.tools.face_analysis_tools import identify_faces_in_image

        result_text = await identify_faces_in_image(
            attachment_path=attachment_path,
            user_id=user_id,
            confidence_threshold=0.85
        )

        logger.info("Face identification complete: %s", result_text[:100])

        result["attachment_analysis"] = {
            "type": "face_identification",
            "result": result_text,
            "image_path": attachment_path
        }

        return result

    except Exception as e:
        logger.error("Failed to process attachments: %s", e)
        return {
            "attachment_processed": True,
            "attachment_analysis": {"error": str(e)},
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }
