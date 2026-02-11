"""
Attachment Analysis Subgraph
Analyzes attached images using vision LLM and face detection
"""

import logging
from typing import Dict, Any, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


class AttachmentAnalysisState(TypedDict):
    """State for attachment analysis subgraph"""
    # Input
    query: str
    attachments: List[Dict[str, Any]]
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]

    # Processing
    primary_attachment: Dict[str, Any]
    attachment_type: str

    # Analysis results
    vision_description: Optional[str]
    face_detection_results: Optional[Dict[str, Any]]
    detected_identities: List[str]

    # Output
    analysis_summary: str
    confidence: float
    response: Dict[str, Any]
    task_status: str
    error: str


async def _select_attachment_node(state: AttachmentAnalysisState) -> Dict[str, Any]:
    """Extract first image attachment for analysis"""
    try:
        attachments = state.get("attachments", [])
        user_id = state.get("user_id", "system")
        metadata = state.get("metadata", {})
        shared_memory = state.get("shared_memory", {})
        query = state.get("query", "")
        messages = state.get("messages", [])

        if not attachments:
            return {
                "primary_attachment": {},
                "attachment_type": "",
                "error": "No attachments provided",
                "task_status": "error",
                "metadata": metadata,
                "user_id": user_id,
                "shared_memory": shared_memory,
                "messages": messages,
                "query": query,
            }

        primary = attachments[0]
        content_type = primary.get("content_type", "")
        is_image = content_type.startswith("image/") if content_type else False

        logger.info(f"Selected attachment: {primary.get('filename', 'unknown')}, is_image={is_image}")

        return {
            "primary_attachment": primary,
            "attachment_type": "image" if is_image else "document",
            "metadata": metadata,
            "user_id": user_id,
            "shared_memory": shared_memory,
            "messages": messages,
            "query": query,
        }
    except Exception as e:
        logger.error(f"select_attachment_node failed: {e}")
        return {
            "primary_attachment": {},
            "attachment_type": "",
            "error": str(e),
            "task_status": "error",
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }


async def _analyze_with_vision_node(state: AttachmentAnalysisState) -> Dict[str, Any]:
    """Describe image using vision LLM (multimodal message)."""
    try:
        primary = state.get("primary_attachment", {})
        user_id = state.get("user_id", "system")
        metadata = state.get("metadata", {})
        shared_memory = state.get("shared_memory", {})
        query = state.get("query", "")
        messages = state.get("messages", [])

        if not primary or state.get("attachment_type") != "image":
            return {
                "vision_description": "",
                "metadata": metadata,
                "user_id": user_id,
                "shared_memory": shared_memory,
                "messages": messages,
                "query": query,
                "primary_attachment": primary,
                "attachment_type": state.get("attachment_type", ""),
            }

        base64_data = primary.get("base64_data") or primary.get("data")
        if not base64_data and isinstance(primary.get("base64"), str):
            base64_data = primary["base64"]

        if not base64_data:
            logger.warning("No base64 image data in primary attachment, skipping vision analysis")
            return {
                "vision_description": "",
                "metadata": metadata,
                "user_id": user_id,
                "shared_memory": shared_memory,
                "messages": messages,
                "query": query,
                "primary_attachment": primary,
                "attachment_type": state.get("attachment_type", ""),
            }

        from orchestrator.agents.base_agent import BaseAgent

        base_agent = BaseAgent("attachment_vision")
        llm = base_agent._get_llm(temperature=0.3, state={"metadata": metadata})
        prompt = query or "Describe this image in detail."
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"}},
        ]
        llm_messages = [
            SystemMessage(content="You are a helpful vision assistant. Describe the image accurately and concisely."),
            HumanMessage(content=content),
        ]
        response = await llm.ainvoke(llm_messages)
        description = response.content if hasattr(response, "content") and response.content else ""

        logger.info(f"Vision description length: {len(description)} chars")

        return {
            "vision_description": description,
            "metadata": metadata,
            "user_id": user_id,
            "shared_memory": shared_memory,
            "messages": messages,
            "query": query,
            "primary_attachment": primary,
            "attachment_type": state.get("attachment_type", ""),
        }
    except Exception as e:
        logger.error(f"analyze_with_vision_node failed: {e}")
        return {
            "vision_description": "",
            "error": str(e),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "primary_attachment": state.get("primary_attachment", {}),
            "attachment_type": state.get("attachment_type", ""),
        }


async def _detect_faces_node(state: AttachmentAnalysisState) -> Dict[str, Any]:
    """Call backend to identify faces in the attached image (requires file_path on backend)"""
    try:
        primary = state.get("primary_attachment", {})
        user_id = state.get("user_id", "system")
        metadata = state.get("metadata", {})
        shared_memory = state.get("shared_memory", {})
        messages = state.get("messages", [])
        query = state.get("query", "")

        file_path = primary.get("file_path")
        face_results = None
        detected_identities = []

        if file_path and state.get("attachment_type") == "image":
            try:
                from orchestrator.backend_tool_client import get_backend_tool_client

                client = await get_backend_tool_client()
                result = await client.identify_faces(
                    attachment_path=file_path,
                    user_id=user_id,
                    confidence_threshold=0.82,
                )
                if result.get("success") and result.get("identified_faces"):
                    face_results = result
                    detected_identities = [
                        f["identity_name"]
                        for f in result["identified_faces"]
                        if f.get("identity_name")
                    ]
                    logger.info(f"Face detection: {len(detected_identities)} identities")
                else:
                    face_results = result
                    logger.info("Face detection completed, no identities matched")
            except Exception as e:
                logger.warning(f"Face detection failed (backend may be unavailable): {e}")
                face_results = {"success": False, "error": str(e)}
        else:
            if not file_path:
                logger.debug("No file_path in attachment, skipping face detection")
            face_results = {}

        return {
            "face_detection_results": face_results,
            "detected_identities": detected_identities,
            "metadata": metadata,
            "user_id": user_id,
            "shared_memory": shared_memory,
            "messages": messages,
            "query": query,
            "vision_description": state.get("vision_description", ""),
            "primary_attachment": primary,
            "attachment_type": state.get("attachment_type", ""),
        }
    except Exception as e:
        logger.error(f"detect_faces_node failed: {e}")
        return {
            "face_detection_results": {"success": False, "error": str(e)},
            "detected_identities": [],
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
            "vision_description": state.get("vision_description", ""),
            "primary_attachment": state.get("primary_attachment", {}),
            "attachment_type": state.get("attachment_type", ""),
        }


async def _format_response_node(state: AttachmentAnalysisState) -> Dict[str, Any]:
    """Combine vision description and face results into natural language summary"""
    try:
        vision_description = state.get("vision_description", "")
        face_results = state.get("face_detection_results") or {}
        detected_identities = state.get("detected_identities", [])
        query = state.get("query", "")
        err = state.get("error", "")

        if err:
            return {
                "analysis_summary": f"I couldn't analyze the image: {err}",
                "confidence": 0.0,
                "response": {
                    "error": err,
                    "task_status": "error",
                },
                "task_status": "error",
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": query,
            }

        parts = []

        if detected_identities:
            if len(detected_identities) == 1:
                parts.append(f"I found one person: **{detected_identities[0]}**.")
            else:
                names = ", ".join(f"**{n}**" for n in detected_identities)
                parts.append(f"I found {len(detected_identities)} people: {names}.")
            identified_faces = face_results.get("identified_faces", [])
            if identified_faces:
                for f in identified_faces:
                    conf = f.get("confidence", 0)
                    if conf:
                        parts.append(f"  - {f.get('identity_name', 'Unknown')}: {conf:.0f}% confidence")
        elif face_results.get("face_count", 0) > 0 and not detected_identities:
            parts.append(
                f"I detected {face_results['face_count']} face(s) in the image, but none matched known identities."
            )

        if vision_description:
            if parts:
                parts.append("")
            parts.append("**Description:**")
            parts.append(vision_description)

        if not parts:
            parts.append(
                "I couldn't extract a description or identify anyone in this image. "
                "The image may be unclear or the analysis services may be unavailable."
            )

        analysis_summary = "\n".join(parts)
        confidence = 0.9 if (vision_description or detected_identities) else 0.5

        response = {
            "description": vision_description,
            "detected_identities": detected_identities,
            "face_detection_results": face_results,
            "task_status": "complete",
        }

        return {
            "analysis_summary": analysis_summary,
            "confidence": confidence,
            "response": response,
            "task_status": "complete",
            "error": "",
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": query,
        }
    except Exception as e:
        logger.error(f"format_response_node failed: {e}")
        return {
            "analysis_summary": f"Analysis failed: {e}",
            "confidence": 0.0,
            "response": {"error": str(e), "task_status": "error"},
            "task_status": "error",
            "error": str(e),
            "metadata": state.get("metadata", {}),
            "user_id": state.get("user_id", "system"),
            "shared_memory": state.get("shared_memory", {}),
            "messages": state.get("messages", []),
            "query": state.get("query", ""),
        }


def build_attachment_analysis_subgraph():
    """Build attachment analysis subgraph"""
    workflow = StateGraph(AttachmentAnalysisState)

    workflow.add_node("select_attachment", _select_attachment_node)
    workflow.add_node("analyze_with_vision", _analyze_with_vision_node)
    workflow.add_node("detect_faces", _detect_faces_node)
    workflow.add_node("format_response", _format_response_node)

    workflow.set_entry_point("select_attachment")
    workflow.add_edge("select_attachment", "analyze_with_vision")
    workflow.add_edge("analyze_with_vision", "detect_faces")
    workflow.add_edge("detect_faces", "format_response")
    workflow.add_edge("format_response", END)

    return workflow.compile()
