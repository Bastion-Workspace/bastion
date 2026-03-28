"""
Face Analysis Tools - LangGraph tools for face detection and identification
"""

import logging
from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field

from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


# ── I/O models for identify_faces_in_image ──────────────────────────────────

class IdentifyFacesInputs(BaseModel):
    """Required inputs for identify_faces_in_image."""
    attachment_path: str = Field(description="Full path to the image file")


class IdentifyFacesParams(BaseModel):
    """Optional configuration."""
    identify: bool = Field(default=True, description="If True match faces to identities; if False only detect faces (no identification)")
    confidence_threshold: float = Field(default=0.85, description="Minimum confidence for identity matches (when identify=True)")


class IdentifiedFace(BaseModel):
    """Single identified face."""
    identity_name: str = Field(description="Name of identified person")
    confidence: float = Field(description="Match confidence 0-1")


class IdentifyFacesOutputs(BaseModel):
    """Outputs for identify_faces_in_image (detection-only when identify=False)."""
    success: bool = Field(description="Whether the operation succeeded")
    face_count: int = Field(default=0, description="Number of faces detected")
    identified_count: int = Field(default=0, description="Number of faces matched to identities (when identify=True)")
    identified_faces: List[Dict[str, Any]] = Field(default_factory=list, description="List of identified faces (when identify=True)")
    faces: List[Dict[str, Any]] = Field(default_factory=list, description="Face detections with encodings/bounding boxes (when identify=False)")
    image_width: Optional[int] = Field(default=None, description="Image width in pixels (when identify=False)")
    image_height: Optional[int] = Field(default=None, description="Image height in pixels (when identify=False)")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


async def identify_faces_in_image(
    attachment_path: str,
    user_id: str = "system",
    identify: bool = True,
    confidence_threshold: float = 0.85
) -> Dict[str, Any]:
    """
    Detect and optionally identify faces in an image.
    When identify=False, only detects faces (encodings, bounding boxes).
    When identify=True, matches faces against known identities.
    """
    try:
        from orchestrator.backend_tool_client import get_backend_tool_client

        client = await get_backend_tool_client()

        if not identify:
            logger.info(f"Detecting faces in image: {attachment_path}")
            result = await client.detect_faces(
                attachment_path=attachment_path,
                user_id=user_id
            )
            if result.get("success"):
                face_count = result.get("face_count", 0)
                formatted = f"Detected {face_count} face(s) in the image."
            else:
                formatted = f"Face detection failed: {result.get('error', 'Unknown error')}"
            return {
                "success": result.get("success", False),
                "face_count": result.get("face_count", 0),
                "identified_count": 0,
                "identified_faces": [],
                "faces": result.get("faces", []),
                "image_width": result.get("image_width"),
                "image_height": result.get("image_height"),
                "error": result.get("error"),
                "formatted": formatted,
            }

        logger.info(f"Identifying faces in image: {attachment_path}")
        result = await client.identify_faces(
            attachment_path=attachment_path,
            user_id=user_id,
            confidence_threshold=confidence_threshold
        )

        if not result.get("success"):
            error_msg = result.get("error", "Unknown error")
            logger.warning(f"Face identification failed: {error_msg}")
            return {
                "success": False,
                "face_count": 0,
                "identified_count": 0,
                "identified_faces": [],
                "error": error_msg,
                "formatted": f"Failed to identify faces: {error_msg}",
            }

        face_count = result.get("face_count", 0)
        identified_faces = result.get("identified_faces", [])
        identified_count = result.get("identified_count", 0)

        if face_count == 0:
            formatted = "No faces detected in the image."
        elif identified_count == 0:
            formatted = f"Detected {face_count} face(s) in the image, but no matching identities were found in the database."
        else:
            descriptions = []
            for face in identified_faces:
                identity_name = face.get("identity_name", "Unknown")
                confidence = face.get("confidence", 0.0)
                confidence_pct = int(confidence * 100)
                descriptions.append(f"{identity_name} (confidence: {confidence_pct}%)")
            if identified_count == 1:
                formatted = f"Identified 1 person: {descriptions[0]}"
            else:
                formatted = f"Identified {identified_count} person(s): {', '.join(descriptions)}"

        return {
            "success": True,
            "face_count": face_count,
            "identified_count": identified_count,
            "identified_faces": identified_faces,
            "faces": [],
            "image_width": None,
            "image_height": None,
            "error": None,
            "formatted": formatted,
        }

    except Exception as e:
        logger.error(f"Face identification tool error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "face_count": 0,
            "identified_count": 0,
            "identified_faces": [],
            "faces": [],
            "image_width": None,
            "image_height": None,
            "error": str(e),
            "formatted": f"Error identifying faces: {str(e)}",
        }


register_action(
    name="identify_faces_in_image",
    category="image",
    description="Detect and optionally identify faces in an image (set identify=False for detection only)",
    inputs_model=IdentifyFacesInputs,
    params_model=IdentifyFacesParams,
    outputs_model=IdentifyFacesOutputs,
    tool_function=identify_faces_in_image,
)


# Tool list for registration
FACE_ANALYSIS_TOOLS = [
    identify_faces_in_image,
]
