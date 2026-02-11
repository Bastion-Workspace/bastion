"""
Face Analysis Tools - LangGraph tools for face detection and identification
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


async def detect_faces_in_image(
    attachment_path: str,
    user_id: str = "system"
) -> Dict[str, Any]:
    """
    Detect faces in an attached image file.
    
    Returns face encodings and bounding boxes for all detected faces.
    Use this when you need to detect faces but don't need to identify who they are.
    
    Args:
        attachment_path: Full path to the image file
        user_id: User ID for access control
        
    Returns:
        Dict with:
        - success: bool
        - faces: List of face detections with encodings and bounding boxes
        - face_count: Number of faces detected
        - image_width: Image width in pixels (if available)
        - image_height: Image height in pixels (if available)
        - error: Error message if detection failed
    """
    try:
        logger.info(f"🔍 Detecting faces in image: {attachment_path}")
        
        from orchestrator.backend_tool_client import get_backend_tool_client
        
        client = await get_backend_tool_client()
        result = await client.detect_faces(
            attachment_path=attachment_path,
            user_id=user_id
        )
        
        if result.get("success"):
            face_count = result.get("face_count", 0)
            logger.info(f"✅ Face detection complete: {face_count} face(s) detected")
        else:
            logger.warning(f"⚠️ Face detection failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Face detection tool error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "faces": [],
            "face_count": 0,
            "error": str(e)
        }


async def identify_faces_in_image(
    attachment_path: str,
    user_id: str = "system",
    confidence_threshold: float = 0.85
) -> str:
    """
    Identify people in an attached image by matching faces against known identities.
    
    This tool detects faces in the image and matches them against the known_identities
    database to identify who the people are. Use this when users ask "who is this?"
    or want to identify people in photos.
    
    Args:
        attachment_path: Full path to the image file
        user_id: User ID for access control
        confidence_threshold: Minimum confidence for identity matches (default: 0.85)
        
    Returns:
        Human-readable string describing identified faces, or error message
    """
    try:
        logger.info(f"👤 Identifying faces in image: {attachment_path}")
        
        from orchestrator.backend_tool_client import get_backend_tool_client
        
        client = await get_backend_tool_client()
        result = await client.identify_faces(
            attachment_path=attachment_path,
            user_id=user_id,
            confidence_threshold=confidence_threshold
        )
        
        if not result.get("success"):
            error_msg = result.get("error", "Unknown error")
            logger.warning(f"⚠️ Face identification failed: {error_msg}")
            return f"Failed to identify faces: {error_msg}"
        
        face_count = result.get("face_count", 0)
        identified_faces = result.get("identified_faces", [])
        identified_count = result.get("identified_count", 0)
        
        if face_count == 0:
            return "No faces detected in the image."
        
        if identified_count == 0:
            return f"Detected {face_count} face(s) in the image, but no matching identities were found in the database."
        
        # Build human-readable description
        descriptions = []
        for face in identified_faces:
            identity_name = face.get("identity_name", "Unknown")
            confidence = face.get("confidence", 0.0)
            confidence_pct = int(confidence * 100)
            descriptions.append(f"{identity_name} (confidence: {confidence_pct}%)")
        
        if identified_count == 1:
            return f"Identified 1 person: {descriptions[0]}"
        else:
            return f"Identified {identified_count} person(s): {', '.join(descriptions)}"
        
    except Exception as e:
        logger.error(f"Face identification tool error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return f"Error identifying faces: {str(e)}"


# Tool list for registration
FACE_ANALYSIS_TOOLS = [
    detect_faces_in_image,
    identify_faces_in_image
]
