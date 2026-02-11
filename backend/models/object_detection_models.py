"""
Pydantic models for object detection API requests and responses.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates (x, y, width, height)."""

    x: int = Field(..., description="Left edge x")
    y: int = Field(..., description="Top edge y")
    width: int = Field(..., description="Width in pixels")
    height: int = Field(..., description="Height in pixels")


class ObjectDetectionRequest(BaseModel):
    """Request for object detection on an image."""

    class_filter: Optional[List[str]] = Field(None, description="YOLO class names to include")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum YOLO confidence")
    semantic_descriptions: Optional[List[str]] = Field(
        None, description="Text descriptions for CLIP semantic matching"
    )
    match_user_annotations: bool = Field(True, description="Match against user-defined objects")


class AnnotateObjectRequest(BaseModel):
    """Request for creating a user-defined object annotation."""

    object_name: str = Field(..., description="User-defined name for the object")
    description: str = Field("", description="Optional text description")
    bbox: BoundingBox = Field(..., description="Bounding box drawn by user")


class AddExampleRequest(BaseModel):
    """Request for adding another example to an existing annotation."""

    document_id: str = Field(..., description="Document containing the example")
    bbox: BoundingBox = Field(..., description="Bounding box of the example")


class UpdateDetectedObjectRequest(BaseModel):
    """Request to update a detected object: tag (refined label) and/or reject (hide)."""

    user_tag: Optional[str] = Field(None, description="Refined label for search/display (e.g. 'BMW i3')")
    rejected: Optional[bool] = Field(None, description="If true, hide this detection from lists/overlays")
