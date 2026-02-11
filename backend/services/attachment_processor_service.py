"""
Attachment Processor Service
Processes attached images for face detection and identification
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

# Lazy import - will import when needed to avoid proto generation issues
# from clients.image_vision_client import get_image_vision_client
from services.face_encoding_service import FaceEncodingService

logger = logging.getLogger(__name__)


class AttachmentProcessorService:
    """Service for processing chat attachment images"""
    
    def __init__(self):
        self.face_encoding_service = FaceEncodingService()
        self._face_service_initialized = False
    
    async def _ensure_face_service_initialized(self):
        """Ensure face encoding service is initialized"""
        if not self._face_service_initialized:
            try:
                await self.face_encoding_service.initialize()
                self._face_service_initialized = True
            except Exception as e:
                logger.warning(f"⚠️ Face encoding service initialization failed: {e}")
    
    async def process_image_for_search(
        self,
        attachment_path: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Process an attached image for face detection and identification
        
        Args:
            attachment_path: Path to the attached image file
            user_id: User ID for identity matching (user's known identities)
            
        Returns:
            Dict with:
            - face_encodings: List[List[float]] (128-dim encodings)
            - bounding_boxes: List[Dict] with x, y, width, height
            - detected_identities: List[str] (matched identity names)
            - face_count: int
        """
        try:
            # Lazy import to avoid import errors if proto not generated
            try:
                from clients.image_vision_client import get_image_vision_client
            except ImportError as e:
                logger.error(f"❌ Failed to import Image Vision client: {e}")
                return {
                    "face_encodings": [],
                    "bounding_boxes": [],
                    "detected_identities": [],
                    "face_count": 0,
                    "error": f"Image Vision client not available: {str(e)}"
                }
            
            # Initialize vision client
            vision_client = await get_image_vision_client()
            await vision_client.initialize(required=False)
            
            if not vision_client._initialized:
                logger.warning("⚠️ Image Vision Service unavailable - skipping face detection")
                return {
                    "face_encodings": [],
                    "bounding_boxes": [],
                    "detected_identities": [],
                    "face_count": 0,
                    "error": "Image Vision Service unavailable"
                }
            
            # Detect faces in the image
            # Use a temporary document_id for attachment processing
            temp_document_id = f"attachment_{user_id}_{Path(attachment_path).stem}"
            detection_result = await vision_client.detect_faces(
                image_path=attachment_path,
                document_id=temp_document_id
            )
            
            if not detection_result or not detection_result.get("faces"):
                return {
                    "face_encodings": [],
                    "bounding_boxes": [],
                    "detected_identities": [],
                    "face_count": 0
                }
            
            faces = detection_result["faces"]
            face_encodings = [face["face_encoding"] for face in faces]
            bounding_boxes = [
                {
                    "x": face["bbox_x"],
                    "y": face["bbox_y"],
                    "width": face["bbox_width"],
                    "height": face["bbox_height"]
                }
                for face in faces
            ]
            
            # Match faces against known identities using FaceEncodingService
            detected_identities = []
            if face_encodings:
                await self._ensure_face_service_initialized()
                
                try:
                    # Match each detected face against known identities via Vector Service
                    for face_encoding in face_encodings:
                        match_result = await self.face_encoding_service.match_face(
                            face_encoding=face_encoding,
                            confidence_threshold=0.82,  # Align with L2 < 0.6 same-person rule (cosine >= 0.82)
                            limit=1
                        )
                        
                        if match_result and match_result.get("matched_identity"):
                            identity_name = match_result["matched_identity"]
                            confidence = match_result.get("confidence", 0)
                            logger.info(f"✨ Matched face to '{identity_name}' ({confidence}% confidence)")
                            detected_identities.append(identity_name)
                        else:
                            logger.debug("No match found for face")
                            
                except Exception as match_error:
                    logger.warning(f"⚠️ Face matching failed: {match_error}")
                    import traceback
                    traceback.print_exc()
                    # Continue without identity matching
            
            return {
                "face_encodings": face_encodings,
                "bounding_boxes": bounding_boxes,
                "detected_identities": list(set(detected_identities)),  # Remove duplicates
                "face_count": len(faces),
                "image_width": detection_result.get("image_width"),
                "image_height": detection_result.get("image_height")
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to process image for search: {e}")
            return {
                "face_encodings": [],
                "bounding_boxes": [],
                "detected_identities": [],
                "face_count": 0,
                "error": str(e)
            }
    
    async def extract_image_features(
        self,
        attachment_path: str
    ) -> Optional[List[float]]:
        """
        Extract visual features for similarity search (future implementation)
        
        This would use a vision model to extract feature vectors for
        visual similarity search (e.g., CLIP embeddings)
        
        Returns:
            Feature vector or None if not implemented
        """
        # TODO: Implement visual feature extraction using CLIP or similar
        # For now, return None
        return None


# Global instance
attachment_processor_service = AttachmentProcessorService()
