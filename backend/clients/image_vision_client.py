"""
Image Vision Service gRPC Client

Provides client interface to the Image Vision Service for face detection.
"""

import grpc
import logging
from typing import Optional, List, Dict, Any

from config import get_settings
from protos import image_vision_pb2, image_vision_pb2_grpc

logger = logging.getLogger(__name__)


class ImageVisionClient:
    """Client for interacting with the Image Vision Service via gRPC"""
    
    def __init__(self, service_url: Optional[str] = None):
        """
        Initialize Image Vision Service client
        
        Args:
            service_url: gRPC service URL (default: from config)
        """
        self.settings = get_settings()
        self.service_url = service_url or self.settings.IMAGE_VISION_SERVICE_URL
        self.channel: Optional[grpc.aio.Channel] = None
        self.stub: Optional[image_vision_pb2_grpc.ImageVisionServiceStub] = None
        self._initialized = False
    
    async def initialize(self, required: bool = False):
        """Initialize the gRPC channel and stub
        
        Args:
            required: If True, raise exception on failure. If False, log warning and continue.
        """
        if self._initialized:
            return
        
        try:
            logger.debug(f"Connecting to Image Vision Service at {self.service_url}")
            
            # Create insecure channel with increased message size limits
            # Face encodings are 128 floats = ~512 bytes per face, but allow for large images
            options = [
                ('grpc.max_send_message_length', 10 * 1024 * 1024),  # 10 MB
                ('grpc.max_receive_message_length', 10 * 1024 * 1024),  # 10 MB
            ]
            self.channel = grpc.aio.insecure_channel(self.service_url, options=options)
            self.stub = image_vision_pb2_grpc.ImageVisionServiceStub(self.channel)
            
            # Test connection
            health_request = image_vision_pb2.HealthCheckRequest()
            response = await self.stub.HealthCheck(health_request, timeout=5.0)
            
            if response.status == "healthy":
                logger.info(f"✅ Connected to Image Vision Service v{response.service_version}")
                logger.info(f"   Device: {response.device}")
                self._initialized = True
            else:
                logger.warning(f"⚠️ Image Vision Service health check returned: {response.status}")
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to Image Vision Service: {e}")
            if required:
                raise
            else:
                logger.warning("⚠️ Image Vision Service unavailable - face detection disabled")
                logger.warning("⚠️ Vision features will be retried when needed")
    
    async def close(self):
        """Close the gRPC channel"""
        if self.channel:
            await self.channel.close()
            self._initialized = False
            logger.info("Image Vision Service client closed")
    
    async def detect_faces(
        self,
        image_path: str,
        document_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Detect faces in an image
        
        Args:
            image_path: Path to image file
            document_id: Document ID for logging
            
        Returns:
            Dict with faces, image dimensions, and processing time, or None if service unavailable
        """
        if not self._initialized:
            logger.info("Image Vision Service not initialized, attempting to connect...")
            try:
                await self.initialize(required=False)
            except Exception as e:
                logger.error(f"❌ Cannot detect faces: Image Vision Service unavailable: {e}")
                return None
        
        if not self._initialized:
            return None
        
        try:
            request = image_vision_pb2.FaceDetectionRequest(
                image_path=image_path,
                document_id=document_id
            )
            
            response = await self.stub.DetectFaces(request, timeout=60.0)  # Face detection can take 30-60s on CPU
            
            if response.error:
                logger.error(f"❌ Face detection error: {response.error}")
                return None
            
            # Convert protobuf response to dict
            faces = []
            for face in response.faces:
                faces.append({
                    "bbox_x": face.bbox_x,
                    "bbox_y": face.bbox_y,
                    "bbox_width": face.bbox_width,
                    "bbox_height": face.bbox_height,
                    "face_encoding": list(face.face_encoding),
                    "confidence": face.confidence
                })
            
            return {
                "faces": faces,
                "image_width": response.image_width,
                "image_height": response.image_height,
                "processing_time_seconds": response.processing_time_seconds
            }
            
        except grpc.RpcError as e:
            logger.error(f"❌ gRPC error detecting faces: {e.code()}: {e.details()}")
            # Mark as uninitialized so next call will retry connection
            self._initialized = False
            return None
        except Exception as e:
            logger.error(f"❌ Error detecting faces: {e}")
            return None
    
    async def match_faces(
        self,
        unknown_faces: List[list],
        known_identities: list,
        confidence_threshold: float = 0.82
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Match unknown faces against known identities
        
        Args:
            unknown_faces: List of face encodings (each is a 128-dimensional list)
            known_identities: List of dicts with identity_name and face_encoding
            confidence_threshold: Minimum confidence (0.0-1.0) for a match. 0.82 aligns with L2 < 0.6 same-person rule.
            
        Returns:
            List of matches with face_index, matched_identity, and confidence, or None if service unavailable
        """
        if not self._initialized:
            logger.info("Image Vision Service not initialized, attempting to connect...")
            try:
                await self.initialize(required=False)
            except Exception as e:
                logger.error(f"❌ Cannot match faces: Image Vision Service unavailable: {e}")
                return None
        
        if not self._initialized:
            return None
        
        try:
            # Build protobuf request
            unknown_face_protos = [
                image_vision_pb2.UnknownFace(face_encoding=encoding)
                for encoding in unknown_faces
            ]
            
            known_identity_protos = [
                image_vision_pb2.KnownIdentity(
                    identity_name=identity["identity_name"],
                    face_encoding=identity["face_encoding"],
                    sample_count=identity.get("sample_count", 1)
                )
                for identity in known_identities
            ]
            
            request = image_vision_pb2.FaceMatchingRequest(
                unknown_faces=unknown_face_protos,
                known_identities=known_identity_protos,
                confidence_threshold=confidence_threshold
            )
            
            response = await self.stub.MatchFaces(request, timeout=30.0)
            
            if response.error:
                logger.error(f"❌ Face matching error: {response.error}")
                return None
            
            # Convert protobuf response to list of dicts
            matches = []
            for match in response.matches:
                matches.append({
                    "face_index": match.face_index,
                    "matched_identity": match.matched_identity,
                    "confidence": match.confidence
                })
            
            return matches
            
        except grpc.RpcError as e:
            logger.error(f"❌ gRPC error matching faces: {e.code()}: {e.details()}")
            self._initialized = False
            return None
        except Exception as e:
            logger.error(f"❌ Error matching faces: {e}")
            return None

    async def detect_objects(
        self,
        image_path: str,
        document_id: str,
        class_filter: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
        semantic_descriptions: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Detect objects in an image (YOLO + optional CLIP semantic matching).

        Args:
            image_path: Path to image file.
            document_id: Document ID for logging.
            class_filter: Optional list of COCO class names to include.
            confidence_threshold: Minimum YOLO confidence (0.0-1.0).
            semantic_descriptions: Optional text descriptions for CLIP matching.

        Returns:
            Dict with objects, image_width, image_height, processing_time_seconds, or None if unavailable.
        """
        if not self._initialized:
            logger.info("Image Vision Service not initialized, attempting to connect...")
            try:
                await self.initialize(required=False)
            except Exception as e:
                logger.error(f"Cannot detect objects: Image Vision Service unavailable: {e}")
                return None

        if not self._initialized:
            return None

        try:
            request = image_vision_pb2.ObjectDetectionRequest(
                image_path=image_path,
                document_id=document_id,
                class_filter=class_filter or [],
                confidence_threshold=confidence_threshold,
                semantic_descriptions=semantic_descriptions or [],
            )

            response = await self.stub.DetectObjects(request, timeout=120.0)

            if response.error:
                logger.error(f"Object detection error: {response.error}")
                return None

            objects = []
            for obj in response.objects:
                objects.append({
                    "class_name": obj.class_name,
                    "class_id": obj.class_id,
                    "confidence": obj.confidence,
                    "bbox_x": obj.bbox_x,
                    "bbox_y": obj.bbox_y,
                    "bbox_width": obj.bbox_width,
                    "bbox_height": obj.bbox_height,
                    "detection_method": obj.detection_method or "yolo",
                    "matched_description": obj.matched_description or "",
                })

            return {
                "objects": objects,
                "image_width": response.image_width,
                "image_height": response.image_height,
                "processing_time_seconds": response.processing_time_seconds,
            }

        except grpc.RpcError as e:
            logger.error(f"gRPC error detecting objects: {e.code()}: {e.details()}")
            self._initialized = False
            return None
        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return None

    async def extract_object_features(
        self,
        image_path: str,
        bbox: Dict[str, int],
        description: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Extract CLIP embeddings for a user-annotated region.

        Args:
            image_path: Path to image file.
            bbox: Dict with bbox_x, bbox_y, bbox_width, bbox_height (or x, y, width, height).
            description: User text description.

        Returns:
            Dict with visual_embedding, semantic_embedding, combined_embedding (lists), embedding_dim, or None.
        """
        if not self._initialized:
            logger.info("Image Vision Service not initialized, attempting to connect...")
            try:
                await self.initialize(required=False)
            except Exception as e:
                logger.error(f"Cannot extract object features: Image Vision Service unavailable: {e}")
                return None

        if not self._initialized:
            return None

        try:
            x = bbox.get("bbox_x", bbox.get("x", 0))
            y = bbox.get("bbox_y", bbox.get("y", 0))
            w = bbox.get("bbox_width", bbox.get("width", 0))
            h = bbox.get("bbox_height", bbox.get("height", 0))

            request = image_vision_pb2.ObjectFeatureExtractionRequest(
                image_path=image_path,
                bbox_x=x,
                bbox_y=y,
                bbox_width=w,
                bbox_height=h,
                description=description,
            )

            response = await self.stub.ExtractObjectFeatures(request, timeout=60.0)

            if response.error:
                logger.error(f"Object feature extraction error: {response.error}")
                return None

            return {
                "visual_embedding": list(response.visual_embedding),
                "semantic_embedding": list(response.semantic_embedding),
                "combined_embedding": list(response.combined_embedding),
                "embedding_dim": response.embedding_dim,
            }

        except grpc.RpcError as e:
            logger.error(f"gRPC error extracting object features: {e.code()}: {e.details()}")
            self._initialized = False
            return None
        except Exception as e:
            logger.error(f"Error extracting object features: {e}")
            return None


# Global client instance
_image_vision_client: Optional[ImageVisionClient] = None


async def get_image_vision_client() -> ImageVisionClient:
    """Get or create global Image Vision Service client"""
    global _image_vision_client
    if _image_vision_client is None:
        _image_vision_client = ImageVisionClient()
    return _image_vision_client
