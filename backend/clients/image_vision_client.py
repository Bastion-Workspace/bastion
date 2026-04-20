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
        self._config_disabled_logged = False

    def _vision_enabled(self) -> bool:
        return bool(getattr(self.settings, "IMAGE_VISION_ENABLED", True))

    def is_ready(self) -> bool:
        """True when image vision is enabled and the gRPC stub passed health check."""
        return self._vision_enabled() and self._initialized and self.stub is not None

    async def _reset_connection(self) -> None:
        if self.channel:
            try:
                await self.channel.close()
            except Exception:
                pass
        self.channel = None
        self.stub = None
        self._initialized = False

    async def reset_connection(self) -> None:
        """Clear gRPC channel and stub (e.g. after a failed health check)."""
        await self._reset_connection()

    async def initialize(self, required: bool = False):
        """Initialize the gRPC channel and stub

        Args:
            required: If True, raise exception on failure. If False, log warning and continue.
        """
        if not self._vision_enabled():
            if required:
                raise RuntimeError(
                    "Image vision is disabled (IMAGE_VISION_ENABLED=false) but a required "
                    "connection was requested"
                )
            if not self._config_disabled_logged:
                logger.info(
                    "Image vision service disabled by configuration (IMAGE_VISION_ENABLED=false)"
                )
                self._config_disabled_logged = True
            await self._reset_connection()
            return

        if self._initialized:
            return

        await self._reset_connection()

        try:
            logger.debug("Connecting to Image Vision Service at %s", self.service_url)

            options = [
                ("grpc.max_send_message_length", 10 * 1024 * 1024),
                ("grpc.max_receive_message_length", 10 * 1024 * 1024),
            ]
            self.channel = grpc.aio.insecure_channel(self.service_url, options=options)
            self.stub = image_vision_pb2_grpc.ImageVisionServiceStub(self.channel)

            health_request = image_vision_pb2.HealthCheckRequest()
            response = await self.stub.HealthCheck(health_request, timeout=5.0)

            if response.status == "healthy":
                logger.info(
                    "Connected to Image Vision Service v%s (device=%s)",
                    response.service_version,
                    response.device,
                )
                self._initialized = True
            else:
                msg = f"Image Vision Service health check returned: {response.status}"
                logger.warning(msg)
                await self._reset_connection()
                if required:
                    raise RuntimeError(msg)

        except Exception as e:
            logger.error("Failed to connect to Image Vision Service: %s", e)
            await self._reset_connection()
            if required:
                raise
            logger.warning("Image Vision Service unavailable - vision features disabled until retry")

    async def close(self):
        """Close the gRPC channel"""
        if self.channel:
            await self.channel.close()
        self.channel = None
        self.stub = None
        self._initialized = False
        logger.info("Image Vision Service client closed")

    async def detect_faces(
        self,
        image_path: str,
        document_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Detect faces in an image

        Args:
            image_path: Path to image file
            document_id: Document ID for logging

        Returns:
            Dict with faces, image dimensions, and processing time, or None if service unavailable
        """
        if not self._vision_enabled():
            return None
        if not self.is_ready():
            try:
                await self.initialize(required=False)
            except Exception as e:
                logger.error("Cannot detect faces: Image Vision Service unavailable: %s", e)
                return None

        if not self.is_ready():
            return None

        try:
            request = image_vision_pb2.FaceDetectionRequest(
                image_path=image_path,
                document_id=document_id,
            )

            response = await self.stub.DetectFaces(request, timeout=60.0)

            if response.error:
                logger.error("Face detection error: %s", response.error)
                return None

            faces = []
            for face in response.faces:
                faces.append(
                    {
                        "bbox_x": face.bbox_x,
                        "bbox_y": face.bbox_y,
                        "bbox_width": face.bbox_width,
                        "bbox_height": face.bbox_height,
                        "face_encoding": list(face.face_encoding),
                        "confidence": face.confidence,
                    }
                )

            return {
                "faces": faces,
                "image_width": response.image_width,
                "image_height": response.image_height,
                "processing_time_seconds": response.processing_time_seconds,
            }

        except grpc.RpcError as e:
            logger.error("gRPC error detecting faces: %s: %s", e.code(), e.details())
            await self._reset_connection()
            return None
        except Exception as e:
            logger.error("Error detecting faces: %s", e)
            return None

    async def match_faces(
        self,
        unknown_faces: List[list],
        known_identities: list,
        confidence_threshold: float = 0.82,
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
        if not self._vision_enabled():
            return None
        if not self.is_ready():
            try:
                await self.initialize(required=False)
            except Exception as e:
                logger.error("Cannot match faces: Image Vision Service unavailable: %s", e)
                return None

        if not self.is_ready():
            return None

        try:
            unknown_face_protos = [
                image_vision_pb2.UnknownFace(face_encoding=encoding) for encoding in unknown_faces
            ]

            known_identity_protos = [
                image_vision_pb2.KnownIdentity(
                    identity_name=identity["identity_name"],
                    face_encoding=identity["face_encoding"],
                    sample_count=identity.get("sample_count", 1),
                )
                for identity in known_identities
            ]

            request = image_vision_pb2.FaceMatchingRequest(
                unknown_faces=unknown_face_protos,
                known_identities=known_identity_protos,
                confidence_threshold=confidence_threshold,
            )

            response = await self.stub.MatchFaces(request, timeout=30.0)

            if response.error:
                logger.error("Face matching error: %s", response.error)
                return None

            matches = []
            for match in response.matches:
                matches.append(
                    {
                        "face_index": match.face_index,
                        "matched_identity": match.matched_identity,
                        "confidence": match.confidence,
                    }
                )

            return matches

        except grpc.RpcError as e:
            logger.error("gRPC error matching faces: %s: %s", e.code(), e.details())
            await self._reset_connection()
            return None
        except Exception as e:
            logger.error("Error matching faces: %s", e)
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
        if not self._vision_enabled():
            return None
        if not self.is_ready():
            try:
                await self.initialize(required=False)
            except Exception as e:
                logger.error("Cannot detect objects: Image Vision Service unavailable: %s", e)
                return None

        if not self.is_ready():
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
                logger.error("Object detection error: %s", response.error)
                return None

            objects = []
            for obj in response.objects:
                objects.append(
                    {
                        "class_name": obj.class_name,
                        "class_id": obj.class_id,
                        "confidence": obj.confidence,
                        "bbox_x": obj.bbox_x,
                        "bbox_y": obj.bbox_y,
                        "bbox_width": obj.bbox_width,
                        "bbox_height": obj.bbox_height,
                        "detection_method": obj.detection_method or "yolo",
                        "matched_description": obj.matched_description or "",
                    }
                )

            return {
                "objects": objects,
                "image_width": response.image_width,
                "image_height": response.image_height,
                "processing_time_seconds": response.processing_time_seconds,
            }

        except grpc.RpcError as e:
            logger.error("gRPC error detecting objects: %s: %s", e.code(), e.details())
            await self._reset_connection()
            return None
        except Exception as e:
            logger.error("Error detecting objects: %s", e)
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
        if not self._vision_enabled():
            return None
        if not self.is_ready():
            try:
                await self.initialize(required=False)
            except Exception as e:
                logger.error("Cannot extract object features: Image Vision Service unavailable: %s", e)
                return None

        if not self.is_ready():
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
                logger.error("Object feature extraction error: %s", response.error)
                return None

            return {
                "visual_embedding": list(response.visual_embedding),
                "semantic_embedding": list(response.semantic_embedding),
                "combined_embedding": list(response.combined_embedding),
                "embedding_dim": response.embedding_dim,
            }

        except grpc.RpcError as e:
            logger.error("gRPC error extracting object features: %s: %s", e.code(), e.details())
            await self._reset_connection()
            return None
        except Exception as e:
            logger.error("Error extracting object features: %s", e)
            return None


_image_vision_client: Optional[ImageVisionClient] = None


async def get_image_vision_client() -> ImageVisionClient:
    """Get or create global Image Vision Service client"""
    global _image_vision_client
    if _image_vision_client is None:
        _image_vision_client = ImageVisionClient()
    return _image_vision_client
