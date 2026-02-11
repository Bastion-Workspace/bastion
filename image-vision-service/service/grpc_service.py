"""
gRPC Service Implementation - Image Vision Service
"""

import grpc
import logging
from typing import Dict, Any

# Import generated proto files (will be generated during Docker build)
import sys
sys.path.insert(0, '/app')

from protos import image_vision_pb2, image_vision_pb2_grpc

from service.vision_engine import VisionEngine
from config.settings import settings

logger = logging.getLogger(__name__)


class ImageVisionServiceImplementation(image_vision_pb2_grpc.ImageVisionServiceServicer):
    """Image Vision Service gRPC Implementation"""
    
    def __init__(self):
        self.vision_engine: VisionEngine = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize all components"""
        try:
            device = settings.get_device()
            self.vision_engine = VisionEngine(device=device)
            await self.vision_engine.initialize()

            try:
                await self.vision_engine.initialize_object_detection()
                logger.info("Object detection (YOLO + CLIP) pre-loaded at startup")
            except Exception as e:
                logger.warning("Object detection not pre-loaded at startup: %s", e)

            self._initialized = True
            logger.info("Image Vision Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Image Vision Service: {e}")
            raise
    
    async def HealthCheck(self, request, context):
        """Health check endpoint"""
        try:
            health = await self.vision_engine.health_check()
            
            return image_vision_pb2.HealthCheckResponse(
                status=health["status"],
                service_version=settings.SERVICE_VERSION,
                device=health["device"],
                details={
                    "model": health.get("model", "unknown"),
                    "face_recognition_available": str(health.get("face_recognition_available", False))
                }
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return image_vision_pb2.HealthCheckResponse(
                status="unhealthy",
                service_version=settings.SERVICE_VERSION,
                device="unknown",
                details={"error": str(e)}
            )
    
    async def DetectFaces(self, request, context):
        """Detect faces in an image"""
        try:
            if not self._initialized:
                return image_vision_pb2.FaceDetectionResponse(
                    faces=[],
                    image_width=0,
                    image_height=0,
                    processing_time_seconds=0.0,
                    error="Service not initialized"
                )
            
            # Detect faces
            result = await self.vision_engine.detect_faces(request.image_path)
            
            # Convert to protobuf format
            detected_faces = []
            for face in result["faces"]:
                detected_face = image_vision_pb2.DetectedFace(
                    bbox_x=face["bbox_x"],
                    bbox_y=face["bbox_y"],
                    bbox_width=face["bbox_width"],
                    bbox_height=face["bbox_height"],
                    face_encoding=face["face_encoding"],
                    confidence=face["confidence"]
                )
                detected_faces.append(detected_face)
            
            return image_vision_pb2.FaceDetectionResponse(
                faces=detected_faces,
                image_width=result["image_width"],
                image_height=result["image_height"],
                processing_time_seconds=result["processing_time_seconds"]
            )
            
        except FileNotFoundError as e:
            logger.error(f"Image not found: {e}")
            return image_vision_pb2.FaceDetectionResponse(
                faces=[],
                image_width=0,
                image_height=0,
                processing_time_seconds=0.0,
                error=f"Image not found: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return image_vision_pb2.FaceDetectionResponse(
                faces=[],
                image_width=0,
                image_height=0,
                processing_time_seconds=0.0,
                error=str(e)
            )
    
    async def MatchFaces(self, request, context):
        """Match unknown faces against known identities"""
        try:
            if not self._initialized:
                return image_vision_pb2.FaceMatchingResponse(
                    matches=[],
                    processing_time_seconds=0.0,
                    error="Service not initialized"
                )
            
            logger.info(f"Face matching: {len(request.unknown_faces)} unknown against {len(request.known_identities)} known")
            
            # Match faces
            result = await self.vision_engine.match_faces(
                unknown_faces=[list(face.face_encoding) for face in request.unknown_faces],
                known_identities={
                    identity.identity_name: list(identity.face_encoding)
                    for identity in request.known_identities
                },
                confidence_threshold=request.confidence_threshold or 0.82
            )
            
            # Convert to protobuf format
            matches = []
            for match in result["matches"]:
                matches.append(image_vision_pb2.FaceMatch(
                    face_index=match["face_index"],
                    matched_identity=match["matched_identity"],
                    confidence=match["confidence"]
                ))
            
            return image_vision_pb2.FaceMatchingResponse(
                matches=matches,
                processing_time_seconds=result["processing_time_seconds"]
            )
            
        except Exception as e:
            logger.error(f"Face matching failed: {e}")
            return image_vision_pb2.FaceMatchingResponse(
                matches=[],
                processing_time_seconds=0.0,
                error=str(e)
            )

    async def DetectObjects(self, request, context):
        """Detect objects in an image (YOLO + optional CLIP semantic matching)."""
        try:
            if not self._initialized:
                return image_vision_pb2.ObjectDetectionResponse(
                    objects=[],
                    image_width=0,
                    image_height=0,
                    processing_time_seconds=0.0,
                    error="Service not initialized"
                )

            class_filter = list(request.class_filter) if request.class_filter else None
            confidence_threshold = request.confidence_threshold if request.confidence_threshold > 0 else 0.5
            result = await self.vision_engine.detect_objects(
                image_path=request.image_path,
                class_filter=class_filter,
                confidence_threshold=confidence_threshold,
            )

            objects = []
            for obj in result["objects"]:
                objects.append(image_vision_pb2.DetectedObject(
                    class_name=obj["class_name"],
                    class_id=obj["class_id"],
                    confidence=obj["confidence"],
                    bbox_x=obj["bbox_x"],
                    bbox_y=obj["bbox_y"],
                    bbox_width=obj["bbox_width"],
                    bbox_height=obj["bbox_height"],
                    detection_method=obj.get("detection_method", "yolo"),
                    matched_description=obj.get("matched_description", ""),
                ))

            if request.semantic_descriptions:
                regions = [
                    {"bbox_x": o["bbox_x"], "bbox_y": o["bbox_y"], "bbox_width": o["bbox_width"], "bbox_height": o["bbox_height"]}
                    for o in result["objects"]
                ]
                semantic_matches = await self.vision_engine.match_objects_semantically(
                    image_path=request.image_path,
                    regions=regions,
                    object_descriptions=list(request.semantic_descriptions),
                    similarity_threshold=0.25,
                )
                for m in semantic_matches:
                    objects.append(image_vision_pb2.DetectedObject(
                        class_name=m["matched_description"],
                        class_id=0,
                        confidence=m["confidence"],
                        bbox_x=m["bbox_x"],
                        bbox_y=m["bbox_y"],
                        bbox_width=m["bbox_width"],
                        bbox_height=m["bbox_height"],
                        detection_method="clip_semantic",
                        matched_description=m["matched_description"],
                    ))
                # CLIP grid sweep: find regions matching semantic terms that YOLO may have missed
                sweep_matches = await self.vision_engine.find_semantic_regions(
                    image_path=request.image_path,
                    object_descriptions=list(request.semantic_descriptions),
                    chunk_size=224,
                    stride=224,
                    similarity_threshold=0.28,
                    max_chunks=200,
                    nms_iou_threshold=0.5,
                )
                for m in sweep_matches:
                    objects.append(image_vision_pb2.DetectedObject(
                        class_name=m["matched_description"],
                        class_id=0,
                        confidence=m["confidence"],
                        bbox_x=m["bbox_x"],
                        bbox_y=m["bbox_y"],
                        bbox_width=m["bbox_width"],
                        bbox_height=m["bbox_height"],
                        detection_method="clip_semantic",
                        matched_description=m["matched_description"],
                    ))

            return image_vision_pb2.ObjectDetectionResponse(
                objects=objects,
                image_width=result["image_width"],
                image_height=result["image_height"],
                processing_time_seconds=result["processing_time_seconds"]
            )

        except FileNotFoundError as e:
            logger.error(f"Image not found: {e}")
            return image_vision_pb2.ObjectDetectionResponse(
                objects=[],
                image_width=0,
                image_height=0,
                processing_time_seconds=0.0,
                error=f"Image not found: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return image_vision_pb2.ObjectDetectionResponse(
                objects=[],
                image_width=0,
                image_height=0,
                processing_time_seconds=0.0,
                error=str(e)
            )

    async def ExtractObjectFeatures(self, request, context):
        """Extract CLIP embeddings for a user-annotated region."""
        try:
            if not self._initialized:
                return image_vision_pb2.ObjectFeatureExtractionResponse(
                    error="Service not initialized"
                )

            bbox = {
                "bbox_x": request.bbox_x,
                "bbox_y": request.bbox_y,
                "bbox_width": request.bbox_width,
                "bbox_height": request.bbox_height,
            }
            result = await self.vision_engine.extract_object_features(
                image_path=request.image_path,
                bbox=bbox,
                description=request.description or "",
            )

            return image_vision_pb2.ObjectFeatureExtractionResponse(
                visual_embedding=result["visual_embedding"],
                semantic_embedding=result["semantic_embedding"],
                combined_embedding=result["combined_embedding"],
                embedding_dim=result["embedding_dim"],
            )

        except FileNotFoundError as e:
            logger.error(f"Image not found: {e}")
            return image_vision_pb2.ObjectFeatureExtractionResponse(error=f"Image not found: {str(e)}")
        except Exception as e:
            logger.error(f"Object feature extraction failed: {e}")
            return image_vision_pb2.ObjectFeatureExtractionResponse(error=str(e))
