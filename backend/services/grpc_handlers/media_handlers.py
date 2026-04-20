"""gRPC handlers for Media operations (images, face/object analysis, audio)."""

import logging

import grpc
from config import settings
from protos import tool_service_pb2

logger = logging.getLogger(__name__)


class MediaHandlersMixin:
    """Mixin providing Media gRPC handlers.

    Mixed into ToolServiceImplementation; accesses self._get_search_service(),
    self._get_document_repo(), etc. via standard Python MRO.
    """

    # ===== Image Search Operations =====
    
    async def SearchImages(
        self,
        request: tool_service_pb2.ImageSearchRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ImageSearchResponse:
        """Search for images with metadata sidecars"""
        try:
            if request.is_random:
                logger.info(f"🎲 SearchImages (RANDOM): type={request.image_type}, author={request.author}, series={request.series}, limit={request.limit}")
            else:
                logger.info(f"SearchImages: query={request.query[:100]}, type={request.image_type}, date={request.date}, author={request.author}, series={request.series}")
            
            from tools_service.services.media_ops import search_images_for_grpc

            exclude_ids = list(request.exclude_document_ids) if request.exclude_document_ids else None
            packed = await search_images_for_grpc(
                query=request.query,
                image_type=request.image_type if request.image_type else None,
                date=request.date if request.date else None,
                author=request.author if request.author else None,
                series=request.series if request.series else None,
                limit=request.limit or 10,
                user_id=request.user_id if request.user_id else None,
                is_random=request.is_random,
                exclude_document_ids=exclude_ids,
            )

            if packed.get("format") == "structured":
                images_markdown = packed.get("images_markdown", "")
                metadata_list = packed.get("metadata", [])
                structured_images = packed.get("structured_images", [])
                pb_metadata = []
                for meta in metadata_list:
                    pb_metadata.append(
                        tool_service_pb2.ImageMetadata(
                            title=meta.get("title", ""),
                            date=meta.get("date", ""),
                            series=meta.get("series", ""),
                            author=meta.get("author", ""),
                            content=meta.get("content", ""),
                            tags=meta.get("tags", []),
                            image_type=meta.get("image_type", ""),
                        )
                    )
                response = tool_service_pb2.ImageSearchResponse(
                    results=images_markdown,
                    success=True,
                    metadata=pb_metadata,
                )
                if structured_images and hasattr(response, "structured_images_json"):
                    import json

                    response.structured_images_json = json.dumps(structured_images)
                return response

            return tool_service_pb2.ImageSearchResponse(
                results=packed.get("results_text", ""),
                success=True,
            )
            
        except Exception as e:
            logger.error(f"SearchImages error: {e}")
            import traceback
            traceback.print_exc()
            return tool_service_pb2.ImageSearchResponse(
                results="",
                success=False,
                error=str(e)
            )
    
    # ===== Face Analysis Operations =====
    
    async def DetectFaces(
        self,
        request: tool_service_pb2.DetectFacesRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.DetectFacesResponse:
        """Detect faces in an attached image"""
        try:
            logger.info(f"🔍 DetectFaces: path={request.attachment_path}, user={request.user_id}")
            
            # Lazy import to avoid import errors if proto not generated
            try:
                from clients.image_vision_client import get_image_vision_client
            except ImportError as e:
                logger.error(f"❌ Failed to import Image Vision client: {e}")
                return tool_service_pb2.DetectFacesResponse(
                    success=False,
                    face_count=0,
                    error=f"Image Vision client not available: {str(e)}"
                )
            
            from pathlib import Path
            
            vision_client = await get_image_vision_client()
            if not settings.IMAGE_VISION_ENABLED:
                return tool_service_pb2.DetectFacesResponse(
                    success=False,
                    face_count=0,
                    error="Image vision is disabled (IMAGE_VISION_ENABLED=false)",
                )
            await vision_client.initialize(required=False)

            if not vision_client.is_ready():
                logger.warning("Image Vision Service unavailable")
                return tool_service_pb2.DetectFacesResponse(
                    success=False,
                    face_count=0,
                    error="Image Vision Service unavailable"
                )
            
            # Use temporary document_id for attachment processing
            temp_document_id = f"attachment_{request.user_id}_{Path(request.attachment_path).stem}"
            detection_result = await vision_client.detect_faces(
                image_path=request.attachment_path,
                document_id=temp_document_id
            )
            
            if not detection_result or not detection_result.get("faces"):
                return tool_service_pb2.DetectFacesResponse(
                    success=True,
                    face_count=0,
                    image_width=detection_result.get("image_width") if detection_result else None,
                    image_height=detection_result.get("image_height") if detection_result else None
                )
            
            faces = detection_result["faces"]
            pb_faces = []
            
            for face in faces:
                pb_face = tool_service_pb2.FaceDetection(
                    face_encoding=face.get("face_encoding", []),
                    bbox_x=face.get("bbox_x", 0),
                    bbox_y=face.get("bbox_y", 0),
                    bbox_width=face.get("bbox_width", 0),
                    bbox_height=face.get("bbox_height", 0)
                )
                pb_faces.append(pb_face)
            
            return tool_service_pb2.DetectFacesResponse(
                success=True,
                faces=pb_faces,
                face_count=len(faces),
                image_width=detection_result.get("image_width"),
                image_height=detection_result.get("image_height")
            )
            
        except Exception as e:
            logger.error(f"DetectFaces error: {e}")
            import traceback
            traceback.print_exc()
            return tool_service_pb2.DetectFacesResponse(
                success=False,
                face_count=0,
                error=str(e)
            )
    
    async def IdentifyFaces(
        self,
        request: tool_service_pb2.IdentifyFacesRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.IdentifyFacesResponse:
        """Identify people in an attached image by matching against known identities"""
        try:
            confidence_threshold = request.confidence_threshold if request.HasField("confidence_threshold") else 0.82
            logger.info(f"👤 IdentifyFaces: path={request.attachment_path}, user={request.user_id}, threshold={confidence_threshold}")

            from tools_service.services.media_ops import identify_faces_for_grpc

            result = await identify_faces_for_grpc(
                attachment_path=request.attachment_path,
                user_id=request.user_id,
                confidence_threshold=confidence_threshold,
            )
            if not result.get("success"):
                return tool_service_pb2.IdentifyFacesResponse(
                    success=False,
                    face_count=0,
                    identified_count=0,
                    error=result.get("error"),
                )
            pb_identified_faces = [
                tool_service_pb2.IdentifiedFace(**f) for f in result.get("identified_faces", [])
            ]
            return tool_service_pb2.IdentifyFacesResponse(
                success=True,
                identified_faces=pb_identified_faces,
                face_count=result.get("face_count", 0),
                identified_count=result.get("identified_count", 0),
            )
            
        except Exception as e:
            logger.error(f"IdentifyFaces error: {e}")
            import traceback
            traceback.print_exc()
            return tool_service_pb2.IdentifyFacesResponse(
                success=False,
                face_count=0,
                identified_count=0,
                error=str(e)
            )

    async def DetectObjects(
        self,
        request: tool_service_pb2.DetectObjectsRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.DetectObjectsResponse:
        """Detect objects in an image (YOLO + optional CLIP semantic matching)."""
        try:
            try:
                from clients.image_vision_client import get_image_vision_client
            except ImportError as e:
                logger.error(f"Failed to import Image Vision client: {e}")
                return tool_service_pb2.DetectObjectsResponse(
                    success=False,
                    object_count=0,
                    error=f"Image Vision client not available: {str(e)}"
                )

            vision_client = await get_image_vision_client()
            if not settings.IMAGE_VISION_ENABLED:
                return tool_service_pb2.DetectObjectsResponse(
                    success=False,
                    object_count=0,
                    error="Image vision is disabled (IMAGE_VISION_ENABLED=false)",
                )
            await vision_client.initialize(required=False)

            if not vision_client.is_ready():
                return tool_service_pb2.DetectObjectsResponse(
                    success=False,
                    object_count=0,
                    error="Image Vision Service unavailable"
                )

            class_filter = list(request.class_filter) if request.class_filter else None
            semantic_descriptions = list(request.semantic_descriptions) if request.semantic_descriptions else None
            confidence_threshold = request.confidence_threshold if request.confidence_threshold > 0 else 0.5

            detection_result = await vision_client.detect_objects(
                image_path=request.attachment_path,
                document_id=request.document_id or "",
                class_filter=class_filter,
                confidence_threshold=confidence_threshold,
                semantic_descriptions=semantic_descriptions,
            )

            if not detection_result or not detection_result.get("objects"):
                return tool_service_pb2.DetectObjectsResponse(
                    success=True,
                    object_count=0,
                    image_width=detection_result.get("image_width") if detection_result else None,
                    image_height=detection_result.get("image_height") if detection_result else None,
                    processing_time_seconds=detection_result.get("processing_time_seconds") if detection_result else None,
                )

            objects = detection_result["objects"]
            pb_objects = []
            for obj in objects:
                pb_objects.append(tool_service_pb2.DetectedObjectProto(
                    class_name=obj.get("class_name", ""),
                    class_id=obj.get("class_id", 0),
                    confidence=obj.get("confidence", 0.0),
                    bbox_x=obj.get("bbox_x", 0),
                    bbox_y=obj.get("bbox_y", 0),
                    bbox_width=obj.get("bbox_width", 0),
                    bbox_height=obj.get("bbox_height", 0),
                    detection_method=obj.get("detection_method", "yolo"),
                    matched_description=obj.get("matched_description", ""),
                ))

            return tool_service_pb2.DetectObjectsResponse(
                success=True,
                objects=pb_objects,
                object_count=len(objects),
                image_width=detection_result.get("image_width"),
                image_height=detection_result.get("image_height"),
                processing_time_seconds=detection_result.get("processing_time_seconds"),
            )

        except Exception as e:
            logger.error(f"DetectObjects error: {e}")
            import traceback
            traceback.print_exc()
            return tool_service_pb2.DetectObjectsResponse(
                success=False,
                object_count=0,
                error=str(e)
            )

    async def IdentifyObjects(
        self,
        request: tool_service_pb2.IdentifyObjectsRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.IdentifyObjectsResponse:
        """Identify objects in an image (YOLO + user-defined annotation matching)."""
        try:
            try:
                from services.object_detection_service import get_object_detection_service
            except ImportError as e:
                logger.error(f"Failed to import Object Detection service: {e}")
                return tool_service_pb2.IdentifyObjectsResponse(
                    success=False,
                    object_count=0,
                    identified_count=0,
                    error=f"Object Detection service not available: {str(e)}"
                )

            obj_service = await get_object_detection_service()
            result = await obj_service.detect_objects_in_image(
                document_id=request.document_id or "",
                image_path=request.attachment_path,
                user_id=request.user_id,
                class_filter=list(request.class_filter) if request.class_filter else None,
                confidence_threshold=request.confidence_threshold if request.confidence_threshold > 0 else 0.5,
                semantic_descriptions=list(request.semantic_descriptions) if request.semantic_descriptions else None,
                match_user_annotations=request.match_user_annotations,
                user_annotation_threshold=request.user_annotation_threshold if request.user_annotation_threshold > 0 else 0.75,
            )

            if result.get("error"):
                return tool_service_pb2.IdentifyObjectsResponse(
                    success=False,
                    object_count=0,
                    identified_count=0,
                    error=result.get("error")
                )

            objects = result.get("objects", [])
            pb_objects = []
            for obj in objects:
                annotation_id_str = str(obj["annotation_id"]) if obj.get("annotation_id") else None
                pb_objects.append(tool_service_pb2.IdentifiedObjectProto(
                    class_name=obj.get("class_name", ""),
                    confidence=obj.get("confidence", 0.0),
                    bbox_x=obj.get("bbox_x", 0),
                    bbox_y=obj.get("bbox_y", 0),
                    bbox_width=obj.get("bbox_width", 0),
                    bbox_height=obj.get("bbox_height", 0),
                    detection_method=obj.get("detection_method", "yolo"),
                    annotation_id=annotation_id_str,
                ))

            return tool_service_pb2.IdentifyObjectsResponse(
                success=True,
                identified_objects=pb_objects,
                object_count=len(objects),
                identified_count=len([o for o in objects if o.get("detection_method") == "user_defined"]),
            )

        except Exception as e:
            logger.error(f"IdentifyObjects error: {e}")
            import traceback
            traceback.print_exc()
            return tool_service_pb2.IdentifyObjectsResponse(
                success=False,
                object_count=0,
                identified_count=0,
                error=str(e)
            )

    # ===== Image Generation Operations =====

    async def GenerateImage(
        self,
        request: tool_service_pb2.ImageGenerationRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.ImageGenerationResponse:
        """Generate images using OpenRouter image models"""
        try:
            logger.info(f"🎨 GenerateImage: prompt={request.prompt[:100]}...")
            
            from tools_service.services.media_ops import generate_images_for_grpc

            model = request.model if request.HasField("model") and request.model else None
            reference_image_data = None
            reference_image_url = None
            reference_strength = 0.5
            if request.HasField("reference_image_data") and request.reference_image_data:
                reference_image_data = request.reference_image_data
                logger.info("📎 Using reference_image_data for image-to-image generation")
            elif request.HasField("reference_image_url") and request.reference_image_url:
                reference_image_url = request.reference_image_url
                logger.info(
                    "📎 Using reference_image_url for image-to-image generation: %s",
                    reference_image_url[:100],
                )
            if request.HasField("reference_strength"):
                reference_strength = request.reference_strength
            folder_id = request.folder_id if request.HasField("folder_id") and request.folder_id else None

            result = await generate_images_for_grpc(
                prompt=request.prompt,
                size=request.size if request.size else "1024x1024",
                fmt=request.format if request.format else "png",
                seed=request.seed if request.HasField("seed") else None,
                num_images=request.num_images if request.num_images else 1,
                negative_prompt=request.negative_prompt if request.HasField("negative_prompt") else None,
                model=model,
                reference_image_data=reference_image_data,
                reference_image_url=reference_image_url,
                reference_strength=reference_strength,
                user_id=request.user_id if request.user_id else None,
                folder_id=folder_id,
            )
            
            # Convert result to proto response
            if result.get("success"):
                images = []
                for img in result.get("images", []):
                    gi = tool_service_pb2.GeneratedImage(
                        filename=img.get("filename", ""),
                        path=img.get("path", ""),
                        url=img.get("url", ""),
                        width=img.get("width", 1024),
                        height=img.get("height", 1024),
                        format=img.get("format", "png")
                    )
                    doc_id = img.get("document_id")
                    if doc_id and hasattr(gi, "document_id"):
                        gi.document_id = doc_id
                    images.append(gi)
                
                response = tool_service_pb2.ImageGenerationResponse(
                    success=True,
                    model=result.get("model", ""),
                    prompt=result.get("prompt", request.prompt),
                    size=result.get("size", "1024x1024"),
                    format=result.get("format", "png"),
                    images=images
                )
                logger.info(f"✅ Generated {len(images)} image(s) successfully")
                return response
            else:
                # Error occurred
                error_msg = result.get("error", "Unknown error")
                logger.error(f"❌ Image generation failed: {error_msg}")
                response = tool_service_pb2.ImageGenerationResponse(
                    success=False,
                    error=error_msg
                )
                return response
            
        except Exception as e:
            logger.error(f"❌ GenerateImage error: {e}")
            response = tool_service_pb2.ImageGenerationResponse(
                success=False,
                error=str(e)
            )
            return response

    async def GetReferenceImageForObject(
        self,
        request: tool_service_pb2.GetReferenceImageForObjectRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.GetReferenceImageForObjectResponse:
        """Return image bytes for an object name (from detected/annotated images) for use as reference in image generation."""
        try:
            from services.langgraph_tools.object_detection_tools import get_reference_image_bytes_for_object
            obj_name = (request.object_name or "").strip()
            user_id = request.user_id or "system"
            if not obj_name:
                return tool_service_pb2.GetReferenceImageForObjectResponse(
                    success=False,
                    error="object_name required"
                )
            result = await get_reference_image_bytes_for_object(object_name=obj_name, user_id=user_id)
            if not result:
                return tool_service_pb2.GetReferenceImageForObjectResponse(
                    success=False,
                    error="No image found for this object"
                )
            img_bytes, document_id = result
            return tool_service_pb2.GetReferenceImageForObjectResponse(
                success=True,
                reference_image_data=img_bytes,
                document_id=document_id
            )
        except Exception as e:
            logger.error("GetReferenceImageForObject error: %s", e)
            return tool_service_pb2.GetReferenceImageForObjectResponse(
                success=False,
                error=str(e)
            )

    async def TranscribeAudio(
        self,
        request: tool_service_pb2.TranscribeAudioRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.TranscribeAudioResponse:
        """Transcribe audio file to text (stub - not yet implemented)"""
        try:
            logger.info(f"🎤 TranscribeAudio: file_path={request.audio_file_path[:100] if request.audio_file_path else 'None'}...")
            
            # Get audio transcription service
            from services.audio_transcription_service import audio_transcription_service
            await audio_transcription_service.initialize()
            
            # Call transcription service (stub)
            result = await audio_transcription_service.transcribe_audio(
                file_path=request.audio_file_path,
                language=request.language if request.HasField("language") and request.language else None,
                model=request.model if request.HasField("model") and request.model else None,
                user_id=request.user_id
            )
            
            # Convert result to proto response
            response = tool_service_pb2.TranscribeAudioResponse(
                success=result.get("success", False),
                transcript=result.get("transcript", ""),
                language_detected=result.get("language_detected") if result.get("language_detected") else None
            )
            
            # Add segments if available
            segments = result.get("segments", [])
            for seg in segments:
                response.segments.append(
                    tool_service_pb2.TranscriptSegment(
                        start_time_ms=seg.get("start_time_ms", 0),
                        end_time_ms=seg.get("end_time_ms", 0),
                        text=seg.get("text", ""),
                        confidence=seg.get("confidence", 1.0)
                    )
                )
            
            if result.get("error"):
                response.error = result["error"]
            
            if not result.get("success"):
                logger.warning(f"⚠️ Audio transcription not yet implemented: {result.get('error', 'Unknown error')}")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ TranscribeAudio error: {e}")
            response = tool_service_pb2.TranscribeAudioResponse(
                success=False,
                transcript="",
                error=str(e)
            )
            return response
    
