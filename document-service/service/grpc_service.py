"""
gRPC Service Implementation - Document Service (Entity extraction via spaCy)
"""

import logging
import sys

sys.path.insert(0, "/app")

import grpc
from protos import document_service_pb2, document_service_pb2_grpc

from service.entity_extractor import EntityExtractor

logger = logging.getLogger(__name__)

SERVICE_VERSION = "1.0.0"


class DocumentServiceImplementation(document_service_pb2_grpc.DocumentServiceServicer):
    """Document Service gRPC implementation: entity extraction using spaCy."""

    def __init__(self) -> None:
        self.entity_extractor = EntityExtractor()
        self._initialized = False

    async def initialize(self) -> None:
        """Load spaCy model."""
        try:
            await self.entity_extractor.initialize()
            self._initialized = True
            logger.info("Document Service initialized")
        except Exception as e:
            logger.error("Failed to initialize Document Service: %s", e)
            raise

    async def ExtractEntities(
        self,
        request: document_service_pb2.ExtractEntitiesRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.ExtractEntitiesResponse:
        """Extract entities from text using spaCy."""
        try:
            text = request.text or ""
            max_length = request.max_length if request.max_length > 0 else None
            entities = self.entity_extractor.extract(text, max_length=max_length)
            entity_messages = [
                document_service_pb2.ExtractedEntity(
                    name=e.name,
                    entity_type=e.entity_type,
                    confidence=e.confidence,
                    context=e.context,
                )
                for e in entities
            ]
            return document_service_pb2.ExtractEntitiesResponse(
                entities=entity_messages,
                success=True,
            )
        except Exception as e:
            logger.exception("ExtractEntities failed")
            return document_service_pb2.ExtractEntitiesResponse(
                success=False,
                error=str(e),
            )

    async def HealthCheck(
        self,
        request: document_service_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> document_service_pb2.HealthCheckResponse:
        """Health check with NER model load status (gliner_loaded = spaCy loaded for API compatibility)."""
        return document_service_pb2.HealthCheckResponse(
            status="healthy" if self.entity_extractor.is_loaded else "degraded",
            service_version=SERVICE_VERSION,
            gliner_loaded=self.entity_extractor.is_loaded,
        )
