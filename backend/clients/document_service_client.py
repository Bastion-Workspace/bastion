"""
Document Service gRPC Client

Provides client interface to the Document Service for entity extraction (spaCy).
"""

import grpc
import logging
from typing import List, Optional

from config import get_settings
from protos import document_service_pb2, document_service_pb2_grpc

from models.api_models import Entity

logger = logging.getLogger(__name__)


class DocumentServiceClient:
    """Client for Document Service entity extraction via gRPC."""

    def __init__(self, service_url: Optional[str] = None) -> None:
        self.settings = get_settings()
        self.service_url = service_url or self.settings.DOCUMENT_SERVICE_URL
        self.channel: Optional[grpc.aio.Channel] = None
        self.stub: Optional[document_service_pb2_grpc.DocumentServiceStub] = None
        self._initialized = False

    async def initialize(self, required: bool = False) -> None:
        """Initialize the gRPC channel and stub."""
        if self._initialized:
            return
        try:
            logger.debug("Connecting to Document Service at %s", self.service_url)
            # Keepalive at 5 min to avoid GOAWAY "too_many_pings" (ENHANCE_YOUR_CALM) from server/proxy
            options = [
                ("grpc.max_send_message_length", 100 * 1024 * 1024),
                ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                ("grpc.keepalive_time_ms", 300000),  # 5 min
                ("grpc.keepalive_timeout_ms", 20000),
                ("grpc.keepalive_permit_without_calls", 1),
            ]
            self.channel = grpc.aio.insecure_channel(self.service_url, options=options)
            self.stub = document_service_pb2_grpc.DocumentServiceStub(self.channel)
            health_request = document_service_pb2.HealthCheckRequest()
            response = await self.stub.HealthCheck(health_request, timeout=10.0)
            if response.status in ("healthy", "degraded") and response.gliner_loaded:
                logger.info(
                    "Connected to Document Service v%s (NER model loaded)",
                    response.service_version,
                )
                self._initialized = True
            elif response.status == "degraded":
                logger.warning("Document Service health check: %s", response.status)
                self._initialized = True
            else:
                logger.warning("Document Service health check returned: %s", response.status)
        except Exception as e:
            logger.error("Failed to connect to Document Service: %s", e)
            if required:
                raise
            logger.warning("Document Service unavailable - entity extraction will fail until connected")

    async def close(self) -> None:
        """Close the gRPC channel."""
        if self.channel:
            await self.channel.close()
            self._initialized = False
            logger.info("Document Service client closed")

    async def extract_entities(
        self, text: str, max_length: Optional[int] = None
    ) -> List[Entity]:
        """
        Extract entities from text using the document-service (spaCy).

        Args:
            text: Raw document text.
            max_length: Optional max character length (service default if 0 or None).

        Returns:
            List of Entity models (name, entity_type, confidence, source_chunk, metadata).
        """
        if not self._initialized:
            try:
                await self.initialize(required=True)
            except Exception as e:
                logger.error("Cannot extract entities: Document Service unavailable: %s", e)
                raise RuntimeError("Document Service is not available") from e
        try:
            request = document_service_pb2.ExtractEntitiesRequest(
                text=text,
                max_length=max_length or 0,
            )
            response = await self.stub.ExtractEntities(request, timeout=120.0)
            if not response.success:
                raise RuntimeError(response.error or "ExtractEntities failed")
            return [
                Entity(
                    name=e.name,
                    entity_type=e.entity_type,
                    confidence=e.confidence,
                    source_chunk="",
                    metadata={"source": "spacy", "context": e.context or ""},
                )
                for e in response.entities
            ]
        except grpc.RpcError as e:
            logger.error("Document Service RPC error: %s", e)
            raise RuntimeError(f"Document Service error: {e}") from e
