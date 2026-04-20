"""
gRPC Tool Service - Backend Data Access for LLM Orchestrator
Provides document, RSS, entity, weather, and org-mode data via gRPC
"""

import asyncio
import logging
from typing import Optional

import grpc
from protos import tool_service_pb2_grpc

from repositories.document_repository import DocumentRepository
from services.direct_search_service import DirectSearchService
from services.embedding_service_wrapper import get_embedding_service
from services.grpc_handlers import (
    OrgTodoHandlersMixin,
    WebHandlersMixin,
    EmailM365HandlersMixin,
    AgentMessagingHandlersMixin,
    RssHandlersMixin,
    MediaHandlersMixin,
    DataWorkspaceHandlersMixin,
    NavigationHandlersMixin,
    SearchUtilityHandlersMixin,
    AnalysisHandlersMixin,
    AgentProfileHandlersMixin,
    AgentSkillsHandlersMixin,
    AgentRuntimeHandlersMixin,
    AgentExecutionTraceHandlersMixin,
    ConnectorMcpHandlersMixin,
    DataConnectorBuilderHandlersMixin,
    ControlPaneHandlersMixin,
    AgentFactoryCrudHandlersMixin,
)


logger = logging.getLogger(__name__)


class ToolServiceImplementation(
    OrgTodoHandlersMixin,
    WebHandlersMixin,
    EmailM365HandlersMixin,
    AgentMessagingHandlersMixin,
    RssHandlersMixin,
    MediaHandlersMixin,
    DataWorkspaceHandlersMixin,
    NavigationHandlersMixin,
    SearchUtilityHandlersMixin,
    AnalysisHandlersMixin,
    AgentProfileHandlersMixin,
    AgentSkillsHandlersMixin,
    AgentRuntimeHandlersMixin,
    AgentExecutionTraceHandlersMixin,
    ConnectorMcpHandlersMixin,
    DataConnectorBuilderHandlersMixin,
    ControlPaneHandlersMixin,
    AgentFactoryCrudHandlersMixin,
    tool_service_pb2_grpc.ToolServiceServicer,
):
    """
    gRPC Tool Service Implementation
    
    Provides data access methods for the LLM Orchestrator service.
    Uses repositories directly for Phase 2 (services via container in Phase 3).
    """
    
    def __init__(self):
        logger.info("Initializing gRPC Tool Service...")
        # Use direct search service for document operations
        self._search_service: Optional[DirectSearchService] = None
        self._document_repo: Optional[DocumentRepository] = None
        self._embedding_manager = None  # EmbeddingServiceWrapper
    
    async def _get_search_service(self) -> DirectSearchService:
        """Lazy initialization of search service"""
        if not self._search_service:
            self._search_service = DirectSearchService()
        return self._search_service
    
    async def _get_embedding_manager(self):
        """Lazy initialization of embedding service wrapper"""
        if not self._embedding_manager:
            self._embedding_manager = await get_embedding_service()
        return self._embedding_manager
    
    def _get_document_repo(self) -> DocumentRepository:
        """Lazy initialization of document repository"""
        if not self._document_repo:
            self._document_repo = DocumentRepository()
        return self._document_repo
    


async def serve_tool_service(port: int = 50052):
    """
    Start the gRPC tool service server
    
    Runs alongside the main FastAPI server to provide data access
    for the LLM orchestrator service.
    """
    try:
        # Import health checking inside function (lesson learned!)
        from grpc_health.v1 import health, health_pb2, health_pb2_grpc
        
        logger.info(f"Starting gRPC Tool Service on port {port}...")
        
        # Create gRPC server with increased message size limits
        # Default is 4MB, increase to 100MB for large document search responses
        options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100 MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100 MB
        ]
        server = grpc.aio.server(options=options)
        
        # Register tool service
        tool_service = ToolServiceImplementation()
        tool_service_pb2_grpc.add_ToolServiceServicer_to_server(tool_service, server)
        
        # Register health checking
        health_servicer = health.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
        health_servicer.set(
            "tool_service.ToolService",
            health_pb2.HealthCheckResponse.SERVING
        )
        
        # Bind to port (use 0.0.0.0 for IPv4 compatibility)
        server.add_insecure_port(f'0.0.0.0:{port}')
        
        # Start server
        await server.start()
        logger.info(f"✅ gRPC Tool Service listening on port {port}")

        # Single sync authority: populate skills vector collection with retry until vector-service is ready.
        async def _sync_skills_with_retry():
            max_attempts = 5
            for attempt in range(max_attempts):
                delay = 2 ** (attempt + 1)  # 2, 4, 8, 16, 32 seconds
                if attempt > 0:
                    logger.info("Skills sync attempt %d/%d in %ds...", attempt + 1, max_attempts, delay)
                    await asyncio.sleep(delay)
                try:
                    from clients.vector_service_client import get_vector_service_client
                    client = await get_vector_service_client(required=False)
                    if not getattr(client, "_initialized", False):
                        await client.initialize(required=False)
                    if not getattr(client, "_initialized", False):
                        continue
                    await client.health_check()
                except Exception as e:
                    logger.debug("Vector service not ready: %s", e)
                    continue
                try:
                    from services.skill_vector_service import sync_all_skills
                    count = await sync_all_skills(user_id=None)
                    if count > 0:
                        logger.info("Skills vector collection populated with %d built-in skills", count)
                        return
                    else:
                        logger.info("Skills sync returned 0 skills (DB may not be seeded yet), retrying...")
                        continue
                except Exception as e:
                    logger.warning("Startup skills sync failed (attempt %d/%d): %s", attempt + 1, max_attempts, e)
            logger.error("Skills sync failed after %d attempts; vector-service may be unreachable", max_attempts)

        asyncio.create_task(_sync_skills_with_retry())

        # Wait for termination
        await server.wait_for_termination()
        
    except Exception as e:
        logger.error(f"❌ gRPC Tool Service failed to start: {e}")
        raise

