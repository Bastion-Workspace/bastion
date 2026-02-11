"""
Service Container for Plato Knowledge Base
Implements dependency injection and shared service instances to eliminate duplication
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from config import settings
from repositories.document_repository import DocumentRepository
from services.settings_service import settings_service
from services.auth_service import auth_service
# Heavy services moved to lazy loading inside methods to avoid dependency bloat in Celery Beat/Flower

from utils.db_context import initialize_db_context, get_db_context

logger = logging.getLogger(__name__)

class ServiceContainer:
    """
    Centralized service container implementing dependency injection pattern.
    Eliminates service duplication and manages shared resources efficiently.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._services: Dict[str, Any] = {}
        self._initialized = False
        self._initializing = False
        self._initialization_lock = asyncio.Lock()
        self.config = config or {}
        
        # Core shared instances
        self.websocket_manager: Optional[Any] = None
        self.db_pool = None
        self.document_repository: Optional["DocumentRepository"] = None
        self.embedding_manager = None  # EmbeddingServiceWrapper (singleton)
        self.db_context = None
        
        # Service instances
        self.conversation_service: Optional[Any] = None # Placeholder, will be initialized later
        self.chat_service: Optional["ChatService"] = None

        self.knowledge_graph_service: Optional["KnowledgeGraphService"] = None
        self.collection_analysis_service: Optional["CollectionAnalysisService"] = None
        self.document_service: Optional["ParallelDocumentService"] = None
        self.enhanced_pdf_service: Optional["EnhancedPDFSegmentationService"] = None
        self.category_service: Optional["CategoryService"] = None

        self.rss_service: Optional[Any] = None
        self.file_manager: Optional[Any] = None
        self.folder_service: Optional["FolderService"] = None
        self.news_service: Optional[Any] = None
        self.direct_search_service: Optional[Any] = None

        # Research plan service removed - migrated to LangGraph subgraph workflows
        self.content_categorization_service: Optional[Any] = None
        self.conversation_context_service: Optional["ConversationContextService"] = None
        self.clarity_assessment_service: Optional["ClarityAssessmentService"] = None

    
    async def initialize(self) -> None:
        """Initialize all services in correct dependency order with shared resources"""
        async with self._initialization_lock:
            if self._initialized or self._initializing:
                return
            
            self._initializing = True
            logger.info("🔧 Initializing Service Container with optimized resource sharing...")
            
            try:
                # Phase 1: Initialize core infrastructure services
                await self._init_infrastructure_services()
                
                # Phase 2: Initialize shared resources
                await self._init_shared_resources()
                
                # Phase 3: Initialize business services
                await self._init_business_services()
                
                # Phase 4: Initialize specialized services
                await self._init_specialized_services()
                
                # Phase 5: Initialize FileManager after all other services are ready
                await self._init_file_manager()
                
                self._initialized = True
                logger.info("✅ Service Container initialized successfully")
                
            except Exception as e:
                logger.error(f"❌ Service Container initialization failed: {e}")
                self._initializing = False
                raise
            finally:
                self._initializing = False
    
    async def _init_infrastructure_services(self) -> None:
        """Initialize core infrastructure services"""
        logger.info("🔧 Phase 1: Initializing infrastructure services...")
        
        # Initialize settings service (already singleton)
        await self._retry_with_backoff(
            settings_service.initialize, 
            "SettingsService"
        )
        
        # Initialize authentication service (already singleton) - but delay until shared pool is ready
        # Auth service will be initialized later with shared pool
        
        # Removed: IntentClassificationService initialization (deprecated - never used)
        # DEPRECATED: SimpleIntentService removed - intent classification now handled by llm-orchestrator gRPC service
        
        logger.info("✅ Infrastructure services initialized")
    
    async def _init_shared_resources(self) -> None:
        """Initialize shared resources that will be reused across services"""
        logger.info("🔧 Phase 2: Initializing shared resources...")
        
        # Single WebSocket manager instance (optional: not available in Celery worker - no FastAPI)
        try:
            from utils.websocket_manager import WebSocketManager
            self.websocket_manager = WebSocketManager()
        except ImportError:
            self.websocket_manager = None
            logger.debug("WebSocket manager unavailable (e.g. Celery worker); real-time updates disabled")
        
        # Single document repository with shared connection pool
        self.document_repository = DocumentRepository()
        await self._retry_with_backoff(
            self.document_repository.initialize,
            "DocumentRepository"
        )
        self.db_pool = self.document_repository.pool
        
        # Initialize database context manager for RLS
        await initialize_db_context(self.db_pool)
        self.db_context = get_db_context()
        
        # Now initialize authentication service with shared pool
        logger.info("🔧 ROOSEVELT'S AUTH INITIALIZATION: Starting auth service initialization...")
        await self._retry_with_backoff(
            lambda: auth_service.initialize(self.db_pool), 
            "AuthenticationService"
        )
        logger.info("✅ ROOSEVELT'S AUTH INITIALIZATION: Auth service initialized successfully!")
        
        # Initialize embedding service wrapper (singleton)
        from services.embedding_service_wrapper import get_embedding_service
        self.embedding_manager = await get_embedding_service()
        logger.info("✅ Embedding service wrapper initialized (shared singleton)")
        
        logger.info("✅ Shared resources initialized")
    
    async def _init_business_services(self) -> None:
        """Initialize core business services using shared resources"""
        logger.info("🔧 Phase 3: Initializing business services...")
        
        # Conversation service (shared across multiple services)
        from services.conversation_service import ConversationService
        self.conversation_service = ConversationService()
        logger.info("✅ Conversation service initialized")
        
        # Knowledge graph service (single instance)
        from services.knowledge_graph_service import KnowledgeGraphService
        self.knowledge_graph_service = KnowledgeGraphService()
        await self._retry_with_backoff(
            self.knowledge_graph_service.initialize,
            "KnowledgeGraphService"
        )
        
        # Document service (single instance with shared resources)
        from services.parallel_document_service import ParallelDocumentService
        self.document_service = ParallelDocumentService()
        # Set WebSocket manager for real-time updates
        self.document_service.websocket_manager = self.websocket_manager
        await self._retry_with_backoff(
            lambda: self.document_service.initialize(
                shared_document_repository=self.document_repository,
                shared_embedding_manager=self.embedding_manager,
                shared_kg_service=self.knowledge_graph_service
            ),
            "ParallelDocumentService"
        )
        
        # Chat service (single instance with shared resources)
        from services.chat_service import ChatService
        self.chat_service = ChatService(self.websocket_manager)
        await self._retry_with_backoff(
            lambda: self.chat_service.initialize(
                shared_db_pool=self.db_pool,
                shared_embedding_manager=self.embedding_manager,
                shared_kg_service=self.knowledge_graph_service
            ),
            "ChatService"
        )
        
        # RSS service (migrated to tools-service)
        from tools_service.services.rss_service import get_rss_service
        self.rss_service = await get_rss_service(shared_db_pool=self.db_pool)

        # News service (single instance)
        from services.news_service import NewsService
        self.news_service = NewsService()
        await self._retry_with_backoff(
            lambda: self.news_service.initialize(shared_db_pool=self.db_pool),
            "NewsService"
        )
        
        # Direct search service
        from services.direct_search_service import DirectSearchService
        self.direct_search_service = DirectSearchService()
        
        # Folder service (single instance with shared resources)
        from services.folder_service import FolderService
        self.folder_service = FolderService()
        await self._retry_with_backoff(
            self.folder_service.initialize,
            "FolderService"
        )
        
        # FileManager service will be initialized lazily to avoid circular dependency
        # It will get services from this container when it initializes itself
        self.file_manager = None  # Will be set by get_file_manager() when needed
        
        logger.info("✅ Business services initialized")
    
    async def _init_specialized_services(self) -> None:
        """Initialize specialized services that depend on business services"""
        logger.info("🔧 Phase 4: Initializing specialized services...")
        
        # Collection analysis service
        from services.collection_analysis_service import CollectionAnalysisService
        self.collection_analysis_service = CollectionAnalysisService(self.chat_service)
        await self._retry_with_backoff(
            self.collection_analysis_service.initialize,
            "CollectionAnalysisService"
        )
        
        # Enhanced PDF segmentation service
        from services.enhanced_pdf_segmentation_service import EnhancedPDFSegmentationService
        self.enhanced_pdf_service = EnhancedPDFSegmentationService(
            self.document_repository, 
            self.embedding_manager
        )
        await self._retry_with_backoff(
            self.enhanced_pdf_service.initialize,
            "EnhancedPDFSegmentationService"
        )
        

        
        # Category Service
        from services.category_service import CategoryService
        self.category_service = CategoryService()
        await self._retry_with_backoff(
            lambda: self.category_service.initialize(shared_db_pool=self.db_pool),
            "CategoryService"
        )
        
        # Context-Aware Services (simplified - no MCP dependency)
        from services.pending_query_manager import PendingQueryManager
        self.pending_query_manager = PendingQueryManager(mcp_chat_service=None)
        await self.pending_query_manager.initialize()
        
        from services.conversation_context_service import ConversationContextService
        self.conversation_context_service = ConversationContextService(
            mcp_chat_service=None,
            db_pool=self.db_pool
        )
        
        from services.clarity_assessment_service import ClarityAssessmentService
        self.clarity_assessment_service = ClarityAssessmentService(
            mcp_chat_service=None,
            pending_query_manager=self.pending_query_manager
        )
        
        # Research Plan Service removed - migrated to LangGraph subgraph workflows
        
        # Content Categorization Service
        from services.content_categorization_service import get_content_categorization_service
        self.content_categorization_service = await get_content_categorization_service()
        await self._retry_with_backoff(
            self.content_categorization_service.initialize,
            "ContentCategorizationService"
        )
        
        # Set up service dependencies
        # self.chat_service.set_services(
        #     self.collection_analysis_service, 
        #     self.document_service
        # )
        
        logger.info("✅ Specialized services initialized")
    
    async def _init_file_manager(self) -> None:
        """Initialize FileManager service after all dependencies are ready"""
        logger.info("🔧 Phase 5: Initializing FileManager service...")
        
        try:
            # Initialize FileManager service - it will get dependencies via lazy loading
            from services.file_manager import get_file_manager
            self.file_manager = await get_file_manager()
            
            # Update FileManager with services from container
            await self.file_manager.update_services_from_container()
            
            logger.info("✅ FileManager service initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize FileManager service: {e}")
            raise
    
    async def _retry_with_backoff(self, func, service_name: str, max_retries: int = 5, 
                                 base_delay: float = 2, max_delay: float = 30) -> Any:
        """Retry service initialization with exponential backoff"""
        for attempt in range(max_retries):
            try:
                return await func()
            except Exception as e:
                delay = min(base_delay * (2 ** attempt), max_delay)
                if attempt < max_retries - 1:
                    logger.warning(f"🔄 {service_name} initialization attempt {attempt + 1} failed: {e}")
                    logger.info(f"⏱️  Retrying {service_name} in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"❌ {service_name} failed after {max_retries} attempts: {e}")
                    raise
    
    def get_service(self, service_name: str) -> Any:
        """Get a service instance by name"""
        if not self._initialized:
            raise RuntimeError("Service container not initialized")
        
        return getattr(self, service_name, None)
    
    # Research plan service getter removed - migrated to LangGraph subgraph workflows
    
    def get_conversation_context_service(self) -> "ConversationContextService":
        """Get the conversation context service"""
        if not self._initialized:
            raise RuntimeError("Service container not initialized")
        return self.conversation_context_service
    
    def get_clarity_assessment_service(self) -> "ClarityAssessmentService":
        """Get the clarity assessment service"""
        if not self._initialized:
            raise RuntimeError("Service container not initialized")
        return self.clarity_assessment_service
    
    async def close(self) -> None:
        """Gracefully close all services"""
        logger.info("🔄 Shutting down Service Container...")
        
        # Close services in reverse dependency order
        services_to_close = [

            ("category_service", self.category_service),
            ("enhanced_pdf_service", self.enhanced_pdf_service),
            ("collection_analysis_service", self.collection_analysis_service),
            ("chat_service", self.chat_service),
            ("document_service", self.document_service),
            ("knowledge_graph_service", self.knowledge_graph_service),
            ("conversation_service", self.conversation_service),
            ("embedding_manager", self.embedding_manager),
            ("document_repository", self.document_repository),
        ]
        
        for service_name, service in services_to_close:
            if service and hasattr(service, 'close'):
                try:
                    await service.close()
                    logger.info(f"✅ {service_name} closed")
                except Exception as e:
                    logger.warning(f"⚠️ Error closing {service_name}: {e}")
        
        # Close singleton services
        try:
            await auth_service.close()
            # Removed: intent_service.close() (service deprecated)
            logger.info("✅ Singleton services closed")
        except Exception as e:
            logger.warning(f"⚠️ Error closing singleton services: {e}")
        
        self._initialized = False
        logger.info("👋 Service Container shutdown complete")
    
    @property
    def is_initialized(self) -> bool:
        """Check if the container is fully initialized"""
        return self._initialized
    
    def get_stats(self) -> Dict[str, Any]:
        """Get container statistics"""
        return {
            "initialized": self._initialized,
            "initializing": self._initializing,
            "services_count": len([s for s in dir(self) if not s.startswith('_') and getattr(self, s) is not None]),
            "websocket_connections": self.websocket_manager.get_connection_count() if self.websocket_manager else 0,
            "embedding_workers": len(self.embedding_manager.workers) if self.embedding_manager else 0,
            "storage_workers": len(self.embedding_manager.storage_workers) if self.embedding_manager else 0,
        }

# Global service container instance
service_container = ServiceContainer()

async def get_service_container() -> ServiceContainer:
    """Get the global service container instance"""
    if not service_container.is_initialized:
        await service_container.initialize()
    return service_container
