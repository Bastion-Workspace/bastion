"""
Lazy Chat Service
Optimized ChatService with lazy loading for performance
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Any
import redis.asyncio as redis

from config import settings
from utils.openrouter_client import get_openrouter_client

logger = logging.getLogger(__name__)


class LazyChatService:
    """
    Lightweight ChatService for orchestrator agents that only loads components when needed
    This avoids initializing heavy components (embedding manager, knowledge graph) for simple queries
    """
    
    def __init__(self):
        self.openai_client = None
        self.redis_client = None
        self.current_model = None
        self.models_enabled = False
        self._embedding_manager = None
        self._kg_service = None
        self._conversation_service = None
        self._initialized = False
    
    async def initialize_minimal(self):
        """Initialize only the essential components for LLM calls"""
        if self._initialized:
            return
        
        logger.info("⚡ Initializing LazyChatService (minimal components only)...")
        start_time = datetime.now()
        
        # Initialize only essential components
        # Use OpenRouterClient wrapper for automatic reasoning support
        self.openai_client = get_openrouter_client()
        
        self.redis_client = redis.from_url(settings.REDIS_URL)
        
        # Initialize model selection (lightweight)
        await self._auto_select_model()
        
        self._initialized = True
        init_duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"⚡ LazyChatService minimal init completed in {init_duration:.2f}s")
    
    async def _auto_select_model(self):
        """Auto-select model if only one is enabled (optimized version)"""
        try:
            from services.settings_service import settings_service
            
            # Ensure settings service is initialized
            if not hasattr(settings_service, '_initialized') or not settings_service._initialized:
                await settings_service.initialize()
            
            # Get enabled models
            enabled_models = await settings_service.get_enabled_models()
            
            if len(enabled_models) >= 1:
                # Use first enabled model (simple and fast)
                self.current_model = enabled_models[0]
                self.models_enabled = True
                logger.debug(f"⚡ Fast model selection: {self.current_model}")
            else:
                logger.error("❌ No models enabled for LazyChatService")
                self.models_enabled = False
                
        except Exception as e:
            logger.error(f"❌ Model selection failed in LazyChatService: {e}")
            self.models_enabled = False
    
    async def get_embedding_manager(self):
        """Lazy load embedding manager only when needed"""
        if self._embedding_manager is None:
            logger.info("🔄 Lazy loading embedding service wrapper...")
            from services.embedding_service_wrapper import get_embedding_service
            self._embedding_manager = await get_embedding_service()
            logger.debug("✅ Embedding service wrapper lazy loaded")
        return self._embedding_manager
    
    async def get_kg_service(self):
        """Lazy load knowledge graph service only when needed"""
        if self._kg_service is None:
            logger.info("🔄 Lazy loading knowledge graph service...")
            from services.knowledge_graph_service import KnowledgeGraphService
            self._kg_service = KnowledgeGraphService()
            await self._kg_service.initialize()
            logger.debug("✅ Knowledge graph service lazy loaded")
        return self._kg_service
    
    async def get_conversation_service(self):
        """Lazy load conversation service only when needed"""
        if self._conversation_service is None:
            logger.debug("🔄 Lazy loading conversation service...")
            from services.conversation_service import ConversationService
            self._conversation_service = ConversationService()
            logger.debug("✅ Conversation service lazy loaded")
        return self._conversation_service
    
    def is_ready_for_llm_calls(self) -> bool:
        """Check if service is ready for basic LLM calls"""
        return (
            self._initialized and 
            self.openai_client is not None and 
            self.current_model is not None and 
            self.models_enabled
        )
    
    async def simple_llm_call(self, messages: list, stream: bool = False, **kwargs):
        """Make a simple LLM call without heavy dependencies"""
        if not self.is_ready_for_llm_calls():
            await self.initialize_minimal()
        
        if not self.is_ready_for_llm_calls():
            raise Exception("LazyChatService not ready for LLM calls")
        
        # Make the LLM call with optional streaming
        response = await self.openai_client.chat.completions.create(
            messages=messages,
            model=self.current_model,
            stream=stream,
            **kwargs
        )
        
        return response
    
    async def stream_llm_call(self, messages: list, **kwargs):
        """Make a streaming LLM call for real-time responses"""
        if not self.is_ready_for_llm_calls():
            await self.initialize_minimal()
        
        if not self.is_ready_for_llm_calls():
            raise Exception("LazyChatService not ready for LLM calls")
        
        # Make streaming LLM call
        stream = await self.openai_client.chat.completions.create(
            messages=messages,
            model=self.current_model,
            stream=True,
            **kwargs
        )
        
        async for chunk in stream:
            yield chunk
    
    def get_status(self) -> dict:
        """Get current status of lazy chat service"""
        return {
            "initialized": self._initialized,
            "current_model": self.current_model,
            "models_enabled": self.models_enabled,
            "openai_client_ready": self.openai_client is not None,
            "redis_client_ready": self.redis_client is not None,
            "embedding_manager_loaded": self._embedding_manager is not None,
            "kg_service_loaded": self._kg_service is not None,
            "conversation_service_loaded": self._conversation_service is not None,
        }
