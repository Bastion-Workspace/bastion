"""
Chat Service - Handles natural language queries and RAG
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import List, AsyncGenerator, Dict, Any, Optional
import redis.asyncio as redis

from config import settings
from utils.openrouter_client import get_openrouter_client
from models.api_models import *
from models.conversation_models import *
from services.embedding_service_wrapper import get_embedding_service
from utils.deduplication_manager import deduplication_manager
from services.knowledge_graph_service import KnowledgeGraphService
from services.conversation_service import ConversationService

logger = logging.getLogger(__name__)


class ChatService:
    """Service for handling chat queries and RAG"""
    
    def __init__(self, websocket_manager=None):
        self.openai_client = None
        self.redis_client = None
        self.embedding_manager = None
        self.kg_service = None
        self.collection_analysis_service = None
        self.document_service = None
        self.conversation_service = None
        self.websocket_manager = websocket_manager
        self.current_model = None  # Will be set during initialization
        self.models_enabled = False  # Track if any models are enabled
        self.current_user_id = "default-user"  # Default for single-user mode
        self._available_models_cache: Optional[List[ModelInfo]] = None  # Cache OpenRouter models
        self._models_cache_timestamp: Optional[float] = None  # Cache timestamp
    
    async def initialize(self, shared_db_pool=None, shared_embedding_manager=None, shared_kg_service=None):
        """Initialize chat service with parallel component loading and shared dependencies"""
        logger.debug("🔧 Initializing Chat Service with ROOSEVELT'S PARALLEL OPTIMIZATION...")
        
        # Initialize synchronous components first (fast)
        # Use OpenRouterClient wrapper for automatic reasoning support
        self.openai_client = get_openrouter_client()
        self.redis_client = redis.from_url(settings.REDIS_URL)
        self.conversation_service = ConversationService()
        
        # Create parallel initialization tasks for heavy components
        async def init_embedding_manager():
            # Use shared embedding manager if provided, otherwise get wrapper
            if shared_embedding_manager:
                self.embedding_manager = shared_embedding_manager
                logger.debug("✅ Using shared embedding manager")
            else:
                self.embedding_manager = await get_embedding_service()
                logger.debug("✅ Embedding service wrapper initialized")
        
        async def init_knowledge_graph():
            # ROOSEVELT FIX: Use shared knowledge graph service if provided
            if shared_kg_service:
                self.kg_service = shared_kg_service
                logger.debug("✅ Using shared knowledge graph service")
            else:
                self.kg_service = KnowledgeGraphService()
                await self.kg_service.initialize()
                logger.debug("✅ Knowledge graph service initialized")
        
        async def init_model_selection():
            # Ensure model is selected and auto-select if needed
            await self.ensure_model_selected()
            await self._auto_select_model()
            logger.debug("✅ Model selection completed")
        
        # Execute heavy initializations in parallel
        logger.debug("⚡ Running parallel component initialization...")
        start_time = datetime.now()
        
        await asyncio.gather(
            init_embedding_manager(),
            init_knowledge_graph(),
            init_model_selection(),
            return_exceptions=True
        )
        
        parallel_duration = (datetime.now() - start_time).total_seconds()
        logger.debug(f"✅ Chat Service initialized with parallel optimization in {parallel_duration:.2f}s")
    
    def set_current_user(self, user_id: str):
        """Set the current user for operations (for multi-user support)"""
        self.current_user_id = user_id
        # Conversation service handles user context internally via lifecycle manager
    
    async def process_query(self, query: str, session_id: str, conversation_id: str = None) -> QueryResponse:
        """Process a natural language query using intelligent retrieval with iterative analysis"""
        start_time = time.time()
        
        try:
            # Check current enabled models status (refresh from database)
            await self._refresh_model_status()
            
            # Check if any models are enabled
            if not self.models_enabled or not self.current_model:
                logger.error("❌ No LLM models are enabled - chat functionality disabled")
                return QueryResponse(
                    answer="I'm sorry, but no language models are currently enabled. Please go to Settings and enable at least one LLM model to use the chat functionality.",
                    citations=[],
                    session_id=session_id,
                    query_time=time.time() - start_time,
                    retrieval_count=0
                )
            
            logger.info(f"🔍 Processing intelligent query: {query[:100]}...")
            
            # Step 0: Preprocess temporal queries
            expanded_query = self._expand_temporal_query(query)
            if expanded_query != query:
                logger.info(f"🕒 Expanded temporal query: {expanded_query}")
                query = expanded_query
            
            # Step 1: Check if this is a collection analysis query
            collection_response = await self._detect_and_handle_collection_analysis(query, session_id)
            if collection_response:
                return collection_response
            
            # All queries now use hybrid search for scalability and performance
            # This works effectively even for broad queries like "themes across documents"
            
            # Step 2: Extract and filter entities from query
            query_entities = await self.kg_service.extract_entities_from_text(query)
            raw_entity_names = [entity['name'] for entity in query_entities]
            
            # Filter and validate entities
            entity_names = self._filter_and_validate_entities(raw_entity_names, query)
            logger.info(f"🔍 Raw entities: {raw_entity_names}")
            logger.info(f"🔍 Filtered entities: {entity_names}")
            
            # Step 3: Phase 1 - Always start with chunk-based retrieval
            logger.info("🔍 Phase 1: Starting with chunk-based retrieval")
            relevant_chunks = await self._hybrid_retrieval(query, entity_names)
            logger.info(f"📚 Retrieved {len(relevant_chunks)} relevant chunks")
            
            # Step 4: Phase 2 - Enhanced LLM assessment with document selection capability
            assessment_result = await self._assess_chunk_sufficiency_with_document_selection(query, relevant_chunks)
            chunk_sufficiency = assessment_result.get("sufficiency", "SUFFICIENT")
            requested_documents = assessment_result.get("requested_documents", [])
            reasoning = assessment_result.get("reasoning", "")
            
            logger.info(f"🧠 Enhanced chunk assessment: {chunk_sufficiency}")
            if reasoning:
                logger.info(f"🧠 LLM reasoning: {reasoning}")
            if requested_documents:
                logger.info(f"🧠 LLM requested documents: {requested_documents}")
            
            # Step 5: Handle insufficient chunks with LLM-guided document selection
            if chunk_sufficiency == "INSUFFICIENT":
                if requested_documents:
                    logger.info(f"📚 Using LLM-guided document selection: {len(requested_documents)} documents requested")
                    full_doc_chunks, is_iterative = await self._intelligent_document_retrieval_with_llm_selection(
                        query, relevant_chunks, requested_documents, entity_names
                    )
                    
                    if is_iterative:
                        logger.info("📚 Using iterative analysis for LLM-selected documents")
                        return await self._iterative_document_analysis(query, full_doc_chunks, session_id, entity_names)
                    else:
                        relevant_chunks = full_doc_chunks
                        logger.info(f"📚 Using {len(relevant_chunks)} chunks from LLM-selected documents")
                else:
                    # Special handling for headline queries - use targeted headline retrieval instead of full document processing
                    query_lower = query.lower()
                    if any(pattern in query_lower for pattern in ['headline', 'headlines', 'top stories', 'news summary']):
                        logger.info("📰 Using targeted headline retrieval instead of full document processing")
                        headline_chunks = await self._targeted_headline_retrieval(query, entity_names)
                        relevant_chunks = headline_chunks
                        logger.info(f"📰 Retrieved {len(relevant_chunks)} headline-focused chunks")
                    else:
                        logger.info("📚 Falling back to automatic document retrieval (no specific documents requested)")
                        full_doc_chunks, is_iterative = await self._intelligent_full_document_retrieval(query, entity_names)
                        if is_iterative:
                            logger.info("📚 Using iterative analysis for automatically selected documents")
                            return await self._iterative_document_analysis(query, full_doc_chunks, session_id, entity_names)
                        else:
                            relevant_chunks = full_doc_chunks
                            logger.info(f"📚 Upgraded to {len(relevant_chunks)} chunks via automatic document retrieval")
            
            # Step 4: Prepare context with entity information
            context_parts = []
            citations = []
            entity_context = await self._build_entity_context(entity_names)
            
            # Add entity context if available
            if entity_context:
                context_parts.append(f"[Entity Context]: {entity_context}")
            
            for i, chunk in enumerate(relevant_chunks):
                context_parts.append(f"[Source {i+1}]: {chunk['content']}")
                
                # Get document metadata for better citation
                doc_metadata = await self._get_document_metadata(chunk['document_id'])
                doc_title = doc_metadata.get('title') or doc_metadata.get('filename', f"Document {chunk['document_id']}") if doc_metadata else f"Document {chunk['document_id']}"
                
                # Check if this chunk comes from a PDF segment
                segment_info = await self._get_segment_info_for_chunk(chunk['chunk_id'])
                
                # Create visual citation data if segment exists
                visual_citation = None
                if segment_info:
                    visual_citation = {
                        "page_number": segment_info.get('page_number'),
                        "page_id": segment_info.get('page_id'),
                        "segment_id": segment_info.get('segment_id'),
                        "segment_type": segment_info.get('segment_type'),
                        "bounds": segment_info.get('bounds'),
                        "page_image_url": f"/api/files/page_{segment_info.get('page_id')}.png" if segment_info.get('page_id') else None,
                        "highlight_text": self._extract_highlight_text(chunk['content'], query)
                    }
                
                # Create citation with visual information and extended snippet
                extended_snippet = self._create_extended_snippet(chunk['content'], query)
                citation = Citation(
                    document_id=chunk['document_id'],
                    document_title=doc_title,
                    chunk_id=chunk['chunk_id'],
                    relevance_score=chunk['score'],
                    snippet=extended_snippet,
                    segment_id=segment_info.get('segment_id') if segment_info else None,
                    page_number=segment_info.get('page_number') if segment_info else None,
                    segment_type=segment_info.get('segment_type') if segment_info else None,
                    segment_bounds=segment_info.get('bounds') if segment_info else None,
                    visual_citation=visual_citation
                )
                citations.append(citation)
            
            # Step 5: Build context and generate response
            context_text = "\n\n".join(context_parts)
            conversation_history = await self._get_recent_conversation(session_id)
            
            # Generate response with entity-aware prompting and metadata context
            answer = await self._generate_entity_aware_response(query, context_text, conversation_history, entity_names, relevant_chunks)
            
            query_time = time.time() - start_time
            
            response = QueryResponse(
                answer=answer,
                citations=citations,
                session_id=session_id,
                query_time=query_time,
                retrieval_count=len(relevant_chunks)
            )
            
            # Store in conversation history (both Redis and persistent)
            await self._store_query_history(session_id, query, response, conversation_id)
            
            logger.info(f"✅ Hybrid query completed in {query_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"❌ Hybrid query processing failed: {e}")
            # Fallback response
            return QueryResponse(
                answer=f"I apologize, but I encountered an error while processing your query: {str(e)}. Please try again or contact support if the issue persists.",
                citations=[],
                session_id=session_id,
                query_time=time.time() - start_time,
                retrieval_count=0
            )
    
    async def stream_response(self, query: str, session_id: str) -> AsyncGenerator[str, None]:
        """Stream response for real-time chat"""
        # Placeholder streaming implementation
        response_parts = [
            "This is a streaming response to your query. ",
            "The full implementation would use the OpenAI streaming API ",
            "to provide real-time responses as the LLM generates them. ",
            "This allows for better user experience with long responses."
        ]
        
        for part in response_parts:
            yield part
            await asyncio.sleep(0.1)  # Simulate streaming delay
    
    async def get_query_history(self, session_id: str, limit: int) -> List[QueryHistoryItem]:
        """Get conversation history for a session"""
        try:
            history_key = f"conversation:{session_id}"
            history_data = await self.redis_client.get(history_key)
            
            if not history_data:
                return []
            
            history = json.loads(history_data)
            
            # Convert to QueryHistoryItem objects
            history_items = []
            for item in history[-limit:]:  # Get last N items
                citations = [
                    Citation(
                        document_id=c["document_id"],
                        document_title=f"Document {c['document_id']}",
                        chunk_id=c["chunk_id"],
                        relevance_score=c["relevance_score"],
                        snippet=""  # Not stored in history for space
                    )
                    for c in item.get("citations", [])
                ]
                
                history_items.append(QueryHistoryItem(
                    timestamp=datetime.fromisoformat(item["timestamp"]),
                    query=item["query"],
                    answer=item["answer"],
                    citations=citations
                ))
            
            return history_items
            
        except Exception as e:
            logger.error(f"❌ Failed to get query history: {e}")
            return []
    
    async def get_available_models(self) -> List[ModelInfo]:
        """Get available OpenRouter models with intelligent caching"""
        import time

        # Check cache first (cache for 1 hour = 3600 seconds)
        CACHE_DURATION = 3600
        current_time = time.time()

        if (self._available_models_cache is not None and
            self._models_cache_timestamp is not None and
            (current_time - self._models_cache_timestamp) < CACHE_DURATION):
            cache_age = int(current_time - self._models_cache_timestamp)
            logger.info(f"📋 Using cached models ({len(self._available_models_cache)} models, {cache_age}s old)")
            return self._available_models_cache

        try:
            logger.info("🔍 Fetching fresh models from OpenRouter API...")

            # Call OpenRouter models API
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={
                        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                        "HTTP-Referer": settings.SITE_URL,
                        "X-Title": "Bastion AI Workspace"
                    },
                    timeout=30.0
                )

                if response.status_code != 200:
                    logger.error(f"❌ OpenRouter API returned status {response.status_code}: {response.text}")
                    return self._get_fallback_models()

                data = response.json()
                models_data = data.get("data", [])

                logger.info(f"✅ Retrieved {len(models_data)} fresh models from OpenRouter")

                # Convert to ModelInfo objects
                models = []
                for model_data in models_data:
                    try:
                        # Extract full pricing information from OpenRouter
                        pricing = model_data.get("pricing", {})
                        input_cost = float(pricing.get("prompt", "0")) if pricing.get("prompt") else None
                        output_cost = float(pricing.get("completion", "0")) if pricing.get("completion") else None
                        request_cost = float(pricing.get("request", "0")) if pricing.get("request") else None
                        image_cost = float(pricing.get("image", "0")) if pricing.get("image") else None

                        # Extract context length
                        context_length = model_data.get("context_length", 0)
                        if isinstance(context_length, str):
                            context_length = int(context_length) if context_length.isdigit() else 0

                        # Extract provider from model ID
                        model_id = model_data.get("id", "")
                        provider = model_id.split("/")[0] if "/" in model_id else "Unknown"
                        provider = provider.replace("-", " ").title()

                        # Create model info with full OpenRouter metadata
                        model_info = ModelInfo(
                            id=model_id,
                            canonical_slug=model_data.get("canonical_slug"),
                            name=model_data.get("name", model_id),
                            provider=provider,
                            context_length=context_length,
                            input_cost=input_cost,
                            output_cost=output_cost,
                            request_cost=request_cost,
                            image_cost=image_cost,
                            description=model_data.get("description"),
                            supported_parameters=model_data.get("supported_parameters"),
                            architecture=model_data.get("architecture"),
                            top_provider=model_data.get("top_provider"),
                            per_request_limits=model_data.get("per_request_limits"),
                            created=model_data.get("created")
                        )
                        
                        models.append(model_info)
                        
                    except Exception as model_error:
                        logger.warning(f"⚠️ Failed to parse model {model_data.get('id', 'unknown')}: {model_error}")
                        continue
                
                # Sort models by provider and name for better organization
                models.sort(key=lambda m: (m.provider, m.name))

                # Update cache with fresh data
                self._available_models_cache = models
                self._models_cache_timestamp = current_time
                logger.info(f"💾 Cached {len(models)} models for future use")

                logger.info(f"✅ Successfully parsed {len(models)} models")
                return models
                
        except Exception as e:
            logger.error(f"❌ Failed to get available models from OpenRouter: {e}")
            fallback_models = self._get_fallback_models()
            # Cache fallback models too (shorter duration)
            self._available_models_cache = fallback_models
            self._models_cache_timestamp = current_time
            logger.info(f"💾 Cached {len(fallback_models)} fallback models")
            return fallback_models

    async def refresh_available_models(self) -> List[ModelInfo]:
        """Force refresh the cached available models from OpenRouter API"""
        logger.info("🔄 Forcing refresh of available models cache...")
        self._available_models_cache = None
        self._models_cache_timestamp = None
        return await self.get_available_models()
    
    async def get_model_context_window(self, model_name: str) -> int:
        """
        Get context window size for a specific model
        
        **ROOSEVELT'S CENTRALIZED MODEL INTELLIGENCE**: Single source of truth for all agents!
        
        This method:
        - Queries OpenRouter API for authoritative model specs
        - Uses cached model data (refreshed hourly)
        - Falls back to conservative estimates if model not found
        - Can be used by ALL agents for consistent behavior
        
        Args:
            model_name: Model ID (e.g., "anthropic/claude-sonnet-4.5")
        
        Returns:
            Context window size in tokens
        """
        try:
            # Get fresh or cached model list
            available_models = await self.get_available_models()
            
            # Find exact match first
            for model in available_models:
                if model.id == model_name and model.context_length > 0:
                    logger.info(f"🎯 MODEL CONTEXT: {model_name} → {model.context_length:,} tokens (OpenRouter)")
                    return model.context_length
            
            # Try partial match (e.g., "claude-sonnet-4.5" matches "anthropic/claude-sonnet-4.5")
            model_name_lower = model_name.lower()
            for model in available_models:
                if model.id.lower().endswith(model_name_lower) and model.context_length > 0:
                    logger.info(f"🎯 MODEL CONTEXT: {model_name} → {model.context_length:,} tokens (matched {model.id})")
                    return model.context_length
                # Also check if model_name contains the model id
                if model.id.lower() in model_name_lower and model.context_length > 0:
                    logger.info(f"🎯 MODEL CONTEXT: {model_name} → {model.context_length:,} tokens (contains {model.id})")
                    return model.context_length
            
            logger.warning(f"⚠️ MODEL CONTEXT: {model_name} not found in OpenRouter, using fallback estimate")
            
        except Exception as e:
            logger.warning(f"⚠️ MODEL CONTEXT: Failed to query models: {e}, using fallback")
        
        # **FALLBACK**: Conservative estimates for common models
        fallback_contexts = {
            "claude": 200000,  # Most Claude models
            "sonnet": 200000,
            "haiku": 200000,
            "opus": 200000,
            "gpt-4": 8000,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-3.5": 16000,
            "gemini-1.5": 1000000,
            "gemini-pro": 32000,
            "llama-3": 8000,
        }
        
        model_name_lower = model_name.lower()
        for keyword, context_size in fallback_contexts.items():
            if keyword in model_name_lower:
                logger.info(f"🎯 MODEL CONTEXT: {model_name} → {context_size:,} tokens (fallback: matched '{keyword}')")
                return context_size
        
        # Ultimate fallback: very conservative
        logger.warning(f"⚠️ MODEL CONTEXT: {model_name} → 8,000 tokens (conservative fallback)")
        return 8000

    def _get_fallback_models(self) -> List[ModelInfo]:
        """Fallback models if OpenRouter API fails"""
        logger.info("🔄 Using fallback models")
        return [
            ModelInfo(
                id="anthropic/claude-3-sonnet",
                name="Claude 3 Sonnet",
                provider="Anthropic",
                context_length=200000,
                input_cost=0.003,
                output_cost=0.015,
                description="Balanced model for complex reasoning tasks"
            ),
            ModelInfo(
                id="openai/gpt-4-turbo-preview",
                name="GPT-4 Turbo",
                provider="OpenAI",
                context_length=128000,
                input_cost=0.01,
                output_cost=0.03,
                description="Latest GPT-4 model with improved capabilities"
            ),
            ModelInfo(
                id="anthropic/claude-3-haiku",
                name="Claude 3 Haiku",
                provider="Anthropic",
                context_length=200000,
                input_cost=0.00025,
                output_cost=0.00125,
                description="Fast and efficient model for simple tasks"
            ),
            ModelInfo(
                id="meta-llama/llama-3-70b-instruct",
                name="Llama 3 70B Instruct",
                provider="Meta",
                context_length=8192,
                input_cost=0.0009,
                output_cost=0.0009,
                description="Open-source model with strong performance"
            )
        ]
    
    async def update_model(self, model_name: str):
        """Update the selected model (called when user selects from Chat UI dropdown)"""
        try:
            # Import settings service to avoid circular imports
            from services.settings_service import settings_service
            
            # Verify the model is still enabled
            enabled_models = await settings_service.get_enabled_models()
            
            if model_name not in enabled_models:
                logger.warning(f"🔄 Cannot select model {model_name} - not in enabled models: {enabled_models}")
                return
            
            # Update current model
            self.current_model = model_name
            self.models_enabled = True
            
            # Save user's selection as preference
            await settings_service.set_llm_model(model_name)
            
            logger.info(f"🔄 Model updated to: {model_name} (saved as preference)")
            
        except Exception as e:
            logger.error(f"❌ Failed to update model: {e}")
            # Still update current model even if saving preference fails
            self.current_model = model_name
            self.models_enabled = True
            logger.info(f"🔄 Model updated to: {model_name} (preference save failed)")
    
    async def _auto_select_model(self):
        """Auto-select model if only one is enabled"""
        try:
            # Import settings service to avoid circular imports
            from services.settings_service import settings_service
            
            # Ensure settings service is initialized in worker context
            if not hasattr(settings_service, '_initialized') or not settings_service._initialized:
                logger.info("🔧 Initializing settings service in worker context...")
                await settings_service.initialize()
            
            # Get enabled models
            enabled_models = await settings_service.get_enabled_models()
            logger.info(f"🔍 Found {len(enabled_models)} enabled models: {enabled_models}")
            
            if len(enabled_models) == 1:
                # Only one model enabled, auto-select it
                selected_model = enabled_models[0]
                self.current_model = selected_model
                self.models_enabled = True
                logger.info(f"🎯 Auto-selected model (only one enabled): {selected_model}")
                
                # Also update the persistent setting
                await settings_service.set_llm_model(selected_model)
                
            elif len(enabled_models) > 1:
                # Multiple models enabled, check if we have a saved preference
                saved_model = await settings_service.get_llm_model()
                if saved_model and saved_model in enabled_models:
                    self.current_model = saved_model
                    self.models_enabled = True
                    logger.info(f"🔄 Using saved model preference: {saved_model}")
                else:
                    # No saved preference or saved model not in enabled list, use first enabled
                    self.current_model = enabled_models[0]
                    self.models_enabled = True
                    logger.info(f"🎯 Using first enabled model: {enabled_models[0]}")
                    await settings_service.set_llm_model(enabled_models[0])
            
            elif len(enabled_models) == 0:
                # No models enabled - chat should not work
                logger.error(f"❌ No LLM models are enabled! Chat functionality will be disabled.")
                self.current_model = None
                self.models_enabled = False
            
        except Exception as e:
            logger.error(f"❌ Failed to auto-select model: {e}")
            # Fall back to no model available
            self.current_model = None
            self.models_enabled = False
    
    async def _refresh_model_status(self):
        """Refresh model status by checking current enabled models from database"""
        try:
            # Import settings service to avoid circular imports
            from services.settings_service import settings_service
            
            # Get current enabled models from database
            enabled_models = await settings_service.get_enabled_models()
            
            if len(enabled_models) == 0:
                # No models enabled
                logger.warning("🔄 No LLM models are currently enabled")
                self.current_model = None
                self.models_enabled = False
                return
            
            # Check if current model (selected from Chat UI dropdown) is still valid
            if self.current_model and self.current_model in enabled_models:
                # User's selected model is still enabled, keep it
                logger.debug(f"🔄 Keeping user-selected model: {self.current_model}")
                self.models_enabled = True
                return
            
            # Current model is not set or not in enabled list, need to pick a new one
            if len(enabled_models) == 1:
                # Only one model enabled, use it
                self.current_model = enabled_models[0]
                logger.info(f"🔄 Auto-selected single enabled model: {self.current_model}")
            else:
                # Multiple models enabled, check for saved preference from last manual selection
                saved_model = await settings_service.get_llm_model()
                if saved_model and saved_model in enabled_models:
                    self.current_model = saved_model
                    logger.info(f"🔄 Restored last selected model: {self.current_model}")
                else:
                    # No valid saved preference, use first enabled model
                    self.current_model = enabled_models[0]
                    logger.info(f"🔄 Defaulted to first enabled model: {self.current_model}")
                    # Save this selection as the new preference
                    await settings_service.set_llm_model(self.current_model)
            
            # Update enabled status
            self.models_enabled = True
            
        except Exception as e:
            logger.error(f"❌ Failed to refresh model status: {e}")
            # Fall back to disabled state
            self.current_model = None
            self.models_enabled = False

    async def ensure_model_selected(self):
        """Ensure a model is selected, called on startup to sync with frontend"""
        try:
            # Import settings service to avoid circular imports
            from services.settings_service import settings_service
            
            # Get current enabled models from database
            enabled_models = await settings_service.get_enabled_models()
            
            if len(enabled_models) == 0:
                logger.warning("🔄 No LLM models are currently enabled")
                return False
            
            # Check if we have a current model that's still valid
            if self.current_model and self.current_model in enabled_models:
                logger.info(f"✅ Model already selected and valid: {self.current_model}")
                return True
            
            # Try to get the saved model from settings
            saved_model = await settings_service.get_llm_model()
            if saved_model and saved_model in enabled_models:
                self.current_model = saved_model
                logger.info(f"🔄 Restored saved model on startup: {self.current_model}")
                return True
            
            # Fallback to first enabled model
            self.current_model = enabled_models[0]
            logger.info(f"🔄 Selected first enabled model on startup: {self.current_model}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to ensure model selection on startup: {e}")
            return False
    
    async def summarize_document(self, document_id: str, session_id: str) -> QueryResponse:
        """Generate a comprehensive summary of an entire document"""
        start_time = time.time()
        
        try:
            logger.info(f"📄 Starting full document summarization for: {document_id}")
            
            # Get all chunks for the document
            all_chunks = await self.embedding_manager.get_all_document_chunks(document_id)
            
            if not all_chunks:
                return QueryResponse(
                    answer="I couldn't find any content for this document. It may not have been processed yet or may have been deleted.",
                    citations=[],
                    session_id=session_id,
                    query_time=time.time() - start_time,
                    retrieval_count=0
                )
            
            logger.info(f"📚 Retrieved {len(all_chunks)} chunks for summarization")
            
            # Combine all chunks in order
            full_content = "\n\n".join([chunk['content'] for chunk in all_chunks])
            
            # Create citations for all chunks
            citations = []
            for i, chunk in enumerate(all_chunks):
                # Create extended snippet for better context
                extended_snippet = self._create_extended_snippet(chunk['content'], f"document summary {document_id}")
                citation = Citation(
                    document_id=chunk['document_id'],
                    document_title=f"Document {chunk['document_id']}",
                    chunk_id=chunk['chunk_id'],
                    relevance_score=1.0,  # All chunks are relevant for full summary
                    snippet=extended_snippet
                )
                citations.append(citation)
            
            # Generate comprehensive summary
            summary = await self._generate_document_summary(full_content, document_id)
            
            query_time = time.time() - start_time
            
            response = QueryResponse(
                answer=summary,
                citations=citations,
                session_id=session_id,
                query_time=query_time,
                retrieval_count=len(all_chunks)
            )
            
            # Store in conversation history
            await self._store_query_history(session_id, f"Summarize document {document_id}", response)
            
            logger.info(f"✅ Document summarization completed in {query_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"❌ Document summarization failed: {e}")
            return QueryResponse(
                answer=f"I apologize, but I encountered an error while summarizing the document: {str(e)}. Please try again or contact support if the issue persists.",
                citations=[],
                session_id=session_id,
                query_time=time.time() - start_time,
                retrieval_count=0
            )
    
    async def check_redis_health(self) -> bool:
        """Check Redis health"""
        try:
            await self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"❌ Redis health check failed: {e}")
            return False
    
    async def _generate_document_summary(self, full_content: str, document_id: str) -> str:
        """Generate a comprehensive summary of the entire document"""
        try:
            # Build specialized prompt for document summarization
            system_prompt = """You are Plato, an AI assistant specialized in document analysis and summarization. Your task is to create comprehensive, well-structured summaries of entire documents.

Your summary should:
1. Capture the main themes and key points
2. Identify the document's purpose and structure
3. Highlight important findings, conclusions, or arguments
4. Note any significant details or data points
5. Maintain the document's tone and perspective
6. Be thorough but well-organized

Structure your summary with clear sections and use markdown formatting for readability."""

            # Truncate content if it's too long for the model
            max_content_length = 50000  # Adjust based on model limits
            if len(full_content) > max_content_length:
                logger.info(f"📄 Truncating document content from {len(full_content)} to {max_content_length} characters")
                full_content = full_content[:max_content_length] + "\n\n[Content truncated due to length...]"

            user_prompt = f"""Please provide a comprehensive summary of the following document:

Document Content:
{full_content}

Please create a detailed summary that captures all the important information, themes, and insights from this document. Structure your response clearly and make it as comprehensive as possible while remaining well-organized."""

            # Prepare messages for the API call
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call the LLM with longer timeout for comprehensive summaries
            logger.info(f"🤖 Generating document summary with model: {self.current_model}")
            logger.info(f"🤖 Content length: {len(full_content)} characters")
            
            try:
                # Reasoning automatically added by OpenRouterClient wrapper
                response = await asyncio.wait_for(
                    self.openai_client.chat.completions.create(
                        model=self.current_model,
                        messages=messages,
                        max_tokens=8000,  # Increased for comprehensive summaries
                        temperature=0.3  # Lower temperature for more focused summaries
                    ),
                    timeout=900.0  # Increased to 15 minutes for large document processing
                )
                
                logger.info(f"🤖 Document summary generated successfully")
                
                if response.choices and len(response.choices) > 0:
                    summary = response.choices[0].message.content
                    if summary:
                        logger.info(f"🤖 Generated summary with {len(summary)} characters")
                        return summary
                    else:
                        logger.error(f"❌ LLM returned empty summary")
                        return "I apologize, but I was unable to generate a summary for this document. Please try again."
                else:
                    logger.error(f"❌ LLM returned no choices for summary")
                    return "I apologize, but I was unable to generate a summary for this document. Please try again."
                    
            except asyncio.TimeoutError:
                logger.error(f"❌ Document summary timed out after 2 minutes")
                return "I apologize, but the document summary took too long to generate. The document may be too large or complex. Please try again or contact support."
            except Exception as llm_error:
                logger.error(f"❌ Document summary API call failed: {llm_error}")
                return f"I apologize, but there was an error generating the summary: {str(llm_error)}. Please try again."
            
        except Exception as e:
            logger.error(f"❌ Document summary generation failed: {e}")
            return f"I apologize, but I encountered an error while generating the document summary: {str(e)}. Please try again."

    async def _generate_response(self, query: str, context: str, conversation_history: List[Dict]) -> str:
        """Generate response using LLM with retrieved context"""
        try:
            # Build the system prompt with current date context using user's timezone
            from utils.system_prompt_utils import get_current_datetime_context_for_user
            datetime_context = await get_current_datetime_context_for_user(self.current_user_id)
            
            system_prompt = f"""You are Plato, a knowledgeable AI assistant that helps people understand their documents. You provide clear, helpful answers in a professional yet approachable way.

{datetime_context}

Keep your responses:
- Clear and informative, but not overly formal or structured
- Thorough but concise - include important details without being verbose
- Professional yet friendly in tone

When referencing information from documents, use natural phrases like "Based on your documents" or "I can see from the files that..." If you can't find relevant information, explain this clearly and suggest what might help."""

            # Build the user prompt with context
            if context.strip():
                user_prompt = f"""Based on the following context from the user's documents, please answer this question: {query}

Context:
{context}

Please provide a comprehensive answer based on the available information. If the context doesn't contain sufficient information to fully answer the question, please indicate what information is missing."""
            else:
                user_prompt = f"""I don't have any relevant documents in the knowledge base to answer this question: {query}

Please let me know that no relevant documents were found and suggest that the user upload relevant documents to get better answers."""

            # Prepare messages for the API call
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add conversation history (last few exchanges)
            for hist_item in conversation_history[-3:]:  # Last 3 exchanges for context
                messages.append({"role": "user", "content": hist_item["query"]})
                messages.append({"role": "assistant", "content": hist_item["answer"]})
            
            # Add current query
            messages.append({"role": "user", "content": user_prompt})
            
            # Call the LLM
            logger.info(f"🤖 Calling LLM with model: {self.current_model}")
            logger.info(f"🤖 Message count: {len(messages)}")
            logger.info(f"🤖 Context length: {len(context)} characters")
            
            try:
                # Reasoning automatically added by OpenRouterClient wrapper
                # Add generous timeout for comprehensive processing
                logger.info(f"🤖 Starting LLM API call...")
                response = await asyncio.wait_for(
                    self.openai_client.chat.completions.create(
                        model=self.current_model,
                        messages=messages,
                        max_tokens=4000,  # Reduced to prevent timeout issues
                        temperature=0.7
                    ),
                    timeout=900.0  # Increased to 15 minutes for large documents
                )
                
                logger.info(f"🤖 LLM response received successfully")
                
                if response.choices and len(response.choices) > 0:
                    answer = response.choices[0].message.content
                    if answer:
                        logger.info(f"🤖 Generated response with {len(answer)} characters")
                        return answer
                    else:
                        logger.error(f"❌ LLM returned empty content")
                        return "I apologize, but the AI model returned an empty response. Please try again."
                else:
                    logger.error(f"❌ LLM returned no choices")
                    return "I apologize, but the AI model didn't provide a response. Please try again."
                    
            except asyncio.TimeoutError:
                logger.error(f"❌ LLM call timed out after 60 seconds")
                return "I apologize, but the response took too long to generate. Please try a simpler question or try again later."
            except Exception as llm_error:
                logger.error(f"❌ LLM API call failed: {llm_error}")
                import traceback
                logger.error(f"❌ LLM error traceback: {traceback.format_exc()}")
                return f"I apologize, but there was an error with the AI model: {str(llm_error)}. Please try again."
            
        except Exception as e:
            logger.error(f"❌ LLM response generation failed: {e}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}. Please try again."
    
    async def _get_recent_conversation(self, session_id: str) -> List[Dict]:
        """Get recent conversation history for context"""
        try:
            # Get from Redis
            history_key = f"conversation:{session_id}"
            history_data = await self.redis_client.get(history_key)
            
            if history_data:
                history = json.loads(history_data)
                return history[-settings.CONVERSATION_MEMORY_SIZE:]  # Last N exchanges
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Failed to get conversation history: {e}")
            return []
    
    async def _store_query_history(self, session_id: str, query: str, response: QueryResponse, conversation_id: str = None):
        """Store query in conversation history (both Redis and persistent)"""
        try:
            # Store in Redis for immediate access (existing functionality)
            history_key = f"conversation:{session_id}"
            
            # Get existing history
            existing_data = await self.redis_client.get(history_key)
            if existing_data:
                history = json.loads(existing_data)
            else:
                history = []
            
            # Add new exchange
            history_item = {
                "timestamp": datetime.utcnow().isoformat(),
                "query": query,
                "answer": response.answer,
                "citations": [
                    {
                        "document_id": c.document_id,
                        "chunk_id": c.chunk_id,
                        "relevance_score": c.relevance_score
                    }
                    for c in response.citations
                ]
            }
            
            history.append(history_item)
            
            # Keep only recent history
            if len(history) > settings.CONVERSATION_MEMORY_SIZE * 2:
                history = history[-settings.CONVERSATION_MEMORY_SIZE * 2:]
            
            # Store back to Redis with expiration (24 hours)
            await self.redis_client.setex(
                history_key,
                86400,  # 24 hours
                json.dumps(history)
            )
            
            # Store in persistent conversation history if conversation_id provided
            if conversation_id and self.conversation_service:
                try:
                    # Create message request for user query
                    user_message_request = CreateMessageRequest(
                        content=query,
                        message_type=MessageType.USER
                    )
                    
                    # Store user message
                    user_message = await self.conversation_service.add_message(
                        conversation_id=conversation_id,
                        user_id=self.current_user_id,
                        role="user",
                        content=query,
                        metadata=user_message_request.metadata
                    )
                    
                    # Store assistant response
                    assistant_message_request = CreateMessageRequest(
                        content=response.answer,
                        message_type=MessageType.ASSISTANT,
                        parent_message_id=user_message["message_id"],
                        metadata={
                            "query_time": response.query_time,
                            "retrieval_count": response.retrieval_count,
                            "citations": [
                                {
                                    "document_id": c.document_id,
                                    "chunk_id": c.chunk_id,
                                    "relevance_score": c.relevance_score
                                }
                                for c in response.citations
                            ]
                        }
                    )
                    
                    assistant_message = await self.conversation_service.add_message(
                        conversation_id=conversation_id,
                        user_id=self.current_user_id,
                        role="assistant",
                        content=response.answer,
                        metadata=assistant_message_request.metadata
                    )
                    
                    logger.debug(f"💾 Stored persistent conversation history for conversation {conversation_id}")
                    
                except Exception as persistent_error:
                    logger.warning(f"⚠️ Failed to store persistent conversation history: {persistent_error}")
                    # Continue execution - Redis storage still succeeded
            
            logger.debug(f"💾 Stored conversation history for session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to store query history: {e}")
    
    def _is_procedural_query(self, query: str) -> bool:
        """Detect if query is asking for procedural/instructional information"""
        query_lower = query.lower()
        
        # Keywords that indicate procedural queries
        procedural_keywords = [
            'how to', 'how do i', 'steps to', 'procedure', 'configure', 'setup', 'install',
            'enable', 'disable', 'activate', 'deactivate', 'enter', 'exit', 'mode',
            'instructions', 'guide', 'tutorial', 'process', 'method', 'way to',
            'config', 'configuration', 'setting', 'settings', 'option', 'options',
            'command', 'commands', 'sequence', 'order', 'first', 'then', 'next',
            'before', 'after', 'prerequisite', 'requirement', 'step by step'
        ]
        
        # Check for procedural patterns
        for keyword in procedural_keywords:
            if keyword in query_lower:
                return True
        
        # Check for question patterns that often indicate procedures
        procedural_patterns = [
            'what are the steps',
            'what is the process',
            'what do i need to do',
            'how can i',
            'what should i do',
            'where do i',
            'when do i'
        ]
        
        for pattern in procedural_patterns:
            if pattern in query_lower:
                return True
        
        return False
    
    async def _standard_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """Enhanced multi-stage retrieval with intelligent ranking"""
        try:
            logger.info(f"🔍 Starting enhanced multi-stage retrieval for: {query[:100]}...")
            
            # Step 1: Detect query type for specialized handling
            query_type = self._detect_query_type(query)
            logger.info(f"🔍 Query type detected: {query_type}")
            
            # Step 2: Multi-stage retrieval based on query type
            if query_type == 'metadata':
                return await self._metadata_focused_retrieval(query)
            elif query_type == 'entity':
                return await self._entity_focused_retrieval(query)
            elif query_type == 'temporal':
                return await self._temporal_focused_retrieval(query)
            else:
                return await self._semantic_focused_retrieval(query)
                
        except Exception as e:
            logger.error(f"❌ Enhanced retrieval failed: {e}")
            # Fallback to basic retrieval
            return await self.embedding_manager.search_similar(
                query_text=query,
                limit=settings.MAX_RETRIEVAL_RESULTS,
                score_threshold=0.3,
            )
    
    async def _enhanced_procedural_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """Enhanced retrieval strategy with optimized LLM-guided second pass"""
        try:
            logger.info("🔧 Using enhanced procedural retrieval with LLM guidance")
            
            # Step 1: Primary search with original query
            primary_chunks = await self.embedding_manager.search_similar(
                query_text=query,
                limit=500,  # Increased query results limit
                score_threshold=0.25,
            )
            
            logger.info(f"🔧 Primary search found {len(primary_chunks)} chunks")
            
            # Step 2: Generate related procedural terms (reduced set)
            procedural_terms = self._extract_procedural_terms(query)[:5]  # Limit to 5 terms
            
            # Step 3: Parallel search for procedural terms
            all_chunks = list(primary_chunks)
            seen_chunk_ids = {chunk['chunk_id'] for chunk in primary_chunks}
            
            # Execute procedural term searches in parallel for speed
            term_search_tasks = []
            for term in procedural_terms:
                task = self.embedding_manager.search_similar(
                    query_text=term,
                    limit=50,  # Increased for better coverage
                    score_threshold=0.2,
                )
                term_search_tasks.append(task)
            
            # Wait for all term searches to complete
            if term_search_tasks:
                term_results = await asyncio.gather(*term_search_tasks, return_exceptions=True)
                
                for result in term_results:
                    if isinstance(result, Exception):
                        logger.warning(f"🔧 Term search failed: {result}")
                        continue
                    
                    for chunk in result:
                        if chunk['chunk_id'] not in seen_chunk_ids:
                            all_chunks.append(chunk)
                            seen_chunk_ids.add(chunk['chunk_id'])
            
            logger.info(f"🔧 After procedural terms: {len(all_chunks)} chunks")
            
            # Step 4: Conditional LLM-guided retrieval (only if we have good initial results)
            if len(all_chunks) >= 8:  # Only do LLM guidance if we have sufficient initial results
                try:
                    # Run LLM guidance with timeout
                    llm_guided_chunks = await asyncio.wait_for(
                        self._llm_guided_retrieval(query, all_chunks[:8]),  # Analyze only top 8 chunks
                        timeout=15.0  # 15 second timeout for LLM guidance
                    )
                    
                    # Merge LLM-guided results
                    for chunk in llm_guided_chunks:
                        if chunk['chunk_id'] not in seen_chunk_ids:
                            all_chunks.append(chunk)
                            seen_chunk_ids.add(chunk['chunk_id'])
                    
                    logger.info(f"🔧 Added {len(llm_guided_chunks)} LLM-guided chunks")
                    
                except asyncio.TimeoutError:
                    logger.warning("🔧 LLM-guided retrieval timed out, proceeding without it")
                except Exception as e:
                    logger.warning(f"🔧 LLM-guided retrieval failed: {e}")
            else:
                logger.info("🔧 Skipping LLM guidance due to insufficient initial results")
            
            # Step 5: Sort by relevance and limit results
            all_chunks.sort(key=lambda x: x['score'], reverse=True)
            final_chunks = all_chunks[:20]  # Reduced limit for faster processing
            
            logger.info(f"🔧 Final enhanced retrieval: {len(final_chunks)} chunks")
            return final_chunks
            
        except Exception as e:
            logger.error(f"❌ Enhanced procedural retrieval failed: {e}")
            # Fallback to standard retrieval
            return await self._standard_retrieval(query)
    
    def _extract_procedural_terms(self, query: str) -> List[str]:
        """Extract key terms for additional procedural searches"""
        query_lower = query.lower()
        
        # Common procedural contexts to search for
        base_terms = []
        
        # Extract main action words
        action_words = ['configure', 'setup', 'install', 'enable', 'disable', 'enter', 'exit', 'activate']
        for word in action_words:
            if word in query_lower:
                base_terms.extend([
                    f"{word} mode",
                    f"{word} configuration",
                    f"how to {word}",
                    f"{word} steps",
                    f"{word} procedure"
                ])
        
        # Add context-specific terms
        if 'config' in query_lower or 'configuration' in query_lower:
            base_terms.extend([
                'configuration mode',
                'config mode',
                'enter configuration',
                'configuration steps',
                'setup configuration',
                'configuration procedure'
            ])
        
        if 'mode' in query_lower:
            base_terms.extend([
                'enter mode',
                'exit mode',
                'mode configuration',
                'switch mode',
                'mode setup'
            ])
        
        # Add prerequisite terms
        base_terms.extend([
            'prerequisites',
            'requirements',
            'before configuring',
            'preparation steps',
            'initial setup'
        ])
        
        return base_terms[:10]  # Limit to avoid too many searches
    
    async def _expand_with_context_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Expand chunks with neighboring context for better procedural understanding"""
        try:
            logger.info("📚 Expanding chunks with neighboring context")
            
            expanded_chunks = []
            processed_docs = set()
            
            for chunk in chunks:
                document_id = chunk['document_id']
                
                # Only expand once per document to avoid duplication
                if document_id in processed_docs:
                    expanded_chunks.append(chunk)
                    continue
                
                processed_docs.add(document_id)
                
                # Get all chunks for this document
                doc_chunks = await self.embedding_manager.get_all_document_chunks(document_id)
                
                if not doc_chunks:
                    expanded_chunks.append(chunk)
                    continue
                
                # Find the current chunk's position
                current_chunk_index = None
                for i, doc_chunk in enumerate(doc_chunks):
                    if doc_chunk['chunk_id'] == chunk['chunk_id']:
                        current_chunk_index = i
                        break
                
                if current_chunk_index is None:
                    expanded_chunks.append(chunk)
                    continue
                
                # Add neighboring chunks (before and after)
                context_window = 2  # Number of chunks before and after
                start_idx = max(0, current_chunk_index - context_window)
                end_idx = min(len(doc_chunks), current_chunk_index + context_window + 1)
                
                # Add context chunks with adjusted scores
                for i in range(start_idx, end_idx):
                    context_chunk = doc_chunks[i].copy()
                    
                    # Adjust score based on distance from original chunk
                    distance = abs(i - current_chunk_index)
                    if distance == 0:
                        # Original chunk keeps its score
                        context_chunk['score'] = chunk['score']
                    else:
                        # Context chunks get reduced score based on distance
                        context_chunk['score'] = chunk['score'] * (0.8 ** distance)
                    
                    # Mark as context chunk
                    if distance > 0:
                        context_chunk['context_type'] = f"context_{distance}_steps_{'before' if i < current_chunk_index else 'after'}"
                    else:
                        context_chunk['context_type'] = "primary_match"
                    
                    expanded_chunks.append(context_chunk)
            
            # Sort by score and remove duplicates
            unique_chunks = {}
            for chunk in expanded_chunks:
                chunk_id = chunk['chunk_id']
                if chunk_id not in unique_chunks or chunk['score'] > unique_chunks[chunk_id]['score']:
                    unique_chunks[chunk_id] = chunk
            
            final_chunks = list(unique_chunks.values())
            final_chunks.sort(key=lambda x: x['score'], reverse=True)
            
            logger.info(f"📚 Expanded to {len(final_chunks)} unique chunks with context")
            return final_chunks[:25]  # Limit to prevent overwhelming the LLM
            
        except Exception as e:
            logger.error(f"❌ Context expansion failed: {e}")
            return chunks  # Return original chunks if expansion fails
    
    async def _generate_enhanced_response(self, query: str, context: str, conversation_history: List[Dict], is_procedural: bool, chunks: List[Dict[str, Any]] = None) -> str:
        """Generate response with enhanced prompting for procedural queries and metadata awareness"""
        try:
            # Build enhanced system prompt based on query type
            if is_procedural:
                # Use centralized prompt service for procedural guidance
                from services.prompt_service import prompt_service, AgentMode, UserPromptSettings
                
                assembled_prompt = prompt_service.assemble_prompt(
                    agent_mode=AgentMode.CHAT,
                    tools_description="",  # No tools for legacy chat service
                    user_settings=None,  # Default neutral/professional settings
                    additional_context="SPECIALIZED MODE: Procedural guidance and step-by-step instructions"
                )
                
                system_prompt = assembled_prompt.content + """
                
For procedural queries, you must:
1. Provide complete, sequential steps in the correct order
2. Include ALL prerequisite steps and requirements
3. Mention any preparation or setup needed BEFORE the main procedure
4. Clearly indicate when steps must be performed in a specific sequence
5. Highlight critical steps that could cause issues if missed
6. Include any verification or confirmation steps
7. Mention what to do if something goes wrong

When answering procedural questions:
- Start with any prerequisites or preparation steps
- Use numbered lists for sequential procedures
- Be explicit about the order of operations
- Include context about WHY certain steps are necessary
- Reference specific sources for each major step
- If information seems incomplete, explicitly state what might be missing
- Always err on the side of including too much detail rather than too little

CRITICAL: If the provided context doesn't include prerequisite steps or seems to start in the middle of a procedure, explicitly mention this and suggest what additional information might be needed."""
            else:
                # Use centralized prompt service for document analysis
                from services.prompt_service import prompt_service, AgentMode, UserPromptSettings
                
                assembled_prompt = prompt_service.assemble_prompt(
                    agent_mode=AgentMode.CHAT,
                    tools_description="",  # No tools for legacy chat service
                    user_settings=None,  # Default neutral/professional settings
                    additional_context="SPECIALIZED MODE: Document analysis and understanding"
                )
                
                system_prompt = assembled_prompt.content + """
                
You are an AI assistant that helps users understand and analyze documents in their knowledge base. 

Your role is to:
1. Answer questions based on the provided document context
2. Provide accurate, well-reasoned responses
3. Cite specific sources when making claims
4. Acknowledge when information is not available in the provided context
5. Be helpful, clear, and concise
6. Pay careful attention to source credibility and content type

When answering:
- Use the provided context to inform your response
- Reference specific sources when making claims (e.g., "According to Source 1...")
- If the context doesn't contain enough information, say so clearly
- Provide thoughtful analysis and synthesis of the information
- Maintain a professional but approachable tone
- Consider the credibility and type of each source (fiction vs non-fiction, academic vs news, etc.)
- When citing fictional sources, clearly indicate this and explain how it relates to the question
- For factual queries, prioritize non-fiction and academic sources over fictional content
- If sources have conflicting credibility levels, acknowledge this in your response"""

            # Build enhanced user prompt with metadata context
            if context.strip():
                if is_procedural:
                    user_prompt = f"""Based on the following context from technical documentation, please provide complete step-by-step instructions for: {query}

Context (including related sections and neighboring content):
{context}

Please provide a comprehensive answer that includes:
1. Any prerequisites or preparation steps
2. Complete step-by-step instructions in the correct order
3. Important warnings or notes
4. Verification steps
5. What to do if something doesn't work as expected

If the context appears to be missing prerequisite steps or seems incomplete, please explicitly mention what additional information would be helpful."""
                else:
                    # Include metadata context if available
                    metadata_context = ""
                    if chunks:
                        metadata_context = self._build_metadata_context(chunks)
                    
                    user_prompt = f"""Based on the following context from the user's documents, please answer this question: {query}

Context with Source Information:
{metadata_context if metadata_context else context}

Please provide a comprehensive answer based on the available information. Consider the credibility and type of each source when formulating your response. If the context doesn't contain sufficient information to fully answer the question, please indicate what information is missing."""
            else:
                user_prompt = f"""I don't have any relevant documents in the knowledge base to answer this question: {query}

Please let me know that no relevant documents were found and suggest that the user upload relevant documents to get better answers."""

            # Prepare messages for the API call
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add conversation history (last few exchanges)
            for hist_item in conversation_history[-3:]:
                messages.append({"role": "user", "content": hist_item["query"]})
                messages.append({"role": "assistant", "content": hist_item["answer"]})
            
            # Add current query
            messages.append({"role": "user", "content": user_prompt})
            
            # Call the LLM with enhanced settings for procedural queries
            logger.info(f"🤖 Calling LLM with {'procedural' if is_procedural else 'standard'} prompting")
            logger.info(f"🤖 Context length: {len(context)} characters")
            
            try:
                response = await asyncio.wait_for(
                    self.openai_client.chat.completions.create(
                        model=self.current_model,
                        messages=messages,
                        max_tokens=6000 if is_procedural else 4000,  # More tokens for detailed procedures
                        temperature=0.3 if is_procedural else 0.7  # Lower temperature for more precise procedures
                    ),
                    timeout=90.0  # Longer timeout for detailed procedural responses
                )
                
                logger.info(f"🤖 Enhanced response generated successfully")
                
                if response.choices and len(response.choices) > 0:
                    answer = response.choices[0].message.content
                    if answer:
                        logger.info(f"🤖 Generated response with {len(answer)} characters")
                        return answer
                    else:
                        logger.error(f"❌ LLM returned empty content")
                        return "I apologize, but the AI model returned an empty response. Please try again."
                else:
                    logger.error(f"❌ LLM returned no choices")
                    return "I apologize, but the AI model didn't provide a response. Please try again."
                    
            except asyncio.TimeoutError:
                logger.error(f"❌ LLM call timed out after 90 seconds")
                return "I apologize, but the response took too long to generate. Please try a simpler question or try again later."
            except Exception as llm_error:
                logger.error(f"❌ LLM API call failed: {llm_error}")
                return f"I apologize, but there was an error with the AI model: {str(llm_error)}. Please try again."
            
        except Exception as e:
            logger.error(f"❌ Enhanced response generation failed: {e}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}. Please try again."

    async def _llm_guided_retrieval(self, original_query: str, initial_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use LLM to analyze initial results and generate better search queries"""
        try:
            logger.info("🧠 Starting LLM-guided retrieval analysis")
            
            # Prepare context from initial chunks for analysis
            initial_context = "\n\n".join([
                f"Chunk {i+1}: {chunk['content'][:300]}..." if len(chunk['content']) > 300 else f"Chunk {i+1}: {chunk['content']}"
                for i, chunk in enumerate(initial_chunks[:10])  # Analyze top 10 chunks
            ])
            
            # Build analysis prompt
            analysis_prompt = f"""You are an expert at analyzing document retrieval results and identifying missing information for procedural queries.

Original Query: {original_query}

Initial Retrieved Content:
{initial_context}

Based on the original query and the retrieved content, analyze what important information might be missing. For procedural queries, this often includes:

1. Prerequisites or preparation steps
2. Initial setup or authentication requirements  
3. Context or background information needed before the main procedure
4. Dependencies or requirements that must be met first
5. Alternative approaches or troubleshooting steps

Generate 3-5 specific search queries that would help find the missing information. Focus on:
- Prerequisites and preparation steps
- Setup and configuration requirements
- Context that comes BEFORE the main procedure
- Related procedures or dependencies

Format your response as a JSON list of search queries:
["search query 1", "search query 2", "search query 3"]

Only return the JSON array, no other text."""

            # Call LLM for analysis
            try:
                analysis_response = await asyncio.wait_for(
                    self.openai_client.chat.completions.create(
                        model=self.current_model,
                        messages=[
                            {"role": "system", "content": "You are an expert at analyzing document retrieval gaps and generating targeted search queries."},
                            {"role": "user", "content": analysis_prompt}
                        ],
                        max_tokens=500,
                        temperature=0.3  # Lower temperature for more focused analysis
                    ),
                    timeout=30.0  # Shorter timeout for analysis
                )
                
                if not analysis_response.choices or not analysis_response.choices[0].message.content:
                    logger.warning("🧠 LLM analysis returned empty response")
                    return []
                
                analysis_text = analysis_response.choices[0].message.content.strip()
                logger.info(f"🧠 LLM analysis: {analysis_text[:200]}...")
                
                # Parse JSON response
                import json
                try:
                    # Extract JSON from response (in case there's extra text)
                    start_idx = analysis_text.find('[')
                    end_idx = analysis_text.rfind(']') + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        json_text = analysis_text[start_idx:end_idx]
                        suggested_queries = json.loads(json_text)
                    else:
                        # Fallback: try to parse the whole response
                        suggested_queries = json.loads(analysis_text)
                    
                    if not isinstance(suggested_queries, list):
                        logger.warning("🧠 LLM analysis didn't return a list")
                        return []
                    
                    logger.info(f"🧠 Generated {len(suggested_queries)} additional search queries")
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"🧠 Failed to parse LLM analysis JSON: {e}")
                    # Fallback: extract queries from text manually
                    suggested_queries = self._extract_queries_from_text(analysis_text)
                
            except asyncio.TimeoutError:
                logger.warning("🧠 LLM analysis timed out")
                return []
            except Exception as e:
                logger.warning(f"🧠 LLM analysis failed: {e}")
                return []
            
            # Execute additional searches based on LLM suggestions
            additional_chunks = []
            seen_chunk_ids = {chunk['chunk_id'] for chunk in initial_chunks}
            
            for suggested_query in suggested_queries[:5]:  # Limit to 5 additional searches
                if not suggested_query or len(suggested_query.strip()) < 3:
                    continue
                
                logger.info(f"🔍 LLM-guided search: {suggested_query}")
                
                try:
                    guided_chunks = await self.embedding_manager.search_similar(
                        query_text=suggested_query,
                        limit=80,  # Increased guided search results
                        score_threshold=0.15,  # Lower threshold for guided searches
                    )
                    
                    # Add unique chunks with adjusted scoring
                    for chunk in guided_chunks:
                        if chunk['chunk_id'] not in seen_chunk_ids:
                            # Mark as LLM-guided and adjust score
                            chunk['retrieval_method'] = 'llm_guided'
                            chunk['guided_query'] = suggested_query
                            chunk['score'] = chunk['score'] * 0.9  # Slightly lower score for guided results
                            
                            additional_chunks.append(chunk)
                            seen_chunk_ids.add(chunk['chunk_id'])
                    
                except Exception as e:
                    logger.warning(f"🔍 LLM-guided search failed for '{suggested_query}': {e}")
                    continue
            
            logger.info(f"🧠 LLM-guided retrieval found {len(additional_chunks)} additional chunks")
            return additional_chunks
            
        except Exception as e:
            logger.error(f"❌ LLM-guided retrieval failed: {e}")
            return []
    
    def _extract_queries_from_text(self, text: str) -> List[str]:
        """Fallback method to extract search queries from LLM response text"""
        queries = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for quoted strings or numbered items
            if ('"' in line and line.count('"') >= 2) or line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '*')):
                # Extract the query part
                if '"' in line:
                    # Extract text between quotes
                    start = line.find('"')
                    end = line.rfind('"')
                    if start < end:
                        query = line[start+1:end].strip()
                        if len(query) > 5:
                            queries.append(query)
                else:
                    # Remove numbering/bullets and clean up
                    query = line
                    for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '*']:
                        if query.startswith(prefix):
                            query = query[len(prefix):].strip()
                            break
                    if len(query) > 5:
                        queries.append(query)
        
        return queries[:5]  # Limit to 5 queries

    async def _detect_and_handle_collection_analysis(self, query: str, session_id: str) -> QueryResponse:
        """Detect if query is asking for collection analysis and handle it"""
        try:
            # Check if this is a collection analysis query
            analysis_info = self._parse_collection_analysis_query(query)
            if not analysis_info:
                return None  # Not a collection analysis query
            
            logger.info(f"📊 Detected collection analysis query: {analysis_info['type']}")
            
            # Ensure we have the required services
            if not self.collection_analysis_service or not self.document_service:
                logger.warning("📊 Collection analysis services not available")
                return None
            
            # Handle different types of collection analysis
            if analysis_info['type'] == 'filter_based':
                return await self._handle_filter_based_analysis(analysis_info, session_id)
            elif analysis_info['type'] == 'category_based':
                return await self._handle_category_based_analysis(analysis_info, session_id)
            elif analysis_info['type'] == 'temporal_based':
                return await self._handle_temporal_based_analysis(analysis_info, session_id)
            elif analysis_info['type'] == 'all_documents':
                return await self._handle_all_documents_analysis(analysis_info, session_id)
            else:
                logger.warning(f"📊 Unknown collection analysis type: {analysis_info['type']}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Collection analysis detection failed: {e}")
            return None
    
    def _parse_collection_analysis_query(self, query: str) -> Dict[str, Any]:
        """Parse query to detect collection analysis patterns"""
        query_lower = query.lower()
        
        # First, check if this is a specific topic/method/concept request (NOT collection analysis)
        # These patterns indicate asking about a specific thing, not collection-wide analysis
        specific_topic_patterns = [
            'themes of the', 'themes in the', 'summary of the', 'analyze the', 'overview of the',
            'summarize the', 'summary of this', 'analyze this', 'overview of this',
            'about the', 'regarding the', 'concerning the', 'principles of the',
            'method by', 'approach by', 'system by', 'framework by',
            'concepts of the', 'ideas of the', 'theory of the', 'philosophy of the',
            'main themes of the', 'key concepts of the', 'what is the', 'explain the',
            'how does the', 'what are the main themes of', 'what are the key points of',
            'what are main themes of the', 'main themes of', 'key themes of'
        ]
        
        # Strong indicators that this is asking about a specific topic, NOT collection analysis
        for pattern in specific_topic_patterns:
            if pattern in query_lower:
                # Only continue to collection analysis if explicitly mentions collection-wide scope
                collection_scope_indicators = [
                    'all documents', 'all my documents', 'entire collection', 'whole collection',
                    'across all', 'in all documents', 'throughout all', 'collection analysis'
                ]
                if not any(indicator in query_lower for indicator in collection_scope_indicators):
                    logger.info(f"🔍 Detected specific topic query (pattern: '{pattern}'), not collection analysis")
                    return None  # This is a specific topic request, not collection analysis
        
        # Check for specific document requests (NOT collection analysis)
        specific_document_patterns = [
            'document', 'file', 'paper', 'report', 'email', 'memo'
        ]
        
        # If it's asking for a specific document, it's NOT collection analysis
        for pattern in specific_document_patterns:
            if pattern in query_lower:
                # Additional check: if it mentions "all" or "collection", it might still be collection analysis
                if not any(word in query_lower for word in ['all documents', 'all my', 'collection', 'everything', 'entire collection']):
                    return None  # This is a specific document request, not collection analysis
        
        # Collection analysis keywords - now more specific and requiring explicit collection context
        explicit_collection_keywords = [
            'analyze all', 'summarize all', 'overview of all', 'summary of all',
            'all documents', 'all my documents', 'all my files', 'everything', 
            'entire collection', 'whole collection', 'complete analysis', 
            'comprehensive analysis', 'collection analysis'
        ]
        
        # Collection-wide theme/pattern keywords - only when asking about collection, not specific topics
        collection_theme_keywords = [
            'themes across', 'themes in all', 'patterns across', 'trends across',
            'insights from all', 'themes from all', 'patterns in all'
        ]
        
        # Temporal keywords
        temporal_keywords = [
            'from', 'between', 'during', 'in 2024', 'in 2023', 'last year',
            'this year', 'past', 'recent', 'since', 'until', 'before', 'after',
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december'
        ]
        
        # Category keywords
        category_keywords = [
            'emails', 'documents', 'reports', 'papers', 'articles', 'files',
            'pdfs', 'presentations', 'spreadsheets', 'notes', 'memos',
            'contracts', 'invoices', 'receipts', 'finance', 'hr', 'legal',
            'technical', 'marketing', 'sales', 'research'
        ]
        
        # Check if this looks like a collection analysis query
        has_explicit_collection_keyword = any(keyword in query_lower for keyword in explicit_collection_keywords)
        has_collection_theme_keyword = any(keyword in query_lower for keyword in collection_theme_keywords)
        has_temporal_keyword = any(keyword in query_lower for keyword in temporal_keywords)
        has_category_keyword = any(keyword in query_lower for keyword in category_keywords)
        
        # Specific patterns that indicate collection analysis - now more explicit
        collection_patterns = [
            'analyze all my', 'summarize all my', 'overview of all my', 'summary of all my',
            'what are the main themes in all', 'what are the key points in all',
            'what patterns across', 'what trends across', 'give me insights from all',
            'analyze my entire', 'summarize my entire', 'review all my',
            'look at all my', 'examine all my', 'go through all my'
        ]
        
        has_collection_pattern = any(pattern in query_lower for pattern in collection_patterns)
        
        # Much stricter criteria: must have explicit collection indicators
        is_collection_analysis = (
            has_explicit_collection_keyword or 
            has_collection_theme_keyword or 
            has_collection_pattern
        )
        
        # If it doesn't look like collection analysis, return None
        if not is_collection_analysis:
            return None
        
        # Determine the type of collection analysis
        analysis_info = {
            'original_query': query,
            'analysis_type': 'comprehensive'  # Default
        }
        
        # Check for temporal analysis
        if has_temporal_keyword:
            analysis_info['type'] = 'temporal_based'
            analysis_info['analysis_type'] = 'temporal'
            # Extract temporal information
            analysis_info['temporal_terms'] = [kw for kw in temporal_keywords if kw in query_lower]
        
        # Check for category-based analysis
        elif has_category_keyword:
            analysis_info['type'] = 'category_based'
            analysis_info['analysis_type'] = 'thematic'
            # Extract category information
            analysis_info['categories'] = [kw for kw in category_keywords if kw in query_lower]
        
        # Check for filter-based patterns (more specific criteria)
        elif any(term in query_lower for term in ['with tag', 'tagged', 'category', 'type']):
            analysis_info['type'] = 'filter_based'
        
        # Default to all documents analysis
        else:
            analysis_info['type'] = 'all_documents'
        
        return analysis_info
    
    async def _handle_filter_based_analysis(self, analysis_info: Dict[str, Any], session_id: str) -> QueryResponse:
        """Handle analysis based on document filters"""
        try:
            # For now, analyze all documents since we'd need more sophisticated parsing
            # to extract specific filter criteria from natural language
            logger.info("📊 Performing filter-based analysis (defaulting to all documents)")
            
            # Get all documents
            all_docs = await self.document_service.list_documents(0, 1000)
            if not all_docs:
                return QueryResponse(
                    answer="I don't have any documents in your knowledge base to analyze. Please upload some documents first.",
                    citations=[],
                    session_id=session_id,
                    query_time=0,
                    retrieval_count=0
                )
            
            document_ids = [doc.document_id for doc in all_docs]
            
            # Perform collection analysis
            response = await self.collection_analysis_service.analyze_document_collection(
                document_ids=document_ids,
                analysis_type=analysis_info.get('analysis_type', 'comprehensive'),
                session_id=session_id
            )
            
            # Enhance response with context
            response.answer = f"**Collection Analysis Results**\n\n{response.answer}\n\n*This analysis covers {len(document_ids)} documents in your knowledge base.*"
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Filter-based analysis failed: {e}")
            return None
    
    async def _handle_category_based_analysis(self, analysis_info: Dict[str, Any], session_id: str) -> QueryResponse:
        """Handle analysis based on document categories"""
        try:
            categories = analysis_info.get('categories', [])
            logger.info(f"📊 Performing category-based analysis for: {categories}")
            
            # Create filter request based on detected categories
            filter_request = DocumentFilterRequest(
                categories=categories if categories else None,
                limit=1000
            )
            
            # Get filtered documents
            filtered_result = await self.document_service.filter_documents(filter_request)
            
            if not filtered_result.documents:
                category_text = ', '.join(categories) if categories else 'the specified categories'
                return QueryResponse(
                    answer=f"I couldn't find any documents matching {category_text}. Please check your document categories or upload relevant documents.",
                    citations=[],
                    session_id=session_id,
                    query_time=0,
                    retrieval_count=0
                )
            
            document_ids = [doc.document_id for doc in filtered_result.documents]
            
            # Perform collection analysis
            response = await self.collection_analysis_service.analyze_document_collection(
                document_ids=document_ids,
                analysis_type=analysis_info.get('analysis_type', 'thematic'),
                session_id=session_id
            )
            
            # Enhance response with context
            category_text = ', '.join(categories) if categories else 'the specified categories'
            response.answer = f"**Category-Based Analysis: {category_text.title()}**\n\n{response.answer}\n\n*This analysis covers {len(document_ids)} documents in the {category_text} category/categories.*"
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Category-based analysis failed: {e}")
            return None
    
    async def _handle_temporal_based_analysis(self, analysis_info: Dict[str, Any], session_id: str) -> QueryResponse:
        """Handle analysis based on temporal criteria"""
        try:
            temporal_terms = analysis_info.get('temporal_terms', [])
            logger.info(f"📊 Performing temporal-based analysis for: {temporal_terms}")
            
            # Create basic temporal filter (this could be enhanced with better date parsing)
            filter_request = DocumentFilterRequest(
                limit=1000
            )
            
            # For now, get all documents and let the temporal analysis handle the time-based grouping
            filtered_result = await self.document_service.filter_documents(filter_request)
            
            if not filtered_result.documents:
                return QueryResponse(
                    answer="I don't have any documents in your knowledge base to analyze temporally. Please upload some documents first.",
                    citations=[],
                    session_id=session_id,
                    query_time=0,
                    retrieval_count=0
                )
            
            document_ids = [doc.document_id for doc in filtered_result.documents]
            
            # Perform temporal collection analysis
            response = await self.collection_analysis_service.analyze_document_collection(
                document_ids=document_ids,
                analysis_type='temporal',
                session_id=session_id
            )
            
            # Enhance response with context
            temporal_text = ', '.join(temporal_terms) if temporal_terms else 'the specified time period'
            response.answer = f"**Temporal Analysis: {temporal_text.title()}**\n\n{response.answer}\n\n*This analysis covers {len(document_ids)} documents with temporal pattern analysis.*"
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Temporal-based analysis failed: {e}")
            return None
    
    async def _handle_all_documents_analysis(self, analysis_info: Dict[str, Any], session_id: str) -> QueryResponse:
        """Handle analysis of all documents in the knowledge base"""
        try:
            logger.info("📊 Performing comprehensive analysis of all documents")
            
            # Get all documents
            all_docs = await self.document_service.list_documents(0, 1000)
            if not all_docs:
                return QueryResponse(
                    answer="I don't have any documents in your knowledge base to analyze. Please upload some documents first.",
                    citations=[],
                    session_id=session_id,
                    query_time=0,
                    retrieval_count=0
                )
            
            document_ids = [doc.document_id for doc in all_docs]
            
            # Perform comprehensive collection analysis
            response = await self.collection_analysis_service.analyze_document_collection(
                document_ids=document_ids,
                analysis_type=analysis_info.get('analysis_type', 'comprehensive'),
                session_id=session_id
            )
            
            # Enhance response with context
            response.answer = f"**Comprehensive Knowledge Base Analysis**\n\n{response.answer}\n\n*This analysis covers all {len(document_ids)} documents in your knowledge base.*"
            
            return response
            
        except Exception as e:
            logger.error(f"❌ All documents analysis failed: {e}")
            return None

    def _detect_query_type(self, query: str) -> str:
        """Detect the type of query for specialized retrieval"""
        query_lower = query.lower()
        
        # Metadata-focused queries (email addresses, names, specific identifiers)
        metadata_patterns = [
            '@', 'email', 'from:', 'to:', 'cc:', 'sent by', 'received from',
            'author:', 'sender:', 'recipient:', 'contact:', 'phone:', 'address:'
        ]
        
        # Entity-focused queries (people, organizations, products)
        entity_patterns = [
            'who is', 'what is', 'about', 'regarding', 'concerning',
            'person named', 'company called', 'organization', 'department'
        ]
        
        # Temporal queries (dates, time periods)
        temporal_patterns = [
            'when', 'date', 'time', 'during', 'between', 'since', 'until',
            'before', 'after', 'in 2024', 'in 2023', 'last year', 'this year',
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december'
        ]
        
        # Check patterns
        if any(pattern in query_lower for pattern in metadata_patterns):
            return 'metadata'
        elif any(pattern in query_lower for pattern in entity_patterns):
            return 'entity'
        elif any(pattern in query_lower for pattern in temporal_patterns):
            return 'temporal'
        else:
            return 'semantic'
    
    async def _metadata_focused_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """Enhanced retrieval for metadata-based queries (like email addresses)"""
        try:
            logger.info(f"📧 Starting metadata-focused retrieval for: {query[:100]}...")
            
            # Step 1: Large initial retrieval with lower threshold
            primary_results = await self.embedding_manager.search_similar(
                query_text=query,
                limit=500,  # Increased initial set for comprehensive results
                score_threshold=0.1,  # Lower threshold to catch more results
            )
            
            logger.info(f"📧 Primary metadata search: {len(primary_results)} results")
            
            # Step 2: Extract metadata terms from query
            metadata_terms = self._extract_metadata_terms(query)
            logger.info(f"📧 Extracted metadata terms: {metadata_terms}")
            
            # Step 3: Parallel searches for metadata terms
            all_results = list(primary_results)
            seen_chunk_ids = {chunk['chunk_id'] for chunk in primary_results}
            
            # Search for each metadata term
            for term in metadata_terms[:10]:  # Limit to prevent too many searches
                try:
                    term_results = await self.embedding_manager.search_similar(
                        query_text=term,
                        limit=200,  # Increased metadata search results
                        score_threshold=0.05,  # Very low threshold for metadata
                    )
                    
                    for chunk in term_results:
                        if chunk['chunk_id'] not in seen_chunk_ids:
                            chunk['metadata_term'] = term
                            all_results.append(chunk)
                            seen_chunk_ids.add(chunk['chunk_id'])
                            
                except Exception as e:
                    logger.warning(f"📧 Metadata term search failed for '{term}': {e}")
                    continue
            
            logger.info(f"📧 After metadata term searches: {len(all_results)} results")
            
            # Step 4: Re-rank results based on metadata relevance
            ranked_results = self._rank_metadata_results(query, all_results)
            
            # Step 5: Return top results
            final_results = ranked_results[:settings.MAX_ENTITY_RESULTS]  # Use configurable entity limit
            logger.info(f"📧 Final metadata-focused results: {len(final_results)}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Metadata-focused retrieval failed: {e}")
            return await self._fallback_retrieval(query)
    
    async def _entity_focused_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """Enhanced retrieval for entity-based queries"""
        try:
            logger.info(f"👤 Starting entity-focused retrieval for: {query[:100]}...")
            
            # Step 1: Primary semantic search
            primary_results = await self.embedding_manager.search_similar(
                query_text=query,
                limit=500,  # Increased query results
                score_threshold=0.2,
            )
            
            # Step 2: Extract entity terms
            entity_terms = self._extract_entity_terms(query)
            
            # Step 3: Search for entity variations
            all_results = list(primary_results)
            seen_chunk_ids = {chunk['chunk_id'] for chunk in primary_results}
            
            for term in entity_terms[:8]:
                try:
                    entity_results = await self.embedding_manager.search_similar(
                        query_text=term,
                        limit=100,  # Increased term search results
                        score_threshold=0.15,
                    )
                    
                    for chunk in entity_results:
                        if chunk['chunk_id'] not in seen_chunk_ids:
                            chunk['entity_term'] = term
                            all_results.append(chunk)
                            seen_chunk_ids.add(chunk['chunk_id'])
                            
                except Exception as e:
                    logger.warning(f"👤 Entity search failed for '{term}': {e}")
                    continue
            
            # Step 4: Rank by entity relevance
            ranked_results = self._rank_entity_results(query, all_results)
            
            return ranked_results[:settings.MAX_ENTITY_RESULTS]
            
        except Exception as e:
            logger.error(f"❌ Entity-focused retrieval failed: {e}")
            return await self._fallback_retrieval(query)
    
    async def _temporal_focused_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """Enhanced retrieval for temporal queries"""
        try:
            logger.info(f"📅 Starting temporal-focused retrieval for: {query[:100]}...")
            
            # Step 1: Primary search
            primary_results = await self.embedding_manager.search_similar(
                query_text=query,
                limit=100,
                score_threshold=0.2,
            )
            
            # Step 2: Extract temporal terms
            temporal_terms = self._extract_temporal_terms(query)
            
            # Step 3: Search for temporal variations
            all_results = list(primary_results)
            seen_chunk_ids = {chunk['chunk_id'] for chunk in primary_results}
            
            for term in temporal_terms[:6]:
                try:
                    temporal_results = await self.embedding_manager.search_similar(
                        query_text=term,
                        limit=50,
                        score_threshold=0.15,
                    )
                    
                    for chunk in temporal_results:
                        if chunk['chunk_id'] not in seen_chunk_ids:
                            chunk['temporal_term'] = term
                            all_results.append(chunk)
                            seen_chunk_ids.add(chunk['chunk_id'])
                            
                except Exception as e:
                    logger.warning(f"📅 Temporal search failed for '{term}': {e}")
                    continue
            
            # Step 4: Rank by temporal relevance
            ranked_results = self._rank_temporal_results(query, all_results)
            
            return ranked_results[:settings.MAX_ENTITY_RESULTS]
            
        except Exception as e:
            logger.error(f"❌ Temporal-focused retrieval failed: {e}")
            return await self._fallback_retrieval(query)
    
    async def _semantic_focused_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """Enhanced semantic retrieval for general queries"""
        try:
            logger.info(f"🧠 Starting semantic-focused retrieval for: {query[:100]}...")
            
            # Step 1: Primary search with expansion
            primary_results = await self.embedding_manager.search_similar(
                query_text=query,
                limit=500,  # Increased comprehensive search results
                score_threshold=0.25,
            )
            
            # Step 2: Secondary search with reformulated query
            reformulated_query = self._reformulate_query(query)
            if reformulated_query != query:
                secondary_results = await self.embedding_manager.search_similar(
                    query_text=reformulated_query,
                    limit=100,  # Increased reformulated query results
                    score_threshold=0.2,
                )
                
                # Merge results
                seen_chunk_ids = {chunk['chunk_id'] for chunk in primary_results}
                for chunk in secondary_results:
                    if chunk['chunk_id'] not in seen_chunk_ids:
                        chunk['reformulated'] = True
                        primary_results.append(chunk)
            
            # Step 3: Sort by relevance
            primary_results.sort(key=lambda x: x['score'], reverse=True)
            
            return primary_results[:settings.MAX_ENTITY_RESULTS]
            
        except Exception as e:
            logger.error(f"❌ Semantic-focused retrieval failed: {e}")
            return await self._fallback_retrieval(query)
    
    def _extract_metadata_terms(self, query: str) -> List[str]:
        """Extract metadata search terms from query"""
        terms = []
        query_lower = query.lower()
        
        # Extract email addresses
        import re
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, query)
        terms.extend(emails)
        
        # Extract quoted terms (often names or specific identifiers)
        quoted_pattern = r'"([^"]*)"'
        quoted_terms = re.findall(quoted_pattern, query)
        terms.extend(quoted_terms)
        
        # Extract terms after metadata keywords
        metadata_keywords = ['from:', 'to:', 'cc:', 'sent by', 'author:', 'sender:']
        for keyword in metadata_keywords:
            if keyword in query_lower:
                # Extract the term following the keyword
                start_idx = query_lower.find(keyword) + len(keyword)
                remaining = query[start_idx:].strip()
                # Take the next word or phrase
                next_term = remaining.split()[0] if remaining.split() else ''
                if next_term and len(next_term) > 2:
                    terms.append(next_term)
        
        # Add variations of extracted terms
        expanded_terms = []
        for term in terms:
            expanded_terms.append(term)
            # Add partial matches for email addresses
            if '@' in term:
                username = term.split('@')[0]
                domain = term.split('@')[1]
                expanded_terms.extend([username, domain])
        
        return expanded_terms
    
    def _extract_entity_terms(self, query: str) -> List[str]:
        """Extract entity search terms from query"""
        terms = []
        query_lower = query.lower()
        
        # Extract capitalized words (likely names)
        import re
        capitalized_pattern = r'\b[A-Z][a-z]+\b'
        capitalized_words = re.findall(capitalized_pattern, query)
        terms.extend(capitalized_words)
        
        # Extract terms after entity keywords
        entity_keywords = ['who is', 'about', 'regarding', 'person named', 'company called']
        for keyword in entity_keywords:
            if keyword in query_lower:
                start_idx = query_lower.find(keyword) + len(keyword)
                remaining = query[start_idx:].strip()
                # Take the next few words
                next_words = remaining.split()[:3]
                if next_words:
                    terms.append(' '.join(next_words))
        
        return terms
    
    def _extract_temporal_terms(self, query: str) -> List[str]:
        """Extract temporal search terms from query"""
        terms = []
        query_lower = query.lower()
        
        # Extract year patterns
        import re
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, query)
        terms.extend([f"{year[0]}{year[1]}" for year in years])
        
        # Extract month names
        months = ['january', 'february', 'march', 'april', 'may', 'june',
                 'july', 'august', 'september', 'october', 'november', 'december']
        for month in months:
            if month in query_lower:
                terms.append(month)
        
        # Extract date patterns
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        dates = re.findall(date_pattern, query)
        terms.extend(dates)
        
        return terms
    
    def _rank_metadata_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank results based on metadata relevance"""
        query_lower = query.lower()
        
        for result in results:
            content_lower = result['content'].lower()
            metadata = result.get('metadata', {})
            
            # Base score
            relevance_score = result['score']
            
            # Boost for exact metadata matches
            if '@' in query_lower:
                # Email query - boost if content contains email addresses
                import re
                email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                if re.search(email_pattern, content_lower):
                    relevance_score *= 2.0
            
            # Boost for metadata field matches
            for key, value in metadata.items():
                if isinstance(value, str) and any(term in value.lower() for term in query_lower.split()):
                    relevance_score *= 1.5
            
            # Boost for header/subject line matches (common in emails)
            if any(header in content_lower for header in ['from:', 'to:', 'subject:', 'cc:']):
                relevance_score *= 1.3
            
            result['relevance_score'] = relevance_score
        
        # Sort by relevance score
        results.sort(key=lambda x: x.get('relevance_score', x['score']), reverse=True)
        return results
    
    def _rank_entity_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank results based on entity relevance"""
        # Simple ranking - could be enhanced with NER
        for result in results:
            result['relevance_score'] = result['score']
        
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results
    
    def _rank_temporal_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank results based on temporal relevance"""
        # Simple ranking - could be enhanced with date parsing
        for result in results:
            result['relevance_score'] = result['score']
        
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results
    
    def _reformulate_query(self, query: str) -> str:
        """Reformulate query for better semantic matching"""
        # Simple reformulation - could be enhanced with LLM
        query_lower = query.lower()
        
        # Add synonyms for common terms
        reformulations = {
            'how to': 'steps to',
            'configure': 'setup',
            'enable': 'activate',
            'disable': 'deactivate'
        }
        
        reformulated = query
        for original, replacement in reformulations.items():
            if original in query_lower:
                reformulated = reformulated.replace(original, replacement)
        
        return reformulated if reformulated != query else query
    
    async def _fallback_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """Fallback to basic retrieval if enhanced methods fail"""
        return await self.embedding_manager.search_similar(
            query_text=query,
            limit=settings.MAX_RETRIEVAL_RESULTS,
            score_threshold=0.3,
        )

    async def _hybrid_retrieval(self, query: str, entity_names: List[str]) -> List[Dict[str, Any]]:
        """Hybrid retrieval combining vector search and knowledge graph"""
        try:
            logger.info(f"🔍 Starting hybrid retrieval for query with {len(entity_names)} entities")
            
            # Step 1: Get entity-filtered documents if entities found
            entity_doc_ids = []
            if entity_names:
                # Get documents directly mentioning these entities
                direct_docs = await self.kg_service.find_documents_by_entities(entity_names)
                
                # Get documents mentioning related entities (1-hop)
                related_docs = await self.kg_service.find_related_documents_by_entities(entity_names, max_hops=1)
                
                entity_doc_ids = list(set(direct_docs + related_docs))
                logger.info(f"🔍 Found {len(entity_doc_ids)} documents via entity graph")
            
            # Step 2: Perform vector searches
            search_tasks = []
            
            # Primary vector search (full corpus) - use configurable limit
            search_tasks.append(
                self.embedding_manager.search_similar(
                    query_text=query,
                    limit=settings.MAX_RETRIEVAL_RESULTS,  # Use configurable limit (500)
                    score_threshold=0.20,  # Slightly lower threshold for broader coverage
                    expansion_model=self.current_model
                )
            )
            
            # Entity-filtered vector search if we have entity documents
            if entity_doc_ids:
                search_tasks.append(
                    self.embedding_manager.search_similar_in_documents(
                        query_text=query,
                        document_ids=entity_doc_ids,
                        limit=settings.MAX_ENTITY_RESULTS,  # Use configurable entity limit (200)
                        score_threshold=0.2
                    )
                )
            
            # Execute searches in parallel
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Step 3: Combine and rank results
            all_chunks = []
            seen_chunk_ids = set()
            
            for i, results in enumerate(search_results):
                if isinstance(results, Exception):
                    logger.warning(f"🔍 Search {i} failed: {results}")
                    continue
                
                for chunk in results:
                    chunk_id = chunk['chunk_id']
                    if chunk_id not in seen_chunk_ids:
                        # Mark source of retrieval
                        if i == 0:
                            chunk['retrieval_source'] = 'vector_full'
                        else:
                            chunk['retrieval_source'] = 'vector_entity_filtered'
                            # Boost score for entity-filtered results
                            chunk['score'] = chunk['score'] * 1.2
                        
                        all_chunks.append(chunk)
                        seen_chunk_ids.add(chunk_id)
            
            # Step 4: Re-rank based on entity relevance
            if entity_names:
                all_chunks = await self._rerank_by_entity_relevance(all_chunks, entity_names)
            
            # Step 5: Apply deduplication if enabled
            if settings.DEDUPLICATION_ENABLED:
                all_chunks = await deduplication_manager.deduplicate_query_results(all_chunks, query)
                logger.info(f"🔍 After deduplication: {len(all_chunks)} chunks")
            else:
                # Sort by final score and limit without deduplication
                all_chunks.sort(key=lambda x: x['score'], reverse=True)
                all_chunks = all_chunks[:settings.FINAL_RESULT_LIMIT]
            
            logger.info(f"🔍 Hybrid retrieval completed: {len(all_chunks)} chunks")
            return all_chunks
            
        except Exception as e:
            logger.error(f"❌ Hybrid retrieval failed: {e}")
            # Fallback to standard vector search
            return await self.embedding_manager.search_similar(
                query_text=query,
                limit=500,  # Increased simple search results
                score_threshold=0.3,
            )
    
    async def _rerank_by_entity_relevance(self, chunks: List[Dict[str, Any]], entity_names: List[str]) -> List[Dict[str, Any]]:
        """Re-rank chunks based on entity relevance"""
        try:
            # Get entity importance scores
            entity_scores = await self.kg_service.get_entity_importance_scores(entity_names)
            
            for chunk in chunks:
                content_lower = chunk['content'].lower()
                entity_boost = 0.0
                
                # Check for entity mentions in content
                for entity_name in entity_names:
                    if entity_name.lower() in content_lower:
                        # Boost based on entity importance
                        importance = entity_scores.get(entity_name, 1.0)
                        entity_boost += importance * 0.1  # 10% boost per importance point
                
                # Apply entity boost
                chunk['score'] = chunk['score'] * (1.0 + entity_boost)
                chunk['entity_boost'] = entity_boost
            
            return chunks
            
        except Exception as e:
            logger.warning(f"⚠️ Entity re-ranking failed: {e}")
            return chunks
    
    async def _build_entity_context(self, entity_names: List[str]) -> str:
        """Build context about entities for LLM"""
        if not entity_names:
            return ""
        
        try:
            context_parts = []
            
            for entity_name in entity_names:
                # Get entity relationships
                relationships = await self.kg_service.get_entity_relationships(entity_name)
                
                # Get co-occurring entities
                co_occurring = await self.kg_service.find_co_occurring_entities([entity_name], min_co_occurrences=1)
                
                entity_info = [f"Entity: {entity_name}"]
                
                if relationships:
                    rel_summary = ", ".join([f"{rel['target_name']} ({rel['relationship_type']})" for rel in relationships[:3]])
                    entity_info.append(f"Related to: {rel_summary}")
                
                if co_occurring:
                    co_summary = ", ".join([f"{co['name']}" for co in co_occurring[:3]])
                    entity_info.append(f"Often mentioned with: {co_summary}")
                
                context_parts.append(" | ".join(entity_info))
            
            return "; ".join(context_parts)
            
        except Exception as e:
            logger.warning(f"⚠️ Entity context building failed: {e}")
            return f"Entities mentioned: {', '.join(entity_names)}"
    
    async def _generate_entity_aware_response(self, query: str, context: str, conversation_history: List[Dict], entity_names: List[str], chunks: List[Dict[str, Any]] = None) -> str:
        """Generate response with entity awareness and metadata context"""
        try:
            # Build entity-aware system prompt with metadata awareness
            system_prompt = """You are Plato, a knowledgeable AI assistant that helps people understand their documents. 

Keep your responses:
- Direct and to the point - avoid lengthy preambles or introductions
- Clear and informative with important details
- Professional yet conversational
- Mindful of source credibility and content type

Start answering the question immediately. When referencing documents, use brief phrases like "Based on your documents..." or "I found that..." If information is missing, state this clearly and suggest what might help.

IMPORTANT: Pay attention to the source type and credibility information provided. When citing sources:
- Clearly indicate if a source is fictional vs non-fiction
- For factual queries, prioritize non-fiction and academic sources
- When using fictional sources, explain how they relate to the question
- Acknowledge source credibility levels when they vary"""

            # Build entity-aware user prompt with metadata context
            entity_context_note = ""
            if entity_names:
                entity_context_note = f"\n\nNote: This query involves the following entities: {', '.join(entity_names)}. The context includes information about these entities and their relationships."

            if context.strip():
                # Include metadata context if available
                metadata_context = ""
                if chunks:
                    metadata_context = self._build_metadata_context(chunks)
                
                user_prompt = f"""Based on the following context from the user's documents, please answer this question: {query}

Context with Source Information:
{metadata_context if metadata_context else context}{entity_context_note}

Please provide a comprehensive answer based on the available information. Consider the credibility and type of each source when formulating your response. If the context doesn't contain sufficient information to fully answer the question, please indicate what information is missing. Use entity relationships to provide additional insights where relevant."""
            else:
                user_prompt = f"""I don't have any relevant documents in the knowledge base to answer this question: {query}

Please let me know that no relevant documents were found and suggest that the user upload relevant documents to get better answers."""

            # Prepare messages for the API call
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add conversation history (last few exchanges)
            for hist_item in conversation_history[-3:]:
                messages.append({"role": "user", "content": hist_item["query"]})
                messages.append({"role": "assistant", "content": hist_item["answer"]})
            
            # Add current query
            messages.append({"role": "user", "content": user_prompt})
            
            # Call the LLM
            logger.info(f"🤖 Calling LLM with entity-aware prompting")
            logger.info(f"🤖 Context length: {len(context)} characters")
            logger.info(f"🤖 Entities: {entity_names}")
            
            try:
                response = await asyncio.wait_for(
                    self.openai_client.chat.completions.create(
                        model=self.current_model,
                        messages=messages,
                        max_tokens=4000,
                        temperature=0.7
                    ),
                    timeout=900.0  # Increased to 15 minutes for large documents
                )
                
                logger.info(f"🤖 Entity-aware response generated successfully")
                
                if response.choices and len(response.choices) > 0:
                    answer = response.choices[0].message.content
                    if answer:
                        logger.info(f"🤖 Generated response with {len(answer)} characters")
                        return answer
                    else:
                        logger.error(f"❌ LLM returned empty content")
                        return "I apologize, but the AI model returned an empty response. Please try again."
                else:
                    logger.error(f"❌ LLM returned no choices")
                    return "I apologize, but the AI model didn't provide a response. Please try again."
                    
            except asyncio.TimeoutError:
                logger.error(f"❌ LLM call timed out after 60 seconds")
                return "I apologize, but the response took too long to generate. Please try a simpler question or try again later."
            except Exception as llm_error:
                logger.error(f"❌ LLM API call failed: {llm_error}")
                return f"I apologize, but there was an error with the AI model: {str(llm_error)}. Please try again."
            
        except Exception as e:
            logger.error(f"❌ Entity-aware response generation failed: {e}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}. Please try again."

    def _extract_highlight_text(self, content: str, query: str) -> str:
        """Extract the most relevant portion for highlighting"""
        
        # Find best matching sentence/phrase
        sentences = content.split('.')
        query_words = query.lower().split()
        
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for word in query_words if word in sentence_lower)
            
            if score > best_score:
                best_score = score
                best_sentence = sentence.strip()
        
        return best_sentence[:100] + "..." if len(best_sentence) > 100 else best_sentence

    def _create_extended_snippet(self, content: str, query: str) -> str:
        """Create an extended snippet with more context around the relevant content"""
        try:
            # Significantly increased snippet length for much better context
            max_snippet_length = 1000  # Increased from 500 to 1000 characters for richer context
            
            if len(content) <= max_snippet_length:
                return content
            
            # Find the most relevant part of the content
            query_words = [word.lower() for word in query.split() if len(word) > 2]  # Filter short words
            content_lower = content.lower()
            
            # Find the best matching position using multiple strategies
            best_position = 0
            best_score = 0
            
            # Strategy 1: Exact phrase matching
            for phrase in [query.lower()]:
                if phrase in content_lower:
                    phrase_pos = content_lower.find(phrase)
                    best_position = phrase_pos
                    best_score = 100  # High score for exact phrase match
                    break
            
            # Strategy 2: Multiple word matching with proximity scoring
            if best_score == 0:
                window_size = max_snippet_length // 3  # Smaller window for better precision
                
                for i in range(0, len(content) - window_size, 100):  # Step by 100 characters
                    window_text = content_lower[i:i + window_size]
                    
                    # Score based on word matches and proximity
                    word_matches = sum(1 for word in query_words if word in window_text)
                    
                    # Bonus for having multiple query words close together
                    proximity_bonus = 0
                    if len(query_words) > 1:
                        word_positions = []
                        for word in query_words:
                            pos = window_text.find(word)
                            if pos >= 0:
                                word_positions.append(pos)
                        
                        if len(word_positions) > 1:
                            word_span = max(word_positions) - min(word_positions)
                            if word_span < window_size // 2:  # Words are close together
                                proximity_bonus = 2
                    
                    total_score = word_matches + proximity_bonus
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_position = i
            
            # Extract content around the best position with more context
            context_padding = max_snippet_length // 4  # 25% of snippet length for padding
            start_pos = max(0, best_position - context_padding)
            end_pos = min(len(content), start_pos + max_snippet_length)
            
            # Adjust to sentence boundaries when possible for better readability
            if start_pos > 0:
                # Look back for sentence endings within reasonable distance
                sentence_start = start_pos
                for i in range(start_pos, max(0, start_pos - 200), -1):
                    if content[i] in ['.', '!', '?', '\n']:
                        sentence_start = i + 1
                        break
                    elif content[i] in [' ', '\t'] and i > 0 and content[i-1] in ['.', '!', '?']:
                        sentence_start = i + 1
                        break
                
                # Use sentence boundary if it's not too far back
                if sentence_start > start_pos - 100:
                    start_pos = sentence_start
                else:
                    # Fall back to word boundaries
                    while start_pos < len(content) and content[start_pos] not in [' ', '\n', '\t']:
                        start_pos += 1
                    start_pos = min(start_pos + 1, len(content))
            
            if end_pos < len(content):
                # Look forward for sentence endings
                sentence_end = end_pos
                for i in range(end_pos, min(len(content), end_pos + 200)):
                    if content[i] in ['.', '!', '?']:
                        sentence_end = i + 1
                        break
                
                # Use sentence boundary if it's not too far forward
                if sentence_end < end_pos + 100:
                    end_pos = sentence_end
                else:
                    # Fall back to word boundaries
                    while end_pos > start_pos and content[end_pos] not in [' ', '\n', '\t', '.', '!', '?']:
                        end_pos -= 1
            
            snippet = content[start_pos:end_pos].strip()
            
            # Add contextual ellipsis
            if start_pos > 0:
                snippet = "..." + snippet
            if end_pos < len(content):
                snippet = snippet + "..."
            
            return snippet
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to create extended snippet: {e}")
            # Fallback to simple truncation with larger size
            return content[:1000] + "..." if len(content) > 1000 else content

    def set_services(self, collection_analysis_service, document_service):
        """Set service dependencies (called during initialization)"""
        self.collection_analysis_service = collection_analysis_service
        self.document_service = document_service
        logger.info("📊 Collection analysis services configured for chat service")

    async def _get_document_metadata(self, document_id: str) -> Dict[str, Any]:
        """Get document metadata for better citations"""
        try:
            if self.document_service:
                # Try the document repository method instead
                if hasattr(self.document_service, 'document_repository'):
                    doc_info = await self.document_service.document_repository.get_document_by_id(document_id)
                    if doc_info:
                        return {
                            'title': doc_info.get('title', ''),
                            'filename': doc_info.get('filename', ''),
                            'author': doc_info.get('author', ''),
                            'doc_type': doc_info.get('doc_type', '')
                        }
            return {}
        except Exception as e:
            logger.warning(f"⚠️ Failed to get document metadata for {document_id}: {e}")
            return {}

    async def _get_segment_info_for_chunk(self, chunk_id: str) -> Dict[str, Any]:
        """Get PDF segment information for a chunk if it exists"""
        try:
            # Use the document service's repository if available
            if self.document_service and hasattr(self.document_service, 'document_repository'):
                doc_repo = self.document_service.document_repository
            else:
                # Fallback: create a new repository instance
                from repositories.document_repository import DocumentRepository
                doc_repo = DocumentRepository()
                await doc_repo.initialize()
            
            # Query to find segment information for this chunk
            query = """
                SELECT 
                    ps.segment_id,
                    ps.segment_type,
                    pp.page_number,
                    pp.id as page_id
                FROM pdf_segments ps
                JOIN pdf_pages pp ON ps.page_id = pp.id
                WHERE ps.manual_text_chunk_id = $1
                LIMIT 1
            """
            
            results = await doc_repo.execute_query(query, chunk_id)
            
            if results:
                segment = results[0]
                return {
                    'segment_id': segment['segment_id'],
                    'segment_type': segment['segment_type'],
                    'page_number': segment['page_number'],
                    'page_id': segment['page_id'],
                    'bounds': segment['bounds']
                }
            
            return {}
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to get segment info for chunk {chunk_id}: {e}")
            return {}

    def _filter_and_validate_entities(self, raw_entities: List[str], original_query: str) -> List[str]:
        """Filter and validate extracted entities to remove artifacts and irrelevant terms"""
        if not raw_entities:
            return []
        
        # Common question words, articles, prepositions to filter out
        stopwords = {
            'what', 'where', 'when', 'who', 'how', 'why', 'which', 'whose',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'within',
            'happens', 'happen', 'occurred', 'occur', 'book', 'document', 'file',
            'chapter', 'page', 'section', 'story', 'novel', 'text', 'article'
        }
        
        valid_entities = []
        
        for entity in raw_entities:
            # Clean punctuation and possessive forms
            clean_entity = entity.strip('.,?!:;"\'').rstrip("'s")
            
            # Skip if empty or too short after cleaning
            if not clean_entity or len(clean_entity) < 2:
                continue
            
            # Skip if it's a stopword
            if clean_entity.lower() in stopwords:
                logger.debug(f"🔍 Filtered stopword entity: {entity}")
                continue
            
            # Skip single letters (unless they're likely names like 'I' or 'A')
            if len(clean_entity) == 1 and clean_entity.upper() not in ['I', 'A']:
                logger.debug(f"🔍 Filtered single letter entity: {entity}")
                continue
            
            # Skip very common words that shouldn't be entities
            if clean_entity.lower() in ['wife', 'husband', 'man', 'woman', 'person', 'people']:
                logger.debug(f"🔍 Filtered common word entity: {entity}")
                continue
            
            # Validation: entity should have at least one alphabetic character
            if not any(c.isalpha() for c in clean_entity):
                logger.debug(f"🔍 Filtered non-alphabetic entity: {entity}")
                continue
            
            # Add the cleaned entity
            valid_entities.append(clean_entity)
            logger.debug(f"🔍 Accepted entity: {entity} -> {clean_entity}")
        
        # Remove duplicates while preserving order
        seen = set()
        final_entities = []
        for entity in valid_entities:
            entity_lower = entity.lower()
            if entity_lower not in seen:
                seen.add(entity_lower)
                final_entities.append(entity)
        
        # Additional validation: if we filtered out too many entities, 
        # it might indicate the original extraction was poor
        if len(raw_entities) > 3 and len(final_entities) == 0:
            logger.warning(f"🔍 All entities filtered out from: {raw_entities}")
        
        logger.info(f"🔍 Entity filtering: {len(raw_entities)} -> {len(final_entities)}")
        return final_entities

    def _expand_temporal_query(self, query: str) -> str:
        """Expand temporal references in queries to include actual dates"""
        from datetime import datetime, timedelta
        
        current_time = datetime.now()
        query_lower = query.lower()
        expanded_query = query
        
        # Define temporal mappings
        temporal_expansions = {}
        
        # Today/yesterday expansions
        today_str = current_time.strftime("%Y-%m-%d")
        yesterday = current_time - timedelta(days=1)
        yesterday_str = yesterday.strftime("%Y-%m-%d")
        
        temporal_expansions.update({
            "today": f"today {today_str}",
            "today's": f"today's {today_str}",
            "todays": f"today's {today_str}",
            "yesterday": f"yesterday {yesterday_str}",
            "yesterday's": f"yesterday's {yesterday_str}",
            "yesterdays": f"yesterday's {yesterday_str}"
        })
        
        # This week/month/year
        current_year = current_time.year
        current_month = current_time.strftime("%B")
        current_week_start = current_time - timedelta(days=current_time.weekday())
        current_week_str = current_week_start.strftime("%Y-%m-%d")
        
        temporal_expansions.update({
            "this week": f"this week {current_week_str}",
            "this month": f"this month {current_month} {current_year}",
            "this year": f"this year {current_year}",
            "last week": f"last week {(current_week_start - timedelta(days=7)).strftime('%Y-%m-%d')}",
            "last month": f"last month {(current_time.replace(day=1) - timedelta(days=1)).strftime('%B %Y')}",
            "last year": f"last year {current_year - 1}"
        })
        
        # Apply expansions
        for temporal_term, expanded_term in temporal_expansions.items():
            if temporal_term in query_lower:
                # Replace the temporal term with expanded version
                expanded_query = expanded_query.replace(temporal_term, expanded_term)
        
        return expanded_query

    async def _assess_chunk_sufficiency(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """LLM assesses whether retrieved chunks are sufficient to answer the query"""
        try:
            if not chunks:
                logger.info("🧠 No chunks retrieved, marking as INSUFFICIENT")
                return "INSUFFICIENT"
            
            # Special handling for headline queries - they should NOT trigger full document retrieval
            query_lower = query.lower()
            if any(pattern in query_lower for pattern in ['headline', 'headlines', 'top stories', 'news summary']):
                logger.info("🧠 Headline query detected - checking for sufficient headline content")
                
                # Check if we have headline-like content in our chunks
                headline_chunks = 0
                for chunk in chunks:
                    content_lower = chunk['content'].lower()
                    if any(indicator in content_lower for indicator in ['headline', 'breaking', 'news', 'story', 'report']):
                        headline_chunks += 1
                
                if headline_chunks >= 5 or len(chunks) >= 20:
                    logger.info(f"🧠 Found {headline_chunks} headline-like chunks out of {len(chunks)} total - marking SUFFICIENT")
                    return "SUFFICIENT"
                else:
                    logger.info(f"🧠 Only {headline_chunks} headline-like chunks found - will use targeted headline retrieval")
                    return "INSUFFICIENT"
            
            # Prepare chunk context for LLM assessment
            chunk_summary = self._prepare_chunk_summary_for_assessment(query, chunks)
            
            assessment_prompt = f"""You are analyzing whether the retrieved information is sufficient to fully answer a user's query.

User Query: "{query}"

Retrieved Information Summary:
{chunk_summary}

Assess whether this information is SUFFICIENT or INSUFFICIENT to fully answer the user's query.

Consider:
- SUFFICIENT: The chunks contain enough relevant information to provide a complete answer
- INSUFFICIENT: The query requires more comprehensive information, complete summaries, or full document analysis

NOTE: For headline queries, prefer SUFFICIENT if we have good headline content rather than triggering full document analysis.

Respond with only: SUFFICIENT or INSUFFICIENT"""

            try:
                response = await asyncio.wait_for(
                    self.openai_client.chat.completions.create(
                        model=self.current_model,
                        messages=[
                            {"role": "system", "content": "You assess information sufficiency. Respond with exactly: SUFFICIENT or INSUFFICIENT"},
                            {"role": "user", "content": assessment_prompt}
                        ],
                        max_tokens=20,
                        temperature=0.1
                    ),
                    timeout=600.0
                )
                
                if response.choices and response.choices[0].message.content:
                    assessment = response.choices[0].message.content.strip().upper()
                    logger.info(f"🧠 LLM chunk assessment: '{assessment}'")
                    
                    if "INSUFFICIENT" in assessment:
                        return "INSUFFICIENT"
                    elif "SUFFICIENT" in assessment:
                        return "SUFFICIENT"
                    else:
                        logger.warning(f"🧠 LLM assessment unclear: '{assessment}', defaulting to SUFFICIENT")
                        return "SUFFICIENT"  # Default to using chunks when unclear
                else:
                    logger.warning("🧠 LLM chunk assessment returned empty, defaulting to SUFFICIENT")
                    return "SUFFICIENT"  # Default to using chunks when empty
                    
            except asyncio.TimeoutError:
                logger.warning("🧠 LLM chunk assessment timed out")
                return self._fallback_sufficiency_assessment(query, chunks)
            except Exception as e:
                logger.warning(f"🧠 LLM chunk assessment failed: {e}")
                return self._fallback_sufficiency_assessment(query, chunks)
                
        except Exception as e:
            logger.error(f"❌ Chunk sufficiency assessment failed: {e}")
            return self._fallback_sufficiency_assessment(query, chunks)
    
    def _prepare_chunk_summary_for_assessment(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Prepare a concise summary of chunks for LLM assessment"""
        try:
            # Get top chunks (by relevance score)
            top_chunks = sorted(chunks, key=lambda x: x['score'], reverse=True)[:10]
            
            chunk_summaries = []
            for i, chunk in enumerate(top_chunks):
                # Create a brief summary of each chunk
                content_preview = chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
                chunk_summaries.append(f"Chunk {i+1} (score: {chunk['score']:.3f}): {content_preview}")
            
            summary_text = "\n\n".join(chunk_summaries)
            
            # Add metadata about the retrieval
            metadata_summary = f"""
Total chunks retrieved: {len(chunks)}
Score range: {min(chunk['score'] for chunk in chunks):.3f} to {max(chunk['score'] for chunk in chunks):.3f}
Documents involved: {len(set(chunk['document_id'] for chunk in chunks))}

Top chunks:
{summary_text}"""
            
            return metadata_summary
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to prepare chunk summary: {e}")
            return f"Retrieved {len(chunks)} chunks for assessment"
    
    def _fallback_sufficiency_assessment(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Fallback assessment when LLM assessment fails"""
        query_lower = query.lower()
        
        # Patterns that typically require full document analysis
        comprehensive_patterns = [
            'headlines', 'all headlines', 'top headlines', 'today\'s headlines',
            'complete', 'comprehensive', 'entire', 'full', 'overview', 'summary',
            'everything about', 'all content', 'analyze all', 'summarize all'
        ]
        
        # Check if query requires comprehensive analysis
        needs_comprehensive = any(pattern in query_lower for pattern in comprehensive_patterns)
        
        # Check chunk quality
        if not chunks:
            return "INSUFFICIENT"
        
        # Check if we have high-quality chunks
        high_quality_chunks = [c for c in chunks if c['score'] > 0.7]
        diverse_documents = len(set(chunk['document_id'] for chunk in chunks))
        
        if needs_comprehensive:
            logger.info(f"🧠 Fallback: Comprehensive query detected, marking INSUFFICIENT")
            return "INSUFFICIENT"
        elif len(high_quality_chunks) >= 3 and diverse_documents >= 1:
            logger.info(f"🧠 Fallback: Good chunks available ({len(high_quality_chunks)} high-quality), marking SUFFICIENT")
            return "SUFFICIENT"
        else:
            logger.info(f"🧠 Fallback: Poor chunk quality or coverage, marking INSUFFICIENT")
            return "INSUFFICIENT"
    
    async def _intelligent_full_document_retrieval(self, query: str, entity_names: List[str]) -> tuple[List[Dict[str, Any]], bool]:
        """Intelligently retrieve full documents based on query intent"""
        try:
            logger.info("📚 Starting intelligent full document retrieval")
            
            # Step 1: Find relevant documents using hybrid search
            relevant_documents = await self._find_relevant_documents_for_query(query, entity_names)
            
            if not relevant_documents:
                logger.info("📚 No relevant documents found, falling back to chunk retrieval")
                chunks = await self._hybrid_retrieval(query, entity_names)
                return chunks, False
            
            logger.info(f"📚 Found {len(relevant_documents)} relevant documents")
            
            # Step 2: Check if documents are too large for context
            total_chunks = 0
            doc_chunk_counts = {}
            for doc_id in relevant_documents:
                doc_chunks = await self.embedding_manager.get_all_document_chunks(doc_id)
                doc_chunk_counts[doc_id] = len(doc_chunks)
                total_chunks += len(doc_chunks)
            
            logger.info(f"📚 Total chunks across documents: {total_chunks}")
            for doc_id, count in doc_chunk_counts.items():
                logger.info(f"📚 Document {doc_id}: {count} chunks")
            
            # Very conservative limits for stability with large email collections
            max_chunks_for_direct = 30    # Reduced from 50
            max_chunks_for_iterative = 150  # Severely reduced from 500
            max_chunks_per_document = 100   # Much stricter limit per document
            
            if total_chunks > max_chunks_for_iterative:
                logger.warning(f"📚 Documents extremely large ({total_chunks} chunks), falling back to chunk-based retrieval for stability")
                # For extremely large document sets, fall back to chunk-based retrieval
                chunks = await self._hybrid_retrieval(query, entity_names)
                return chunks, False
            elif total_chunks > max_chunks_for_direct:
                logger.info(f"📚 Documents large ({total_chunks} chunks), will use iterative analysis with strict limits")
                # Filter out documents that are too large individually and limit total chunks
                manageable_docs = []
                manageable_chunks = []
                total_added = 0
                
                for doc_id in relevant_documents:
                    doc_chunks = await self.embedding_manager.get_all_document_chunks(doc_id)
                    
                    # Skip documents that are individually too large
                    if len(doc_chunks) > max_chunks_per_document:
                        logger.warning(f"📚 Skipping document {doc_id} ({len(doc_chunks)} chunks) - exceeds per-document limit of {max_chunks_per_document}")
                        continue
                    
                    # Check if adding this document would exceed our total limit
                    if total_added + len(doc_chunks) > max_chunks_for_iterative:
                        logger.warning(f"📚 Stopping at document {doc_id} - would exceed total limit of {max_chunks_for_iterative} chunks")
                        break
                    
                    manageable_docs.append(doc_id)
                    manageable_chunks.extend(doc_chunks)
                    total_added += len(doc_chunks)
                    
                    logger.info(f"📚 Added document {doc_id} ({len(doc_chunks)} chunks) - total now {total_added}")
                
                if not manageable_chunks:
                    logger.warning("📚 No documents small enough for iterative processing, falling back to chunk retrieval")
                    chunks = await self._hybrid_retrieval(query, entity_names)
                    return chunks, False
                
                logger.info(f"📚 Using {len(manageable_docs)} documents with {len(manageable_chunks)} chunks for iterative processing (limit: {max_chunks_for_iterative})")
                return manageable_chunks, True
            
            # Step 3: Retrieve all chunks for direct processing
            all_chunks = []
            for doc_id in relevant_documents:
                doc_chunks = await self.embedding_manager.get_all_document_chunks(doc_id)
                for chunk in doc_chunks:
                    chunk['retrieval_source'] = 'full_document'
                    chunk['score'] = 1.0  # All chunks are relevant for full document analysis
                all_chunks.extend(doc_chunks)
            
            logger.info(f"📚 Retrieved {len(all_chunks)} chunks from {len(relevant_documents)} documents for direct processing")
            return all_chunks, False
            
        except Exception as e:
            logger.error(f"❌ Intelligent full document retrieval failed: {e}")
            # Fallback to hybrid retrieval
            chunks = await self._hybrid_retrieval(query, entity_names)
            return chunks, False
    
    async def _find_relevant_documents_for_query(self, query: str, entity_names: List[str]) -> List[str]:
        """Find documents most relevant to the query for full document analysis"""
        try:
            # Use a combination of approaches to find relevant documents
            document_candidates = set()
            
            # Approach 1: Entity-based document discovery
            if entity_names:
                entity_docs = await self.kg_service.find_documents_by_entities(entity_names)
                document_candidates.update(entity_docs)
            
            # Approach 2: Semantic search to find most relevant documents
            initial_chunks = await self.embedding_manager.search_similar(
                query_text=query,
                limit=500,  # Increased fallback search results
                score_threshold=0.3,
            )
            
            # Get document IDs from top chunks
            for chunk in initial_chunks[:20]:  # Top 20 chunks
                document_candidates.add(chunk['document_id'])
            
            # Approach 3: Query-specific document patterns
            query_lower = query.lower()
            if any(pattern in query_lower for pattern in ['headlines', 'wall street journal', 'wsj', 'news']):
                # For news queries, look for news-related documents
                news_chunks = await self.embedding_manager.search_similar(
                    query_text="news headlines wall street journal",
                    limit=100,  # Increased news search results
                    score_threshold=0.2,
                )
                for chunk in news_chunks:
                    document_candidates.add(chunk['document_id'])
            
            relevant_docs = list(document_candidates)[:5]  # Limit to top 5 documents
            logger.info(f"📚 Found {len(relevant_docs)} candidate documents for full retrieval")
            
            return relevant_docs
            
        except Exception as e:
            logger.error(f"❌ Failed to find relevant documents: {e}")
            return []
    
    async def _iterative_document_analysis(self, query: str, document_chunks: List[Dict[str, Any]], session_id: str, entity_names: List[str]) -> QueryResponse:
        """Perform iterative memory-based analysis for large documents"""
        try:
            start_time = time.time()
            logger.info(f"🔄 Starting iterative analysis of {len(document_chunks)} chunks")
            
            # Additional safety check - if still too many chunks, fall back immediately
            if len(document_chunks) > 400:
                logger.warning(f"🔄 Too many chunks for iterative analysis ({len(document_chunks)}), falling back to standard processing")
                return await self._fallback_large_document_processing(query, document_chunks[:30], session_id, entity_names)
            
            # Group chunks by document for organized processing
            docs_chunks = {}
            for chunk in document_chunks:
                doc_id = chunk['document_id']
                if doc_id not in docs_chunks:
                    docs_chunks[doc_id] = []
                docs_chunks[doc_id].append(chunk)
            
            logger.info(f"🔄 Grouped into {len(docs_chunks)} documents")
            
            # Process each document iteratively
            analysis_memory = []
            citations = []
            
            for i, (doc_id, chunks) in enumerate(docs_chunks.items()):
                try:
                    logger.info(f"🔄 Processing document {i+1}/{len(docs_chunks)}: {doc_id} with {len(chunks)} chunks")
                    
                    # Safety check per document
                    if len(chunks) > 200:
                        logger.warning(f"🔄 Document {doc_id} too large ({len(chunks)} chunks), skipping")
                        continue
                    
                    # Sort chunks by index to maintain order
                    chunks.sort(key=lambda x: x.get('chunk_index', 0))
                    
                    # Process document in batches with timeout
                    doc_analysis = await asyncio.wait_for(
                        self._process_document_iteratively(query, chunks, doc_id),
                        timeout=300.0  # 5 minute timeout per document
                    )
                    
                    if doc_analysis:
                        analysis_memory.append(doc_analysis)
                        logger.info(f"🔄 Successfully processed document {doc_id}")
                    else:
                        logger.warning(f"🔄 No analysis generated for document {doc_id}")
                    
                    # Create citations for this document
                    doc_metadata = await self._get_document_metadata(doc_id)
                    doc_title = doc_metadata.get('title') or doc_metadata.get('filename', f"Document {doc_id}")
                    
                    citation = Citation(
                        document_id=doc_id,
                        document_title=doc_title,
                        chunk_id=f"{doc_id}_analysis",
                        relevance_score=1.0,
                        snippet=doc_analysis[:500] + "..." if doc_analysis and len(doc_analysis) > 500 else doc_analysis or "Analysis in progress..."
                    )
                    citations.append(citation)
                    
                except asyncio.TimeoutError:
                    logger.error(f"🔄 Document {doc_id} processing timed out after 5 minutes")
                    continue
                except Exception as doc_error:
                    logger.error(f"🔄 Failed to process document {doc_id}: {doc_error}")
                    continue
            
            if not analysis_memory:
                logger.warning("🔄 No documents were successfully analyzed, falling back to standard processing")
                return await self._fallback_large_document_processing(query, document_chunks[:30], session_id, entity_names)
            
            # Step 3: Final synthesis with timeout
            logger.info(f"🔄 Synthesizing analysis from {len(analysis_memory)} documents")
            try:
                final_answer = await asyncio.wait_for(
                    self._synthesize_iterative_analysis(query, analysis_memory),
                    timeout=120.0  # 2 minute timeout for synthesis
                )
            except asyncio.TimeoutError:
                logger.error("🔄 Final synthesis timed out, using combined analyses")
                final_answer = f"Based on analysis of {len(analysis_memory)} documents:\n\n" + "\n\n---\n\n".join(analysis_memory)
            
            query_time = time.time() - start_time
            
            response = QueryResponse(
                answer=final_answer,
                citations=citations,
                session_id=session_id,
                query_time=query_time,
                retrieval_count=len(document_chunks)
            )
            
            # Store in conversation history
            await self._store_query_history(session_id, query, response)
            
            logger.info(f"✅ Iterative analysis completed in {query_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"❌ Iterative document analysis failed: {e}")
            import traceback
            logger.error(f"❌ Iterative analysis traceback: {traceback.format_exc()}")
            # Fallback to regular processing with truncated content
            return await self._fallback_large_document_processing(query, document_chunks[:30], session_id, entity_names)
    
    async def _process_document_iteratively(self, query: str, chunks: List[Dict[str, Any]], doc_id: str) -> str:
        """Process a single document iteratively in batches"""
        try:
            # Split chunks into manageable batches
            batch_size = 15  # Process 15 chunks at a time
            batch_analyses = []
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_content = "\n\n".join([chunk['content'] for chunk in batch_chunks])
                
                # Analyze this batch
                batch_analysis = await self._analyze_document_batch(query, batch_content, i // batch_size + 1, doc_id)
                if batch_analysis:
                    batch_analyses.append(batch_analysis)
                
                # Small delay to avoid overwhelming the LLM
                await asyncio.sleep(0.1)
            
            # Combine batch analyses
            if batch_analyses:
                combined_analysis = await self._combine_batch_analyses(query, batch_analyses, doc_id)
                return combined_analysis
            else:
                return f"Analysis of document {doc_id} completed, but no significant content found."
                
        except Exception as e:
            logger.error(f"❌ Iterative document processing failed for {doc_id}: {e}")
            return f"Unable to complete analysis of document {doc_id} due to processing error."
    
    async def _analyze_document_batch(self, query: str, batch_content: str, batch_number: int, doc_id: str) -> str:
        """Analyze a single batch of document content"""
        try:
            analysis_prompt = f"""Analyze this section of a document to answer the user's query.

User Query: "{query}"

Document Section {batch_number}:
{batch_content}

Provide a concise analysis of this section focusing on:
1. Key information relevant to the user's query
2. Important facts, headlines, or findings
3. Any significant details that answer the query

Keep your analysis focused and under 200 words. If this section doesn't contain relevant information, say "No relevant information in this section."""

            try:
                response = await asyncio.wait_for(
                    self.openai_client.chat.completions.create(
                        model=self.current_model,
                        messages=[
                            {"role": "system", "content": "You are an expert document analyst. Provide concise, focused analysis."},
                            {"role": "user", "content": analysis_prompt}
                        ],
                        max_tokens=300,
                        temperature=0.3
                    ),
                    timeout=30.0
                )
                
                if response.choices and response.choices[0].message.content:
                    analysis = response.choices[0].message.content.strip()
                    if "no relevant information" not in analysis.lower():
                        return f"Section {batch_number}: {analysis}"
                    else:
                        return None
                else:
                    return None
                    
            except asyncio.TimeoutError:
                logger.warning(f"🔄 Batch analysis timed out for section {batch_number}")
                return None
            except Exception as e:
                logger.warning(f"🔄 Batch analysis failed for section {batch_number}: {e}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Document batch analysis failed: {e}")
            return None
    
    async def _combine_batch_analyses(self, query: str, batch_analyses: List[str], doc_id: str) -> str:
        """Combine multiple batch analyses into a coherent document analysis"""
        try:
            combined_content = "\n\n".join(batch_analyses)
            
            synthesis_prompt = f"""Synthesize the following section analyses into a coherent document summary.

User Query: "{query}"

Section Analyses:
{combined_content}

Create a comprehensive summary that:
1. Directly answers the user's query
2. Combines information from all sections
3. Maintains logical flow and organization
4. Highlights the most important findings

Provide a well-structured response under 400 words."""

            try:
                response = await asyncio.wait_for(
                    self.openai_client.chat.completions.create(
                        model=self.current_model,
                        messages=[
                            {"role": "system", "content": "You are an expert at synthesizing document analyses into coherent summaries."},
                            {"role": "user", "content": synthesis_prompt}
                        ],
                        max_tokens=600,
                        temperature=0.4
                    ),
                    timeout=45.0
                )
                
                if response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content.strip()
                else:
                    return f"Combined analysis of document {doc_id}: {combined_content}"
                    
            except asyncio.TimeoutError:
                logger.warning(f"🔄 Batch synthesis timed out for document {doc_id}")
                return f"Combined analysis of document {doc_id}: {combined_content}"
            except Exception as e:
                logger.warning(f"🔄 Batch synthesis failed for document {doc_id}: {e}")
                return f"Combined analysis of document {doc_id}: {combined_content}"
                
        except Exception as e:
            logger.error(f"❌ Batch combination failed: {e}")
            return f"Analysis of document {doc_id} (combined from {len(batch_analyses)} sections)"
    
    async def _synthesize_iterative_analysis(self, query: str, analysis_memory: List[str]) -> str:
        """Synthesize all document analyses into a final comprehensive answer"""
        try:
            combined_analyses = "\n\n---\n\n".join(analysis_memory)
            
            final_prompt = f"""Based on the comprehensive document analyses provided, give a final answer to the user's query.

User Query: "{query}"

Document Analyses:
{combined_analyses}

Provide a complete, well-organized answer that:
1. Directly addresses the user's query
2. Synthesizes information from all documents
3. Maintains clarity and logical flow
4. Includes specific details and findings
5. Uses a professional but approachable tone

Your response should be comprehensive but well-structured."""

            try:
                response = await asyncio.wait_for(
                    self.openai_client.chat.completions.create(
                        model=self.current_model,
                        messages=[
                            {"role": "system", "content": "You are Plato, providing comprehensive answers based on document analysis. Be thorough but well-organized."},
                            {"role": "user", "content": final_prompt}
                        ],
                        max_tokens=1500,
                        temperature=0.5
                    ),
                    timeout=60.0
                )
                
                if response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content.strip()
                else:
                    return f"Based on the analysis of {len(analysis_memory)} documents: {combined_analyses}"
                    
            except asyncio.TimeoutError:
                logger.warning("🔄 Final synthesis timed out")
                return f"Based on comprehensive analysis of {len(analysis_memory)} documents:\n\n{combined_analyses}"
            except Exception as e:
                logger.warning(f"🔄 Final synthesis failed: {e}")
                return f"Based on analysis of {len(analysis_memory)} documents:\n\n{combined_analyses}"
                
        except Exception as e:
            logger.error(f"❌ Final synthesis failed: {e}")
            return f"Analysis completed for {len(analysis_memory)} documents, but synthesis encountered an error."
    
    async def _fallback_large_document_processing(self, query: str, chunks: List[Dict[str, Any]], session_id: str, entity_names: List[str]) -> QueryResponse:
        """Fallback processing for when iterative analysis fails"""
        try:
            logger.info("🔄 Using fallback processing for large document")
            
            # Use standard processing with limited chunks
            limited_chunks = chunks[:30]  # Limit to 30 chunks
            
            # Continue with normal processing flow
            context_parts = []
            citations = []
            
            for i, chunk in enumerate(limited_chunks):
                context_parts.append(f"[Source {i+1}]: {chunk['content']}")
                
                doc_metadata = await self._get_document_metadata(chunk['document_id'])
                doc_title = doc_metadata.get('title') or doc_metadata.get('filename', f"Document {chunk['document_id']}") if doc_metadata else f"Document {chunk['document_id']}"
                
                citation = Citation(
                    document_id=chunk['document_id'],
                    document_title=doc_title,
                    chunk_id=chunk['chunk_id'],
                    relevance_score=chunk['score'],
                    snippet=chunk['content'][:300] + "..." if len(chunk['content']) > 300 else chunk['content']
                )
                citations.append(citation)
            
            context_text = "\n\n".join(context_parts)
            conversation_history = await self._get_recent_conversation(session_id)
            
            # Generate response
            answer = await self._generate_entity_aware_response(query, context_text, conversation_history, entity_names)
            
            # Add note about processing limitation
            answer += f"\n\n*Note: This analysis is based on a subset of the available content due to document size. For a complete analysis, consider breaking down your query into more specific questions.*"
            
            return QueryResponse(
                answer=answer,
                citations=citations,
                session_id=session_id,
                query_time=0,
                retrieval_count=len(limited_chunks)
            )
            
        except Exception as e:
            logger.error(f"❌ Fallback processing failed: {e}")
            return QueryResponse(
                answer="I apologize, but I encountered difficulties processing this large document. Please try asking a more specific question or contact support.",
                citations=[],
                session_id=session_id,
                query_time=0,
                retrieval_count=0
            )

    async def close(self):
        """Clean up resources"""
        if self.redis_client:
            await self.redis_client.close()
        if self.embedding_manager:
            await self.embedding_manager.close()
        logger.info("🔄 Chat Service closed")

    async def _assess_chunk_sufficiency_with_document_selection(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced LLM assessment that can specify which documents it wants to see in full"""
        try:
            if not chunks:
                logger.info("🧠 No chunks retrieved, marking as INSUFFICIENT")
                return {"sufficiency": "INSUFFICIENT", "requested_documents": []}
            
            # Special handling for headline queries - they should NOT trigger full document retrieval
            query_lower = query.lower()
            if any(pattern in query_lower for pattern in ['headline', 'headlines', 'top stories', 'news summary']):
                logger.info("🧠 Headline query detected - checking for sufficient headline content")
                
                # Check if we have headline-like content in our chunks
                headline_chunks = 0
                for chunk in chunks:
                    content_lower = chunk['content'].lower()
                    if any(indicator in content_lower for indicator in ['headline', 'breaking', 'news', 'story', 'report']):
                        headline_chunks += 1
                
                if headline_chunks >= 5 or len(chunks) >= 20:
                    logger.info(f"🧠 Found {headline_chunks} headline-like chunks out of {len(chunks)} total - marking SUFFICIENT")
                    return {"sufficiency": "SUFFICIENT", "requested_documents": []}
                else:
                    logger.info(f"🧠 Only {headline_chunks} headline-like chunks found - will use targeted headline retrieval")
                    return {"sufficiency": "INSUFFICIENT", "requested_documents": []}
            
            # Prepare chunk context for LLM assessment with document information
            chunk_summary_with_docs = self._prepare_chunk_summary_with_document_info(query, chunks)
            
            assessment_prompt = f"""Analyze if these chunks are sufficient to answer the user's query.

USER QUERY: "{query}"

AVAILABLE CHUNKS:
{chunk_summary_with_docs}

TASK: Are these chunks SUFFICIENT or do you need full documents?

If INSUFFICIENT, list up to 3 Document IDs you want to see completely.

Return only this JSON:
{{
    "sufficiency": "SUFFICIENT",
    "requested_documents": [],
    "reasoning": "Brief explanation"
}}

OR

{{
    "sufficiency": "INSUFFICIENT", 
    "requested_documents": ["doc_id_1", "doc_id_2"],
    "reasoning": "Need full documents because..."
}}"""

            try:
                response = await asyncio.wait_for(
                    self.openai_client.chat.completions.create(
                        model=self.current_model,
                        messages=[
                            {"role": "system", "content": "You assess information sufficiency and can request specific documents. Respond with exactly the requested JSON format."},
                            {"role": "user", "content": assessment_prompt}
                        ],
                        max_tokens=500,
                        temperature=0.1
                    ),
                    timeout=600.0
                )
                
                if response.choices and response.choices[0].message.content:
                    assessment_text = response.choices[0].message.content.strip()
                    logger.info(f"🧠 LLM enhanced assessment: {assessment_text[:500]}...")
                    
                    # Parse JSON response
                    try:
                        import json
                        
                        # Try to extract JSON if it's embedded in other text
                        json_start = assessment_text.find('{')
                        json_end = assessment_text.rfind('}') + 1
                        
                        if json_start >= 0 and json_end > json_start:
                            json_text = assessment_text[json_start:json_end]
                            logger.info(f"🧠 Extracted JSON: {json_text}")
                            assessment = json.loads(json_text)
                        else:
                            logger.warning(f"🧠 No JSON found in response: '{assessment_text}'")
                            # Fallback: try to parse the whole thing
                            assessment = json.loads(assessment_text)
                        
                        sufficiency = assessment.get("sufficiency", "SUFFICIENT").upper()
                        requested_docs = assessment.get("requested_documents", [])
                        reasoning = assessment.get("reasoning", "")
                        confidence = assessment.get("confidence", "medium")
                        
                        # Validate and limit requested documents
                        if isinstance(requested_docs, list):
                            # Limit to 5 documents and ensure they're in our chunk set
                            available_doc_ids = {chunk['document_id'] for chunk in chunks}
                            valid_requested_docs = [
                                doc_id for doc_id in requested_docs[:5] 
                                if doc_id in available_doc_ids
                            ]
                        else:
                            valid_requested_docs = []
                        
                        logger.info(f"🧠 Assessment: {sufficiency}, Requested docs: {valid_requested_docs}")
                        logger.info(f"🧠 Reasoning: {reasoning}")
                        
                        return {
                            "sufficiency": sufficiency,
                            "requested_documents": valid_requested_docs,
                            "reasoning": reasoning,
                            "confidence": confidence
                        }
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"🧠 Failed to parse LLM assessment JSON: {e}")
                        logger.warning(f"🧠 Raw response was: '{assessment_text}'")
                        # Fallback: check for keywords in response
                        if "INSUFFICIENT" in assessment_text.upper():
                            return {"sufficiency": "INSUFFICIENT", "requested_documents": [], "reasoning": "JSON parse failed, detected INSUFFICIENT keyword", "confidence": "low"}
                        else:
                            return {"sufficiency": "SUFFICIENT", "requested_documents": [], "reasoning": "JSON parse failed, defaulting to SUFFICIENT", "confidence": "low"}
                else:
                    logger.warning("🧠 LLM chunk assessment returned empty, defaulting to SUFFICIENT")
                    return {"sufficiency": "SUFFICIENT", "requested_documents": []}
                    
            except asyncio.TimeoutError:
                logger.warning("🧠 LLM chunk assessment timed out")
                return self._fallback_sufficiency_assessment_enhanced(query, chunks)
            except Exception as e:
                logger.warning(f"🧠 LLM chunk assessment failed: {e}")
                return self._fallback_sufficiency_assessment_enhanced(query, chunks)
                
        except Exception as e:
            logger.error(f"❌ Enhanced chunk sufficiency assessment failed: {e}")
            return {"sufficiency": "SUFFICIENT", "requested_documents": []}

    def _prepare_chunk_summary_with_document_info(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Prepare chunk summary including document IDs and titles for LLM assessment"""
        try:
            # Group chunks by document
            doc_chunks = {}
            for chunk in chunks:
                doc_id = chunk['document_id']
                if doc_id not in doc_chunks:
                    doc_chunks[doc_id] = []
                doc_chunks[doc_id].append(chunk)
            
            summary_parts = []
            for doc_id, doc_chunk_list in doc_chunks.items():
                # Sort chunks by score for this document
                doc_chunk_list.sort(key=lambda x: x.get('score', 0), reverse=True)
                
                # Get document title/filename for better context
                doc_title = "Unknown Document"
                try:
                    # Try to get document metadata for title
                    if hasattr(self, 'document_service') and self.document_service:
                        import asyncio
                        # Note: This is a sync method, so we can't await here
                        # We'll include the doc_id and let the LLM work with that
                        doc_title = f"Document {doc_id}"
                    else:
                        doc_title = f"Document {doc_id}"
                except:
                    doc_title = f"Document {doc_id}"
                
                # Take top 3 chunks from this document for summary
                top_chunks = doc_chunk_list[:3]
                
                chunk_previews = []
                for i, chunk in enumerate(top_chunks):
                    preview = chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
                    score = chunk.get('score', 0)
                    chunk_previews.append(f"  Chunk {i+1} (score: {score:.3f}): {preview}")
                
                # Include both ID and any title information
                metadata = chunk.get('metadata', {}) if top_chunks else {}
                source_info = f"Document ID: {doc_id}"
                if metadata.get('filename'):
                    source_info += f" | File: {metadata['filename']}"
                if metadata.get('title'):
                    source_info += f" | Title: {metadata['title']}"
                
                summary_parts.append(f"{source_info}\n" + "\n".join(chunk_previews))
            
            return "\n\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare chunk summary with document info: {e}")
            # Fallback to simple summary
            return "\n\n".join([
                f"Chunk {i+1} from {chunk['document_id']}: {chunk['content'][:200]}..." 
                for i, chunk in enumerate(chunks[:10])
            ])

    def _fallback_sufficiency_assessment_enhanced(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced fallback assessment when LLM assessment fails"""
        query_lower = query.lower()
        
        # Patterns that typically require full document analysis
        comprehensive_patterns = [
            'headlines', 'all headlines', 'top headlines', 'today\'s headlines',
            'complete', 'comprehensive', 'entire', 'full', 'overview', 'summary',
            'everything about', 'all content', 'analyze all', 'summarize all'
        ]
        
        # Check if query requires comprehensive analysis
        needs_comprehensive = any(pattern in query_lower for pattern in comprehensive_patterns)
        
        # Check chunk quality
        if not chunks:
            return {"sufficiency": "INSUFFICIENT", "requested_documents": []}
        
        # Check if we have high-quality chunks
        high_quality_chunks = [c for c in chunks if c['score'] > 0.7]
        diverse_documents = len(set(chunk['document_id'] for chunk in chunks))
        
        if needs_comprehensive:
            logger.info(f"🧠 Fallback: Comprehensive query detected, marking INSUFFICIENT")
            # For comprehensive queries, suggest top documents
            doc_scores = {}
            for chunk in chunks:
                doc_id = chunk['document_id']
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = []
                doc_scores[doc_id].append(chunk.get('score', 0))
            
            # Get top 3 documents by average score
            top_docs = sorted(
                doc_scores.items(), 
                key=lambda x: sum(x[1]) / len(x[1]), 
                reverse=True
            )[:3]
            
            requested_docs = [doc_id for doc_id, scores in top_docs]
            
            return {
                "sufficiency": "INSUFFICIENT", 
                "requested_documents": requested_docs,
                "reasoning": "Comprehensive query requires full document analysis",
                "confidence": "high"
            }
        elif len(high_quality_chunks) >= 3 and diverse_documents >= 1:
            logger.info(f"🧠 Fallback: Good chunks available ({len(high_quality_chunks)} high-quality), marking SUFFICIENT")
            return {"sufficiency": "SUFFICIENT", "requested_documents": []}
        else:
            logger.info(f"🧠 Fallback: Poor chunk quality or coverage, marking INSUFFICIENT")
            return {"sufficiency": "INSUFFICIENT", "requested_documents": []}

    async def _intelligent_document_retrieval_with_llm_selection(
        self, 
        query: str, 
        initial_chunks: List[Dict[str, Any]], 
        requested_doc_ids: List[str], 
        entity_names: List[str]
    ) -> tuple[List[Dict[str, Any]], bool]:
        """Retrieve full documents requested by LLM plus remaining chunks from other documents"""
        try:
            logger.info(f"📚 LLM requested {len(requested_doc_ids)} documents: {requested_doc_ids}")
            
            # Step 1: Retrieve full content for requested documents
            requested_chunks = []
            requested_doc_set = set(requested_doc_ids)
            
            for doc_id in requested_doc_ids:
                try:
                    doc_chunks = await self.embedding_manager.get_all_document_chunks(doc_id)
                    if doc_chunks:
                        # Mark these as LLM-requested full documents
                        for chunk in doc_chunks:
                            chunk['retrieval_source'] = 'llm_requested_full'
                            chunk['score'] = 1.0  # High relevance since LLM specifically requested
                        requested_chunks.extend(doc_chunks)
                        logger.info(f"📚 Retrieved {len(doc_chunks)} chunks from requested document {doc_id}")
                    else:
                        logger.warning(f"📚 No chunks found for requested document {doc_id}")
                except Exception as e:
                    logger.error(f"📚 Failed to retrieve document {doc_id}: {e}")
            
            # Step 2: Filter initial chunks to exclude those from requested documents
            remaining_chunks = [
                chunk for chunk in initial_chunks 
                if chunk['document_id'] not in requested_doc_set
            ]
            
            logger.info(f"📚 Kept {len(remaining_chunks)} chunks from other documents")
            
            # Step 3: Combine full documents + remaining chunks
            all_chunks = requested_chunks + remaining_chunks
            
            # Step 4: Check size limits and handle accordingly
            total_chunks = len(all_chunks)
            max_chunks_for_direct = 50    # Can be higher since LLM specifically chose these
            max_chunks_for_iterative = 200
            
            logger.info(f"📚 Total chunks after LLM selection: {total_chunks}")
            
            if total_chunks > max_chunks_for_iterative:
                logger.warning(f"📚 Too many chunks even with LLM selection ({total_chunks}), truncating")
                # Prioritize LLM-requested documents, then highest scoring remaining chunks
                remaining_chunks.sort(key=lambda x: x.get('score', 0), reverse=True)
                max_remaining = max_chunks_for_iterative - len(requested_chunks)
                final_chunks = requested_chunks + remaining_chunks[:max_remaining]
                return final_chunks, True  # Use iterative processing
            elif total_chunks > max_chunks_for_direct:
                logger.info(f"📚 Using iterative processing for {total_chunks} chunks")
                return all_chunks, True  # Use iterative processing
            else:
                logger.info(f"📚 Using direct processing for {total_chunks} chunks")
                return all_chunks, False  # Use direct processing
            
        except Exception as e:
            logger.error(f"❌ LLM-guided document retrieval failed: {e}")
            # Fallback to original chunks
            return initial_chunks, False

    def _build_metadata_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Build comprehensive metadata context for LLM including fiction/non-fiction detection"""
        try:
            context_parts = []
            
            for i, chunk in enumerate(chunks):
                metadata = chunk.get('metadata', {})
                tags = metadata.get('tags', [])
                category = metadata.get('category', '')
                author = metadata.get('author', '')
                title = metadata.get('title', metadata.get('filename', 'Unknown'))
                doc_type = metadata.get('doc_type', '')
                publication_date = metadata.get('publication_date', '')
                
                # Detect fiction/non-fiction indicators
                fiction_indicators = [
                    'fiction', 'novel', 'story', 'tale', 'narrative', 'fictional',
                    'fantasy', 'science fiction', 'mystery', 'romance', 'thriller',
                    'drama', 'comedy', 'adventure', 'horror', 'western', 'historical fiction'
                ]
                
                non_fiction_indicators = [
                    'non-fiction', 'nonfiction', 'factual', 'research', 'study',
                    'academic', 'technical', 'manual', 'reference', 'textbook',
                    'history', 'biography', 'autobiography', 'memoir', 'journalism',
                    'news', 'report', 'analysis', 'documentary'
                ]
                
                # Determine content type
                tags_lower = ' '.join(tags).lower()
                category_lower = category.lower() if category else ''
                title_lower = title.lower() if title else ''
                
                # Enhanced fiction detection
                is_fiction = any(indicator in tags_lower or indicator in category_lower or indicator in title_lower
                               for indicator in fiction_indicators)
                is_non_fiction = any(indicator in tags_lower or indicator in category_lower or indicator in title_lower
                                   for indicator in non_fiction_indicators)
                
                # Determine source credibility
                source_type = "UNKNOWN"
                credibility_note = ""
                
                if is_fiction:
                    source_type = "FICTION"
                    credibility_note = "This source contains fictional content and should not be used for factual claims."
                elif is_non_fiction:
                    source_type = "NON-FICTION"
                    credibility_note = "This source contains factual content suitable for research and analysis."
                elif category_lower in ['academic', 'research', 'technical', 'medical', 'legal']:
                    source_type = "ACADEMIC/TECHNICAL"
                    credibility_note = "This source appears to be academic or technical in nature."
                elif category_lower in ['news', 'journalism']:
                    source_type = "NEWS/JOURNALISM"
                    credibility_note = "This source appears to be news or journalistic content."
                elif doc_type in ['pdf', 'docx', 'txt'] and not tags:
                    source_type = "DOCUMENT"
                    credibility_note = "This is a document with limited metadata - assess credibility based on content."
                
                # Build source information
                source_info = f"Source {i+1}: {title}"
                if author:
                    source_info += f" by {author}"
                if publication_date:
                    source_info += f" ({publication_date})"
                
                # Add metadata if available
                if metadata.get('series'):
                    source_info += f"\nSeries: {metadata['series']}"
                    if metadata.get('series_index'):
                        source_info += f" #{metadata['series_index']}"
                
                if metadata.get('publisher'):
                    source_info += f"\nPublisher: {metadata['publisher']}"
                
                if metadata.get('isbn'):
                    source_info += f"\nISBN: {metadata['isbn']}"
                
                if metadata.get('rating'):
                    source_info += f"\nRating: {metadata['rating']}/5"
                
                source_info += f"\nType: {source_type}"
                source_info += f"\nCategory: {category}" if category else "\nCategory: Unknown"
                source_info += f"\nDocument Type: {doc_type}" if doc_type else "\nDocument Type: Unknown"
                
                if tags:
                    source_info += f"\nTags: {', '.join(tags)}"
                
                source_info += f"\nCredibility Note: {credibility_note}"
                source_info += f"\nContent: {chunk.get('content', '')[:300]}..."
                
                context_parts.append(source_info)
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"❌ Failed to build metadata context: {e}")
            # Fallback to simple context
            return "\n\n".join([
                f"Source {i+1}: {chunk.get('content', '')[:300]}..." 
                for i, chunk in enumerate(chunks)
            ])
