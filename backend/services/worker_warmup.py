"""
Worker Warm-up Service
Pre-initialize services in Celery workers for faster response times
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class WorkerWarmupService:
    """Service for warming up Celery workers with pre-initialized services"""
    
    def __init__(self):
        self.services = {}
        self.is_warmed_up = False
        self.warmup_start_time = None
        self.warmup_completion_time = None
        self._shared_db_pool = None
    
    async def warmup_worker(self) -> Dict[str, Any]:
        """
        Warm up the worker by pre-initializing all heavy services
        This should be called when the worker starts
        """
        if self.is_warmed_up:
            logger.info("🔥 Worker already warmed up, skipping...")
            return {"status": "already_warm", "services": list(self.services.keys())}
        
        self.warmup_start_time = datetime.now()
        logger.info("🔥 ROOSEVELT'S WORKER WARMUP: Starting service pre-initialization...")
        
        # Initialize shared database pool first
        try:
            from utils.shared_db_pool import get_shared_db_pool
            self._shared_db_pool = await get_shared_db_pool()
            logger.info("✅ Shared database pool initialized for worker warmup")
        except Exception as e:
            logger.error(f"❌ Failed to initialize shared database pool: {e}")
            return {"status": "failed", "error": f"Database pool initialization failed: {e}"}
        
        warmup_tasks = []
        
        # Create parallel warmup tasks for all services
        # Note: Chat service removed - not used by any active Celery tasks (only deprecated base_agent)
        # Note: Prompt service removed - lightweight service, doesn't need warmup (used lazily by grpc_context_gatherer)
        warmup_tasks.extend([
            self._warmup_settings_service(),
            self._warmup_conversation_service(),
            self._warmup_category_service(),
        ])
        
        # Execute all warmup tasks in parallel
        results = await asyncio.gather(*warmup_tasks, return_exceptions=True)
        
        # Process results
        successful_services = []
        failed_services = []
        
        service_names = ["settings", "conversation", "category"]
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_services.append(f"{service_names[i]}: {result}")
                logger.error(f"❌ {service_names[i]} service warmup failed: {result}")
            else:
                successful_services.append(service_names[i])
                logger.info(f"✅ {service_names[i]} service warmed up successfully")
        
        self.warmup_completion_time = datetime.now()
        warmup_duration = (self.warmup_completion_time - self.warmup_start_time).total_seconds()
        
        if failed_services:
            logger.warning(f"⚠️ Worker warmup completed with {len(failed_services)} failures in {warmup_duration:.2f}s")
            return {
                "status": "partial_success",
                "successful_services": successful_services,
                "failed_services": failed_services,
                "duration_seconds": warmup_duration
            }
        else:
            self.is_warmed_up = True
            logger.info(f"✅ ROOSEVELT'S WORKER WARMUP: All services warmed up successfully in {warmup_duration:.2f}s")
            return {
                "status": "success",
                "services": successful_services,
                "duration_seconds": warmup_duration
            }
    
    async def _warmup_settings_service(self):
        """Pre-initialize settings service"""
        try:
            from services.settings_service import settings_service
            
            if not hasattr(settings_service, '_initialized') or not settings_service._initialized:
                await settings_service.initialize()
            
            # Pre-load commonly accessed settings
            enabled_models = await settings_service.get_enabled_models()
            llm_model = await settings_service.get_llm_model()
            
            self.services['settings'] = {
                'service': settings_service,
                'enabled_models': enabled_models,
                'current_model': llm_model,
                'status': 'ready'
            }
            
            logger.debug(f"🔧 Settings service warmed: {len(enabled_models)} models, current: {llm_model}")
            
        except Exception as e:
            logger.error(f"❌ Settings service warmup failed: {e}")
            raise
    
    async def _warmup_chat_service(self):
        """Pre-initialize chat service"""
        try:
            from services.chat_service import ChatService
            
            chat_service = ChatService()
            await chat_service.initialize()
            
            # Ensure model is selected
            if not chat_service.current_model:
                await chat_service._auto_select_model()
            
            self.services['chat'] = {
                'service': chat_service,
                'current_model': chat_service.current_model,
                'models_enabled': chat_service.models_enabled,
                'status': 'ready'
            }
            
            logger.debug(f"💬 Chat service warmed: model={chat_service.current_model}, enabled={chat_service.models_enabled}")
            
        except Exception as e:
            logger.error(f"❌ Chat service warmup failed: {e}")
            raise
    
    async def _warmup_conversation_service(self):
        """Pre-initialize conversation service"""
        try:
            from services.conversation_service import ConversationService
            
            conversation_service = ConversationService()
            # ConversationService doesn't need async initialization, but we can warm up the connection
            
            self.services['conversation'] = {
                'service': conversation_service,
                'status': 'ready'
            }
            
            logger.debug("💬 Conversation service warmed")
            
        except Exception as e:
            logger.error(f"❌ Conversation service warmup failed: {e}")
            raise
    
    async def _warmup_prompt_service(self):
        """Pre-initialize prompt service"""
        try:
            from services.prompt_service import PromptService
            
            prompt_service = PromptService()
            # PromptService is typically initialized with the app, just verify it's accessible
            
            self.services['prompt'] = {
                'service': prompt_service,
                'status': 'ready'
            }
            
            logger.debug("📝 Prompt service warmed")
            
        except Exception as e:
            logger.error(f"❌ Prompt service warmup failed: {e}")
            raise
    
    async def _warmup_category_service(self) -> Dict[str, Any]:
        """Warm up the category service with shared database pool"""
        try:
            from services.category_service import CategoryService
            
            category_service = CategoryService()
            await category_service.initialize(shared_db_pool=self._shared_db_pool)
            
            self.services["category"] = category_service
            return {"status": "success", "service": "category"}
            
        except Exception as e:
            logger.error(f"❌ Category service warmup failed: {e}")
            raise
    
    def get_service(self, service_name: str) -> Optional[Any]:
        """Get a pre-warmed service"""
        if not self.is_warmed_up:
            logger.warning(f"⚠️ Worker not warmed up, service {service_name} may not be ready")
            return None
        
        service_data = self.services.get(service_name)
        if service_data and service_data['status'] == 'ready':
            return service_data['service']
        
        return None
    
    def get_warmup_status(self) -> Dict[str, Any]:
        """Get current warmup status"""
        return {
            "is_warmed_up": self.is_warmed_up,
            "services_count": len(self.services),
            "services": list(self.services.keys()),
            "warmup_duration": (
                (self.warmup_completion_time - self.warmup_start_time).total_seconds()
                if self.warmup_start_time and self.warmup_completion_time
                else None
            )
        }


# Global warmup service instance
worker_warmup_service = WorkerWarmupService()
