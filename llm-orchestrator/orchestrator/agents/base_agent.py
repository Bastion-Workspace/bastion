"""
Base Agent Class for LLM Orchestrator Agents
Provides common functionality for all agents running in the llm-orchestrator microservice
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from openai import NotFoundError, APIError, RateLimitError, AuthenticationError

from config.settings import settings
from orchestrator.middleware.message_preprocessor import MessagePreprocessor

logger = logging.getLogger(__name__)


class OpenRouterError(Exception):
    """Custom exception for OpenRouter API errors with user-friendly messages"""
    def __init__(self, user_message: str, error_type: str, original_error: str = ""):
        self.user_message = user_message
        self.error_type = error_type
        self.original_error = original_error
        super().__init__(user_message)


class TaskStatus(str, Enum):
    """Agent task completion status"""
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    PERMISSION_REQUIRED = "permission_required"
    ERROR = "error"


class BaseAgent:
    """
    Base class for all LLM Orchestrator agents
    
    Provides:
    - LLM access with proper configuration
    - Message history management
    - Common helper methods
    - Error handling patterns
    - Centralized workflow management with PostgreSQL checkpointing
    """
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.llm = None
        self.workflow = None  # Will be initialized lazily with checkpointer
        self._workflow_lock = asyncio.Lock()
        logger.info(f"Initializing {agent_type} agent")
    
    async def _get_workflow(self) -> StateGraph:
        """
        Get or build workflow with checkpointer (lazy initialization)
        
        This method ensures all agents automatically get PostgreSQL checkpointing.
        Subclasses should implement _build_workflow(checkpointer) to define their workflow.
        """
        if self.workflow is not None:
            return self.workflow
        
        async with self._workflow_lock:
            # Double-check after acquiring lock
            if self.workflow is not None:
                return self.workflow
            
            # Get checkpointer
            from orchestrator.checkpointer import get_async_postgres_saver
            checkpointer = await get_async_postgres_saver()
            
            # Build and compile workflow with checkpointer
            self.workflow = self._build_workflow(checkpointer)
            logger.info(f"✅ {self.agent_type} workflow compiled with PostgreSQL checkpointer")
            return self.workflow
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """
        Build LangGraph workflow for this agent
        
        Subclasses must implement this method to define their workflow.
        The checkpointer parameter is provided automatically for state persistence.
        
        Args:
            checkpointer: AsyncPostgresSaver instance for state persistence
            
        Returns:
            Compiled StateGraph with checkpointer
        """
        raise NotImplementedError("Subclasses must implement _build_workflow(checkpointer) method")
    
    def _get_checkpoint_config(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get checkpoint configuration for workflow invocation
        
        Creates normalized thread_id matching backend format: {user_id}:{conversation_id}
        This ensures checkpoints are readable by both orchestrator and backend.
        
        Args:
            metadata: Optional metadata dictionary containing conversation_id and user_id
            
        Returns:
            Configuration dict with thread_id for checkpointing
        """
        metadata = metadata or {}
        user_id = metadata.get("user_id", "system")
        conversation_id = metadata.get("conversation_id")
        branch_suffix = metadata.get("branch_thread_suffix")

        # Use normalized thread_id format matching backend: {user_id}:{conversation_id}[:branch_{suffix}]
        if conversation_id:
            if ":" in conversation_id and conversation_id.startswith(f"{user_id}:"):
                thread_id = conversation_id
            else:
                thread_id = f"{user_id}:{conversation_id}"
            if branch_suffix:
                thread_id = f"{thread_id}:branch_{branch_suffix}"
        else:
            thread_id = f"thread_{user_id}"

        return {
            "configurable": {
                "thread_id": thread_id
            }
        }
    
    async def _load_checkpoint_shared_memory(
        self,
        workflow: Any,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Load shared_memory from checkpoint state
        
        This ensures continuity data like primary_agent_selected is preserved.
        
        Args:
            workflow: Compiled LangGraph workflow
            config: Checkpoint configuration with thread_id
            
        Returns:
            Dict with shared_memory from checkpoint, or empty dict if not found
        """
        try:
            checkpoint_state = await workflow.aget_state(config)
            if checkpoint_state and checkpoint_state.values:
                shared_memory = checkpoint_state.values.get("shared_memory", {})
                if shared_memory:
                    logger.info(f"📚 Loaded shared_memory from checkpoint: {list(shared_memory.keys())}")
                    return shared_memory
            return {}
        except Exception as e:
            logger.debug(f"⚠️ Failed to load checkpoint shared_memory: {e}")
            return {}
    
    async def _load_and_merge_checkpoint_messages(
        self, 
        workflow: Any, 
        config: Dict[str, Any], 
        new_messages: List[Any],
        look_back_limit: int = 20
    ) -> List[Any]:
        """
        Load checkpointed messages and merge with new messages

        This ensures conversation history is preserved across requests.
        Uses a standardized look-back limit (default 20 messages) to keep context manageable.

        Args:
            workflow: Compiled LangGraph workflow
            config: Checkpoint configuration with thread_id
            new_messages: New messages to add (typically just the current query)
            look_back_limit: Maximum number of previous messages to keep (default: 20)
            
        Returns:
            Merged list of messages with checkpointed history + new messages (limited to look_back_limit)
        """
        try:
            # Try to load existing checkpoint state
            checkpoint_state = await workflow.aget_state(config)
            
            if checkpoint_state and checkpoint_state.values:
                # Get existing messages from checkpoint
                checkpointed_messages = checkpoint_state.values.get("messages", [])
                
                if checkpointed_messages:
                    # If backend sent a full history that exceeds the lookback,
                    # trust the DB as source of truth -- skip checkpoint merge.
                    if len(new_messages) >= look_back_limit:
                        logger.info(
                            "Backend sent %d messages (>= lookback %d), using fresh DB history",
                            len(new_messages), look_back_limit,
                        )
                        return new_messages[-look_back_limit:]

                    # Apply look-back limit: keep only the last N messages
                    # This ensures we have recent context without overwhelming the LLM
                    if len(checkpointed_messages) > look_back_limit:
                        checkpointed_messages = checkpointed_messages[-look_back_limit:]
                        logger.info(f"📚 Loaded {len(checkpointed_messages)} messages from checkpoint (limited from larger history)")
                    else:
                        logger.info(f"📚 Loaded {len(checkpointed_messages)} messages from checkpoint")
                    
                    # Find where checkpointed history ends within new_messages so we only append
                    # genuinely new messages. Prefer sequence_number (stable); fall back to content match.
                    def _message_seq(msg: Any) -> Optional[int]:
                        ak = getattr(msg, "additional_kwargs", None) or {}
                        if "sequence_number" not in ak:
                            return None
                        try:
                            return int(ak["sequence_number"])
                        except (TypeError, ValueError):
                            return None

                    last_ckpt = checkpointed_messages[-1]
                    last_ckpt_seq = _message_seq(last_ckpt)
                    new_messages_have_seq = any(_message_seq(m) is not None for m in new_messages)

                    if last_ckpt_seq is not None and last_ckpt_seq > 0 and new_messages_have_seq:
                        start_idx = len(new_messages)
                        for i, m in enumerate(new_messages):
                            s = _message_seq(m)
                            if s is not None and s > last_ckpt_seq:
                                start_idx = i
                                break
                        truly_new = new_messages[start_idx:]
                    else:
                        start_idx = 0
                        last_ckpt_content = getattr(last_ckpt, "content", None)
                        if last_ckpt_content:
                            for i in range(len(new_messages) - 1, -1, -1):
                                if getattr(new_messages[i], "content", None) == last_ckpt_content:
                                    start_idx = i + 1
                                    break
                        truly_new = new_messages[start_idx:]
                    merged_messages = list(checkpointed_messages) + truly_new

                    # Apply look-back limit to final merged messages too
                    if len(merged_messages) > look_back_limit:
                        merged_messages = merged_messages[-look_back_limit:]
                    
                    return merged_messages
                else:
                    logger.debug("No checkpointed messages found, using new messages only")
                    return new_messages
            else:
                logger.debug("No checkpoint state found, starting fresh conversation")
                return new_messages
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to load checkpoint messages: {e}, starting fresh")
            return new_messages
    
    def _get_llm(self, temperature: float = 0.7, model: Optional[str] = None, state: Optional[Dict[str, Any]] = None, max_tokens: Optional[int] = None) -> ChatOpenAI:
        """Get configured LLM instance, using user model preferences if available"""
        # Check for user model preferences in state metadata or shared_memory
        user_model = None
        if state:
            if model is None:  # Only override if no explicit model provided
                # First check metadata (standard location)
                metadata = state.get("metadata", {})
                user_model = metadata.get("user_chat_model")
                
                # Fallback to shared_memory (for agents like research that don't have metadata in state)
                if not user_model:
                    shared_memory = state.get("shared_memory", {})
                    user_model = shared_memory.get("user_chat_model")
                
                logger.debug(f"🔍 MODEL SELECTION: user_chat_model from metadata/shared_memory = {user_model}")
        
        # Use user model, explicit model, or default
        final_model = model or user_model or settings.DEFAULT_MODEL
        logger.info(f"🎯 SELECTED MODEL: {final_model} (explicit={model}, user={user_model}, default={settings.DEFAULT_MODEL})")
        logger.info(f"🌡️ SELECTED TEMPERATURE: {temperature}")
        if max_tokens:
            logger.info(f"🔢 SELECTED MAX_TOKENS: {max_tokens}")
        
        # Add reasoning support via extra_body (skip for providers that do not support it, e.g. Groq)
        # Note: LangChain ChatOpenAI passes model_kwargs to the underlying OpenAI client
        # extra_body is a top-level parameter in OpenAI SDK, so we include it in model_kwargs
        metadata = state.get("metadata", {}) if state else {}
        provider_type = metadata.get("user_llm_provider_type") or ""
        skip_reasoning = provider_type in ("groq",)
        model_kwargs = {}
        if not skip_reasoning:
            from orchestrator.utils.llm_reasoning_utils import add_reasoning_to_extra_body
            reasoning_extra_body = add_reasoning_to_extra_body(extra_body=None, model=final_model, use_enabled_flag=True)
            if reasoning_extra_body:
                model_kwargs["extra_body"] = reasoning_extra_body
                logger.info(f"Reasoning enabled for model {final_model}: {reasoning_extra_body}")
        else:
            logger.debug(f"Reasoning extras skipped for provider_type={provider_type}")

        # User-level LLM providers: use per-user API key and base_url when present
        user_api_key = metadata.get("user_llm_api_key")
        user_base_url = metadata.get("user_llm_base_url")
        api_key = user_api_key if user_api_key else settings.OPENROUTER_API_KEY
        base_url = user_base_url if user_base_url else settings.OPENROUTER_BASE_URL

        return ChatOpenAI(
            model=final_model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            model_kwargs=model_kwargs if model_kwargs else None
        )

    def _handle_openrouter_error(self, error: Exception) -> Dict[str, Any]:
        """
        Transform OpenRouter API errors into user-friendly messages
        
        Args:
            error: The exception raised by OpenRouter API
            
        Returns:
            Dict with user-friendly error message and error type
        """
        error_type = "api_error"
        user_message = "An error occurred while processing your request."
        
        # Handle specific OpenRouter error types
        if isinstance(error, NotFoundError):
            error_str = str(error)
            # Check for data policy error
            if "data policy" in error_str.lower() or "free model training" in error_str.lower():
                user_message = (
                    "⚠️ **OpenRouter Data Policy Configuration**\n\n"
                    "No endpoints found matching your data policy settings. This means your OpenRouter account "
                    "has data policy restrictions that prevent the selected model from being used.\n\n"
                    "**To fix this:**\n"
                    "1. Visit https://openrouter.ai/settings/privacy\n"
                    "2. Review your data policy settings (Free model training, etc.)\n"
                    "3. Adjust your privacy/data policy preferences to allow the model you're trying to use\n"
                    "4. Try your request again\n\n"
                    "**Note:** Some models may require specific data policy settings. Check the model's requirements "
                    "on OpenRouter if this issue persists."
                )
                error_type = "data_policy_error"
            # Check for ignored providers error
            elif "All providers have been ignored" in error_str or "ignored providers" in error_str.lower():
                user_message = (
                    "⚠️ **OpenRouter Configuration Issue**\n\n"
                    "All available AI providers have been ignored in your OpenRouter settings. "
                    "This means no models are available to process your request.\n\n"
                    "**To fix this:**\n"
                    "1. Visit https://openrouter.ai/settings/preferences\n"
                    "2. Review your ignored providers list\n"
                    "3. Remove providers from the ignored list or adjust your account requirements\n"
                    "4. Try your request again\n\n"
                    "If you need help, check your OpenRouter account settings or contact support."
                )
                error_type = "provider_configuration_error"
            else:
                user_message = (
                    f"⚠️ **Model Not Found**\n\n"
                    f"The requested AI model is not available. This could mean:\n"
                    f"- The model name is incorrect\n"
                    f"- The model is not available in your OpenRouter account\n"
                    f"- Your account doesn't have access to this model\n"
                    f"- Your data policy settings don't match this model's requirements\n\n"
                    f"**Error details:** {str(error)}\n\n"
                    f"**To resolve:** Check your OpenRouter settings at https://openrouter.ai/settings/privacy"
                )
                error_type = "model_not_found"
        
        elif isinstance(error, RateLimitError):
            user_message = (
                "⚠️ **Rate Limit Exceeded**\n\n"
                "You've exceeded the rate limit for API requests. Please wait a moment and try again.\n\n"
                "If this persists, you may need to:\n"
                "- Upgrade your OpenRouter plan\n"
                "- Reduce the frequency of requests\n"
                "- Check your account usage limits"
            )
            error_type = "rate_limit_error"
        
        elif isinstance(error, AuthenticationError):
            user_message = (
                "⚠️ **Authentication Error**\n\n"
                "There's an issue with your OpenRouter API credentials. Please check:\n"
                "- Your API key is correct and active\n"
                "- Your account has sufficient credits\n"
                "- Your account is in good standing"
            )
            error_type = "authentication_error"
        
        elif isinstance(error, APIError):
            error_str = str(error)
            # Check for common OpenRouter-specific errors
            if "account requirements" in error_str.lower() or "provider" in error_str.lower():
                user_message = (
                    "⚠️ **OpenRouter Account Configuration**\n\n"
                    "Your OpenRouter account settings are preventing this request. "
                    "This could be due to:\n"
                    "- Provider restrictions in your account preferences\n"
                    "- Account requirements not being met by available providers\n"
                    "- Model availability restrictions\n\n"
                    "**To resolve:**\n"
                    "1. Visit https://openrouter.ai/settings/preferences\n"
                    "2. Review and adjust your provider preferences\n"
                    "3. Check your account requirements\n"
                    "4. Try your request again\n\n"
                    f"**Error details:** {error_str}"
                )
                error_type = "account_configuration_error"
            else:
                user_message = (
                    f"⚠️ **API Error**\n\n"
                    f"An error occurred while communicating with OpenRouter:\n\n"
                    f"**Error details:** {error_str}"
                )
                error_type = "api_error"
        
        else:
            # Generic error - check if it's an OpenAI-compatible error
            error_str = str(error)
            # Check for OpenRouter errors in the error message
            if "openrouter" in error_str.lower() or "provider" in error_str.lower() or "data policy" in error_str.lower():
                # Check for data policy errors even in generic exceptions
                if "data policy" in error_str.lower() or "free model training" in error_str.lower():
                    user_message = (
                        "⚠️ **OpenRouter Data Policy Configuration**\n\n"
                        "No endpoints found matching your data policy settings. This means your OpenRouter account "
                        "has data policy restrictions that prevent the selected model from being used.\n\n"
                        "**To fix this:**\n"
                        "1. Visit https://openrouter.ai/settings/privacy\n"
                        "2. Review your data policy settings (Free model training, etc.)\n"
                        "3. Adjust your privacy/data policy preferences to allow the model you're trying to use\n"
                        "4. Try your request again\n\n"
                        "**Note:** Some models may require specific data policy settings. Check the model's requirements "
                        "on OpenRouter if this issue persists."
                    )
                    error_type = "data_policy_error"
                else:
                    user_message = (
                        f"⚠️ **OpenRouter Error**\n\n"
                        f"An error occurred with OpenRouter:\n\n"
                        f"**Error details:** {error_str}\n\n"
                        f"If this persists, check your OpenRouter account settings at "
                        f"https://openrouter.ai/settings/preferences"
                    )
                    error_type = "openrouter_error"
            else:
                user_message = f"An unexpected error occurred: {error_str}"
        
        return {
            "error_message": user_message,
            "error_type": error_type,
            "original_error": str(error)
        }

    def _prepare_messages_with_query(self, messages: Optional[List[Any]], query: str) -> List[Any]:
        """
        Prepare messages list with current user query for checkpoint persistence
        
        This ensures the current user query is added to messages before workflow invocation,
        matching the backend's behavior of adding queries to conversation history.
        
        Args:
            messages: Optional existing conversation messages
            query: Current user query to add
            
        Returns:
            List of messages with current query appended
        """
        from langchain_core.messages import HumanMessage
        conversation_messages = list(messages) if messages else []
        conversation_messages.append(HumanMessage(content=query))
        return conversation_messages

    def _clear_request_scoped_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clear request-scoped data from shared_memory before checkpoint save
        
        Request-scoped data (like active_editor and active_artifact) should:
        - Persist during a single request (for subgraph communication)
        - Be cleared before checkpoint save (it's request-scoped, not conversation-scoped)
        
        This ensures:
        - Subgraphs can access active_editor / active_artifact during the request
        - active_editor and active_artifact don't persist in checkpoint (prevents stale data)
        - Each request gets fresh editor and artifact state from the frontend
        
        Args:
            state: Current LangGraph state
            
        Returns:
            Updated state with request-scoped data cleared from shared_memory
        """
        shared_memory = state.get("shared_memory", {})
        to_clear = [k for k in ("active_editor", "active_data_workspace", "active_artifact") if k in shared_memory]
        if shared_memory and to_clear:
            shared_memory = shared_memory.copy()
            for key in to_clear:
                del shared_memory[key]
            logger.debug("Cleared request-scoped data from shared_memory (before checkpoint save): %s", to_clear)
            state["shared_memory"] = shared_memory
        return state

    PINNED_DOCUMENT_TTL_MINUTES = 30

    def _get_valid_pinned_document(self, shared_memory: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return pinned_document if it exists and hasn't expired. None otherwise."""
        pin = shared_memory.get("pinned_document")
        if not pin or not pin.get("document_id"):
            return None
        last_active = pin.get("last_active_at") or pin.get("pinned_at")
        if last_active:
            from datetime import timezone, timedelta
            try:
                dt = datetime.fromisoformat(last_active.replace("Z", "+00:00"))
                age = datetime.now(timezone.utc) - dt
                if age > timedelta(minutes=self.PINNED_DOCUMENT_TTL_MINUTES):
                    return None
            except Exception:
                pass
        return pin

    def _pin_document(
        self,
        shared_memory: Dict[str, Any],
        document_id: str,
        title: str = "",
        filename: str = "",
    ) -> Dict[str, Any]:
        """Update or set pinned_document in shared_memory."""
        from datetime import timezone
        now = datetime.now(timezone.utc).isoformat()
        existing = shared_memory.get("pinned_document", {})
        shared_memory["pinned_document"] = {
            "document_id": document_id,
            "title": title or existing.get("title", ""),
            "filename": filename or existing.get("filename", ""),
            "pinned_at": existing.get("pinned_at", now) if existing.get("document_id") == document_id else now,
            "last_active_at": now,
        }
        return shared_memory

    LAST_TOOL_RESULTS_MAX_BYTES = 8192
    _BULK_FIELDS = frozenset({"content", "formatted", "body", "text", "html", "geometry", "steps"})
    _REF_FIELD_SUFFIXES = ("_id", "_url")
    _REF_FIELD_NAMES = frozenset({"title", "filename", "name", "subject", "label", "success", "count", "applied_count"})
    _MAX_SINGLE_VALUE_CHARS = 2000
    _MAX_LIST_ITEM_CHARS = 500
    _MAX_LIST_ITEMS = 5

    def _extract_persistable_refs(self, typed_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract reference-like fields from a tool result; drop bulk content. Used for last_tool_results."""
        if not typed_dict or not isinstance(typed_dict, dict):
            return {}
        out: Dict[str, Any] = {}
        for k, v in typed_dict.items():
            if k in self._BULK_FIELDS:
                continue
            if not (k.endswith(self._REF_FIELD_SUFFIXES) or k in self._REF_FIELD_NAMES):
                continue
            if v is None:
                out[k] = None
                continue
            if isinstance(v, (str, int, float, bool)):
                s = str(v)
                if len(s) > self._MAX_SINGLE_VALUE_CHARS:
                    out[k] = s[: self._MAX_SINGLE_VALUE_CHARS] + "..."
                else:
                    out[k] = v
                continue
            if isinstance(v, list):
                kept = []
                for i, item in enumerate(v):
                    if i >= self._MAX_LIST_ITEMS:
                        kept.append(f"... and {len(v) - self._MAX_LIST_ITEMS} more")
                        break
                    if isinstance(item, (str, int, float, bool)):
                        s = str(item)
                        if len(s) > self._MAX_LIST_ITEM_CHARS:
                            kept.append(s[: self._MAX_LIST_ITEM_CHARS] + "...")
                        else:
                            kept.append(item)
                    elif isinstance(item, dict):
                        sub = self._extract_persistable_refs(item)
                        if sub:
                            kept.append(sub)
                    else:
                        kept.append(str(item)[: self._MAX_LIST_ITEM_CHARS])
                if kept:
                    out[k] = kept
                continue
            if isinstance(v, dict):
                sub = self._extract_persistable_refs(v)
                if sub:
                    out[k] = sub
        return out

    def _persist_tool_results_to_shared_memory(
        self,
        shared_memory: Dict[str, Any],
        tool_outputs_typed: Dict[str, Any],
    ) -> None:
        """Store extracted refs from tool_outputs_typed in shared_memory['last_tool_results'] with 8KB budget."""
        if not tool_outputs_typed:
            return
        extracted: Dict[str, Any] = {}
        for tool_key, typed in tool_outputs_typed.items():
            if not isinstance(typed, dict):
                continue
            refs = self._extract_persistable_refs(typed)
            if refs:
                extracted[tool_key] = refs
        if not extracted:
            return
        import json
        try:
            raw = json.dumps(extracted, default=str)
        except Exception:
            return
        if len(raw) > self.LAST_TOOL_RESULTS_MAX_BYTES:
            while extracted and len(raw) > self.LAST_TOOL_RESULTS_MAX_BYTES:
                first_key = next(iter(extracted))
                del extracted[first_key]
                raw = json.dumps(extracted, default=str)
        if extracted:
            shared_memory["last_tool_results"] = extracted

    def _get_datetime_context(self, state: Optional[Dict[str, Any]] = None) -> str:
        """
        Get current date/time context for agent grounding using user's timezone
        
        Args:
            state: Optional state dictionary containing user_id and shared_memory.
                   If provided, will use user's timezone from shared_memory or fetch from backend.
                   If not provided or timezone not found, defaults to UTC.
        
        Returns:
            Formatted datetime string for inclusion in prompts.
            This ensures all agents know the current date/time for proper grounding.
        """
        from datetime import datetime, timezone as dt_timezone
        import pytz
        
        # Try to get user timezone from state
        user_timezone = "UTC"  # Default fallback
        
        if state:
            # First check shared_memory for timezone (passed from backend)
            shared_memory = state.get("shared_memory", {})
            if "user_timezone" in shared_memory:
                user_timezone = shared_memory["user_timezone"]
                logger.debug(f"🌍 Using timezone from shared_memory: {user_timezone}")
            else:
                # Try to get from metadata
                metadata = state.get("metadata", {})
                if "user_timezone" in metadata:
                    user_timezone = metadata["user_timezone"]
                    logger.debug(f"🌍 Using timezone from metadata: {user_timezone}")
                else:
                    # Could fetch via gRPC, but for now fallback to UTC
                    # This ensures backward compatibility
                    logger.debug("🌍 No timezone in state, using UTC")
        
        # Get timezone-aware datetime
        try:
            if user_timezone.upper() == "UTC":
                current_time = datetime.now(dt_timezone.utc)
                timezone_name = "UTC"
            else:
                # For pytz timezones, use the recommended pytz approach
                tz = pytz.timezone(user_timezone)
                # Get naive UTC time, then localize to target timezone
                utc_naive = datetime.utcnow()
                utc_aware = pytz.utc.localize(utc_naive)
                current_time = utc_aware.astimezone(tz)
                timezone_name = current_time.strftime('%Z')  # Use strftime to get timezone abbreviation
        except Exception as e:
            # Fallback to UTC if timezone is invalid
            logger.warning(f"Invalid timezone '{user_timezone}', falling back to UTC: {e}")
            current_time = datetime.now(dt_timezone.utc)
            timezone_name = "UTC"
        
        current_date = current_time.strftime("%A, %B %d, %Y")
        iso_date = current_time.strftime("%Y-%m-%d")
        current_time_str = current_time.strftime("%I:%M %p")
        current_year = current_time.year

        # Format timezone more clearly (e.g., "EST" or "America/New_York")
        if timezone_name == "UTC":
            timezone_display = "UTC"
        else:
            # Show both abbreviation and full timezone name for clarity
            timezone_display = f"{timezone_name} ({user_timezone})"

        return (
            f"**Current Context (LOCAL TIME) — for grounding only:**\n"
            f"- Today's date: {current_date}\n"
            f"- Today's date (YYYY-MM-DD, use for search/filter): {iso_date}\n"
            f"- Current time: {current_time_str} {timezone_display}\n"
            f"- Current year: {current_year}\n"
            f"- **IMPORTANT**: This is the LOCAL time in the user's timezone ({user_timezone}), NOT UTC.\n"
            f"- This date/time is provided so you can interpret \"today\", \"current\", \"recent\", and \"latest\" correctly. It is NOT from search results or documents — it is the user's actual current date.\n"
            f"- When users refer to \"today\", \"yesterday\", \"this week\", \"this month\", or \"this year\", use this date context to understand what they mean.\n"
            f"- When answering time/date questions, use this LOCAL time directly - do NOT convert to UTC unless specifically requested."
        )

    def _build_conversational_agent_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        messages_list: List[Any],
        look_back_limit: int = 10,
        state: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        Build message list for non-editing conversational agents with conversation history
        
        STANDARDIZED METHOD FOR NON-EDITING AGENTS (Chat, Electronics, Reference,
        Dictionary, Entertainment, Data Formatting, General Project)
        
        Message Structure:
        1. SystemMessage: system_prompt
        2. SystemMessage: datetime_context (automatic, uses user timezone if available)
        3. Conversation history as alternating HumanMessage/AIMessage objects
        4. HumanMessage: user_prompt (contains query + all context embedded)
        
        Args:
            system_prompt: System-level instructions for the agent
            user_prompt: User's query with all context embedded in one string
            messages_list: Conversation history from state.get("messages", [])
            look_back_limit: Number of previous messages to include (default: 10)
            state: Optional state dictionary for timezone-aware datetime context
            
        Returns:
            List of LangChain message objects ready for LLM
            
        Example usage:
            messages_list = state.get("messages", [])
            prompt = f"USER QUERY: {query}\n\n**CONTEXT**:\n{context}..."
            llm_messages = self._build_conversational_agent_messages(
                system_prompt=system_prompt,
                user_prompt=prompt,
                messages_list=messages_list,
                look_back_limit=10,
                state=state
            )
        """
        return MessagePreprocessor.build_conversational_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            messages_list=messages_list,
            look_back_limit=look_back_limit,
            datetime_context=self._get_datetime_context(state),
            sanitize_ai_responses=False,
        )
    
    def _build_editing_agent_messages(
        self,
        system_prompt: str,
        context_parts: List[str],
        current_request: str,
        messages_list: List[Any],
        look_back_limit: int = 6,
        state: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        Build message list for editing agents with conversation history and separate context
        
        STANDARDIZED METHOD FOR EDITING AGENTS (Rules, Outline, Style, Character, Fiction)
        
        Message Structure:
        1. SystemMessage: system_prompt
        2. SystemMessage: datetime_context (automatic, uses user timezone if available)
        3. Conversation history as alternating HumanMessage/AIMessage objects
        4. HumanMessage: file context (from context_parts - file content, references)
        5. HumanMessage: current_request (user query + mode-specific instructions)
        
        Args:
            system_prompt: System-level instructions for the agent
            context_parts: List of context strings (file content, references, etc.)
            current_request: User's request with mode-specific instructions
            messages_list: Conversation history from state.get("messages", [])
            look_back_limit: Number of previous messages to include (default: 6)
            state: Optional state dictionary for timezone-aware datetime context
            
        Returns:
            List of LangChain message objects ready for LLM
            
        Example usage:
            context_parts = [
                "=== FILE CONTEXT ===\n",
                file_content,
                "\n=== REFERENCES ===\n",
                references
            ]
            request = f"USER REQUEST: {query}\n\n**INSTRUCTIONS**:..."
            messages = self._build_editing_agent_messages(
                system_prompt=system_prompt,
                context_parts=context_parts,
                current_request=request,
                messages_list=state.get("messages", []),
                look_back_limit=6,
                state=state
            )
        """
        return MessagePreprocessor.build_editing_messages(
            system_prompt=system_prompt,
            context_parts=context_parts,
            current_request=current_request,
            messages_list=messages_list,
            look_back_limit=look_back_limit,
            datetime_context=self._get_datetime_context(state),
            sanitize_ai_responses=True,
        )

    def _create_error_response(self, error_message: str, task_status: TaskStatus = TaskStatus.ERROR) -> Dict[str, Any]:
        """Create standardized error response"""
        # Check if this is an OpenRouterError with user-friendly message
        if isinstance(error_message, OpenRouterError):
            return {
                "task_status": task_status.value,
                "response": error_message.user_message,
                "error_message": error_message.user_message,
                "error_type": error_message.error_type,
                "timestamp": datetime.now().isoformat()
            }
        # Check if it's a string that might contain OpenRouter error info
        elif isinstance(error_message, str):
            # Check for OpenRouter error patterns in the string
            if any(keyword in error_message.lower() for keyword in ["openrouter", "data policy", "free model training", "no endpoints found"]):
                # Try to extract and format as OpenRouter error
                temp_error = NotFoundError(error_message) if "404" in error_message or "not found" in error_message.lower() else APIError(error_message)
                error_info = self._handle_openrouter_error(temp_error)
                return {
                    "task_status": task_status.value,
                    "response": error_info["error_message"],
                    "error_message": error_info["error_message"],
                    "error_type": error_info["error_type"],
                    "timestamp": datetime.now().isoformat()
                }
        
        return {
            "task_status": task_status.value,
            "response": f"Error: {error_message}",
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        }

    async def process(
        self, 
        query: str, 
        metadata: Dict[str, Any] = None, 
        messages: List[Any] = None,
        cancellation_token: Optional[asyncio.Event] = None
    ) -> Dict[str, Any]:
        """
        Process agent request - to be implemented by subclasses
        
        Args:
            query: User query string
            metadata: Optional metadata dictionary (persona, editor context, etc.)
            messages: Optional conversation history
            cancellation_token: Optional asyncio.Event that will be set when cancellation is requested
            
        Returns:
            Dictionary with agent response
        """
        raise NotImplementedError("Subclasses must implement process() method")
