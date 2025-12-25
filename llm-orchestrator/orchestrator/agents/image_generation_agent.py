"""
Image Generation Agent
Generates images from user prompts using OpenRouter image models via backend gRPC tool
"""

import json
import logging
from typing import Dict, Any, Optional, List, TypedDict
from langchain_core.messages import AIMessage, HumanMessage

from langgraph.graph import StateGraph, END
from orchestrator.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ImageGenerationState(TypedDict):
    """State for image generation agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    persona: Optional[Dict[str, Any]]
    prompt: str
    generation_params: Dict[str, Any]
    response: Dict[str, Any]
    task_status: str
    error: str


class ImageGenerationAgent(BaseAgent):
    """
    Image Generation Agent
    Generates images from user prompts using configured OpenRouter image models
    Uses LangGraph workflow for explicit state management
    """
    
    def __init__(self):
        super().__init__("image_generation_agent")
        self._grpc_client = None
        logger.info("🎨 Image Generation Agent ready!")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for image generation agent"""
        workflow = StateGraph(ImageGenerationState)
        
        # Add nodes
        workflow.add_node("prepare_context", self._prepare_context_node)
        workflow.add_node("generate_image", self._generate_image_node)
        
        # Entry point
        workflow.set_entry_point("prepare_context")
        
        # Linear flow: prepare_context -> generate_image -> END
        workflow.add_edge("prepare_context", "generate_image")
        workflow.add_edge("generate_image", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    async def _get_grpc_client(self):
        """Get or create gRPC client for backend tools"""
        if self._grpc_client is None:
            from orchestrator.clients.grpc_tool_client import get_grpc_tool_client
            self._grpc_client = await get_grpc_tool_client()
        return self._grpc_client
    
    async def _prepare_context_node(self, state: ImageGenerationState) -> Dict[str, Any]:
        """Prepare context: extract prompt and generation parameters"""
        try:
            logger.info("📋 Preparing context for image generation...")
            
            messages = state.get("messages", []) or []
            shared_memory = state.get("shared_memory", {}) or {}
            
            # Extract user prompt from messages
            user_prompt = ""
            for msg in reversed(messages):
                # Check LangChain message objects
                content = getattr(msg, "content", None)
                role = getattr(msg, "type", None)
                if content and (role == "human" or role is None):
                    user_prompt = str(content)
                    break
                
                # Check dict messages
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_prompt = str(msg.get("content") or "")
                    break
            
            user_prompt = user_prompt.strip()
            
            # Extract optional parameters from shared_memory
            gen_params = (shared_memory.get("image_generation_params") or {}) if isinstance(shared_memory, dict) else {}
            generation_params = {
                "size": str(gen_params.get("size", "1024x1024")),
                "format": str(gen_params.get("format", "png")),
                "seed": gen_params.get("seed"),
                "num_images": int(gen_params.get("num_images", 1)),
                "negative_prompt": gen_params.get("negative_prompt")
            }
            
            return {
                "prompt": user_prompt,
                "generation_params": generation_params,
                # ✅ CRITICAL: Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "persona": state.get("persona")
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare context: {e}")
            return {
                "prompt": "",
                "generation_params": {},
                "error": str(e),
                # ✅ CRITICAL: Preserve critical state keys even on error
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "persona": state.get("persona")
            }
    
    async def _generate_image_node(self, state: ImageGenerationState) -> Dict[str, Any]:
        """Generate image using gRPC tool"""
        try:
            logger.info("🎨 Generating image...")
            
            prompt = state.get("prompt", "").strip()
            if not prompt:
                return {
                    "response": self._create_error_result("No prompt provided for image generation"),
                    "task_status": "error",
                    # ✅ CRITICAL: Preserve critical state keys even on early return
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", ""),
                    "persona": state.get("persona")
                }
            
            generation_params = state.get("generation_params", {})
            persona = state.get("persona")
            
            # Call backend gRPC tool for image generation
            grpc_client = await self._get_grpc_client()
            
            num_images = generation_params.get("num_images", 1)
            logger.info(f"🎨 Generating {num_images} image(s) with prompt: {prompt[:100]}...")
            
            # Call GenerateImage gRPC method
            user_id = state.get("user_id", "system")
            tool_result_json = await grpc_client.generate_image(
                prompt=prompt,
                size=generation_params.get("size", "1024x1024"),
                format=generation_params.get("format", "png"),
                seed=generation_params.get("seed"),
                num_images=num_images,
                negative_prompt=generation_params.get("negative_prompt"),
                user_id=user_id
            )
            
            tool_result = json.loads(tool_result_json) if isinstance(tool_result_json, str) else tool_result_json
            
            if not tool_result.get("success"):
                error_msg = tool_result.get("error", "Unknown error")
                logger.error(f"❌ Image generation failed: {error_msg}")
                return {
                    "response": self._create_error_result(f"Image generation failed: {error_msg}"),
                    "task_status": "error",
                    # ✅ CRITICAL: Preserve critical state keys even on error
                    "metadata": state.get("metadata", {}),
                    "user_id": state.get("user_id", "system"),
                    "shared_memory": state.get("shared_memory", {}),
                    "messages": state.get("messages", []),
                    "query": state.get("query", ""),
                    "persona": state.get("persona")
                }
            
            # Extract image URLs from result
            images = tool_result.get("images", [])
            urls = [img.get("url") for img in images if isinstance(img, dict) and img.get("url")]
            urls_text = "\n".join(urls) if urls else ""
            
            # Build persona-aware response
            response_text = self._format_completion_message(
                len(images), 
                urls_text, 
                persona
            )
            
            logger.info(f"✅ Successfully generated {len(images)} image(s)")
            
            # Build structured response
            structured = {
                "task_status": "complete",
                "prompt": prompt,
                "model_used": tool_result.get("model", "unknown"),
                "images": images,
                "message": response_text,
            }
            
            return {
                "response": {
                "messages": [AIMessage(content=response_text)],
                "agent_results": {
                    "agent_type": "image_generation_agent",
                    "structured_response": structured,
                    "task_status": "complete",
                    "is_complete": True
                },
                "is_complete": True,
                "shared_memory": state.get("shared_memory", {})
                },
                "task_status": "complete",
                # ✅ CRITICAL: Preserve critical state keys
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "persona": state.get("persona")
            }
        
        except Exception as e:
            logger.error(f"❌ Image generation failed: {e}")
            return {
                "response": self._create_error_result(f"Image generation failed: {str(e)}"),
                "task_status": "error",
                "error": str(e),
                # ✅ CRITICAL: Preserve critical state keys even on exception
                "metadata": state.get("metadata", {}),
                "user_id": state.get("user_id", "system"),
                "shared_memory": state.get("shared_memory", {}),
                "messages": state.get("messages", []),
                "query": state.get("query", ""),
                "persona": state.get("persona")
            }
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """
        Process image generation request using LangGraph workflow
        
        Args:
            query: User query string
            metadata: Optional metadata dictionary (persona, editor context, etc.)
            messages: Optional conversation history
            
        Returns:
            Dictionary with image generation response and metadata
        """
        try:
            logger.info(f"🎨 Image Generation Agent: Starting image generation: {query[:100]}...")
            
            # Extract user_id and shared_memory from metadata
            metadata = metadata or {}
            user_id = metadata.get("user_id", "system")
            shared_memory = metadata.get("shared_memory", {}) or {}
            
            # Prepare new messages (current query)
            new_messages = self._prepare_messages_with_query(messages, query)
            
            # Get workflow to access checkpoint
            workflow = await self._get_workflow()
            config = self._get_checkpoint_config(metadata)
            
            # Load and merge checkpointed messages to preserve conversation history
            conversation_messages = await self._load_and_merge_checkpoint_messages(
                workflow, config, new_messages
            )
            
            # Load shared_memory from checkpoint if available
            checkpoint_state = await workflow.aget_state(config)
            existing_shared_memory = {}
            if checkpoint_state and checkpoint_state.values:
                existing_shared_memory = checkpoint_state.values.get("shared_memory", {})
            
            # Merge with any shared_memory from incoming metadata
            shared_memory_merged = shared_memory.copy()
            shared_memory_merged.update(existing_shared_memory)
            
            # Extract persona from metadata or shared_memory
            persona = metadata.get("persona") or shared_memory_merged.get("persona")
            
            # Build initial state for LangGraph workflow
            initial_state: ImageGenerationState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": conversation_messages,
                "shared_memory": shared_memory_merged,
                "persona": persona,
                "prompt": "",
                "generation_params": {},
                "response": {},
                "task_status": "",
                "error": ""
            }
            
            # Invoke LangGraph workflow with checkpointing
            final_state = await workflow.ainvoke(initial_state, config=config)
            
            # Return response from final state
            return final_state.get("response", {
                "messages": [AIMessage(content="Image generation failed")],
                "agent_results": {
                    "agent_type": self.agent_type,
                    "is_complete": False
                },
                "is_complete": False
            })
            
        except Exception as e:
            logger.error(f"❌ Image Generation Agent failed: {e}")
            return self._create_error_result(f"Image generation failed: {str(e)}")
    
    def _format_completion_message(
        self, 
        num_images: int, 
        urls_text: str, 
        persona: Optional[Dict[str, Any]]
    ) -> str:
        """Format completion message based on persona style"""
        style = str((persona or {}).get("persona_style") or "professional").lower()
        
        if style == "friendly":
            base = f"All set! Generated {num_images} image(s)."
        elif style == "snarky":
            base = f"Done. Generated {num_images} image(s)."
        elif style == "sycophantic":
            base = f"Delighted to deliver {num_images} image(s)."
        elif style == "rude_insulting":
            # Keep output neutral even for abrasive styles in utility responses
            base = f"Generated {num_images} image(s)."
        else:
            base = f"Your image request is complete. Generated {num_images} image(s)."
        
        return (base + ("\n" + urls_text if urls_text else "")).strip()
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        logger.error(f"❌ Image generation error: {error_message}")
        return {
            "messages": [AIMessage(content=error_message)],
            "agent_results": {
                "agent_type": "image_generation_agent",
                "task_status": "error",
                "error_message": error_message,
                "is_complete": True
            },
            "is_complete": True
        }


# Singleton instance
_image_generation_agent_instance = None


def get_image_generation_agent() -> ImageGenerationAgent:
    """Get global image generation agent instance"""
    global _image_generation_agent_instance
    if _image_generation_agent_instance is None:
        _image_generation_agent_instance = ImageGenerationAgent()
    return _image_generation_agent_instance

