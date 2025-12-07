"""
Title Generation Service for Chat Conversations
Uses OpenRouter LLM to generate meaningful titles based on user messages and agent responses
"""

import asyncio
import logging
import os
from typing import Optional
from orchestrator.utils.openrouter_client import get_openrouter_client

logger = logging.getLogger(__name__)


class TitleGenerationService:
    """Service for generating conversation titles using LLM"""
    
    def __init__(self):
        self.openrouter_client = None
        self._classification_model = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenRouter client with automatic reasoning support"""
        try:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                logger.warning("⚠️ OPENROUTER_API_KEY not set, title generation will use fallback")
                return
            
            self.openrouter_client = get_openrouter_client(api_key=api_key)
            logger.info("✅ Title Generation Service - OpenRouter client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenRouter client: {e}")
            self.openrouter_client = None
    
    def _get_classification_model(self) -> str:
        """
        Get classification model from environment
        
        Falls back to fast model if not specified.
        Matches backend's settings_service.get_classification_model()
        """
        if self._classification_model is None:
            # Try environment variable first
            self._classification_model = os.getenv(
                "CLASSIFICATION_MODEL",
                "anthropic/claude-haiku-4.5"  # Fast model default
            )
            logger.debug(f"Title generation model: {self._classification_model}")
        return self._classification_model
    
    async def generate_title(
        self, 
        user_message: str, 
        agent_response: Optional[str] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Generate a conversation title based on user message and optionally agent response
        
        Args:
            user_message: The user's message
            agent_response: Optional agent response for better context
            model: The LLM model to use for generation (defaults to classification model)
            
        Returns:
            Generated title or fallback title if generation fails
        """
        try:
            if not self.openrouter_client:
                logger.warning("⚠️ OpenRouter client not available, using fallback title")
                return self._generate_fallback_title(user_message)
            
            if not model:
                model = self._get_classification_model()
                if not model:
                    logger.warning("⚠️ No classification model configured, using fallback title")
                    return self._generate_fallback_title(user_message)
            
            # Build prompt with user message and optionally agent response
            if agent_response:
                # Use both user message and agent response for better context
                context = f"User: {user_message}\n\nAssistant: {agent_response[:200]}"
                user_prompt = f"Generate a title for this conversation:\n\n{context}"
            else:
                # Just use user message
                user_prompt = f"Generate a title for this conversation: {user_message}"
            
            # Prepare the system prompt
            system_prompt = """You are a helpful assistant that generates concise, descriptive titles for chat conversations.

Rules:
- Generate a title that captures the main topic or intent of the conversation
- Keep it under 50 characters
- Make it descriptive but concise
- Don't include quotes or special formatting
- Focus on the key subject or question being asked
- If both user message and assistant response are provided, use both for context

Examples:
- User: "How do I set up a Docker container for my React app?" → "Docker React App Setup"
- User: "Explain quantum computing concepts" → "Quantum Computing Concepts"
- User: "Can you help me debug this Python error?" → "Python Error Debugging"
- User: "What are the best practices for database design?" → "Database Design Best Practices"

Return only the title, nothing else."""
            
            logger.info(f"🔤 Generating title for message: {user_message[:100]}...")
            
            response = await asyncio.wait_for(
                self.openrouter_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=50,
                    temperature=0.3  # Lower temperature for more consistent titles
                ),
                timeout=10.0  # Quick timeout for title generation
            )
            
            if response.choices and len(response.choices) > 0:
                title = response.choices[0].message.content.strip()
                # Clean up the title
                title = title.replace('"', '').replace("'", "").strip()
                if len(title) > 60:
                    title = title[:57] + "..."
                
                logger.info(f"✅ Generated title: {title}")
                return title
            else:
                logger.warning("⚠️ LLM returned no choices, using fallback")
                return self._generate_fallback_title(user_message)
                
        except asyncio.TimeoutError:
            logger.warning("⚠️ Title generation timed out, using fallback")
            return self._generate_fallback_title(user_message)
        except Exception as e:
            logger.error(f"❌ Failed to generate title: {e}")
            return self._generate_fallback_title(user_message)
    
    def _generate_fallback_title(self, user_message: str) -> str:
        """Generate a fallback title when LLM generation fails"""
        # Extract first few words from the message
        words = user_message.strip().split()
        if len(words) <= 5:
            title = " ".join(words)
        else:
            title = " ".join(words[:5]) + "..."
        
        # Limit length and clean up
        if len(title) > 50:
            title = title[:47] + "..."
        
        return title.capitalize()


# Global service instance
_title_generation_service: Optional[TitleGenerationService] = None


def get_title_generation_service() -> TitleGenerationService:
    """Get or create the global title generation service"""
    global _title_generation_service
    
    if _title_generation_service is None:
        _title_generation_service = TitleGenerationService()
    
    return _title_generation_service

