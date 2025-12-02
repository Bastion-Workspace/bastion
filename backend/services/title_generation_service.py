"""
Title Generation Service for Chat Conversations
Uses OpenRouter LLM to generate meaningful titles based on initial user messages
"""

import asyncio
import logging
from typing import Optional
from openai import AsyncOpenAI
from config import settings

logger = logging.getLogger(__name__)


class TitleGenerationService:
    """Service for generating conversation titles using LLM"""
    
    def __init__(self):
        self.openrouter_client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenRouter client with automatic reasoning support"""
        try:
            from utils.openrouter_client import get_openrouter_client
            self.openrouter_client = get_openrouter_client()
            logger.info("✅ Title Generation Service - OpenRouter client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenRouter client: {e}")
            self.openrouter_client = None
    
    async def generate_title(self, initial_message: str, model: str = None) -> str:
        """
        Generate a conversation title based on the initial user message
        
        Args:
            initial_message: The first message from the user
            model: The LLM model to use for generation
            
        Returns:
            Generated title or fallback title if generation fails
        """
        try:
            if not self.openrouter_client:
                logger.warning("⚠️ OpenRouter client not available, using fallback title")
                return self._generate_fallback_title(initial_message)
                
            if not model:
                # **ROOSEVELT FIX**: Use user-configured classification model from Settings
                from services.settings_service import settings_service
                try:
                    model = await settings_service.get_classification_model()
                except Exception:
                    logger.warning("⚠️ Failed to get classification model for title generation, using fallback title")
                    return self._generate_fallback_title(initial_message)
                
                if not model:
                    logger.warning("⚠️ No fast model configured for title generation, using fallback title")
                    return self._generate_fallback_title(initial_message)
            
            # Import datetime context utility
            from utils.system_prompt_utils import add_datetime_context_to_system_prompt
            
            # Prepare the prompt for title generation
            system_prompt = add_datetime_context_to_system_prompt(
                """You are a helpful assistant that generates concise, descriptive titles for chat conversations based on the user's initial message.

Rules:
- Generate a title that captures the main topic or intent of the message
- Keep it under 50 characters
- Make it descriptive but concise
- Don't include quotes or special formatting
- Focus on the key subject or question being asked

Examples:
- User: "How do I set up a Docker container for my React app?" → "Docker React App Setup"
- User: "Explain quantum computing concepts" → "Quantum Computing Concepts"
- User: "Can you help me debug this Python error?" → "Python Error Debugging"
- User: "What are the best practices for database design?" → "Database Design Best Practices"

Return only the title, nothing else."""
            )

            user_prompt = f"Generate a title for this conversation: {initial_message}"
            
            logger.info(f"🔤 Generating title for message: {initial_message[:100]}...")
            
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
                return self._generate_fallback_title(initial_message)
                
        except asyncio.TimeoutError:
            logger.warning("⚠️ Title generation timed out, using fallback")
            return self._generate_fallback_title(initial_message)
        except Exception as e:
            logger.error(f"❌ Failed to generate title: {e}")
            return self._generate_fallback_title(initial_message)
    
    def _generate_fallback_title(self, initial_message: str) -> str:
        """Generate a fallback title when LLM generation fails"""
        # Extract first few words from the message
        words = initial_message.strip().split()
        if len(words) <= 5:
            title = " ".join(words)
        else:
            title = " ".join(words[:5]) + "..."
        
        # Limit length and clean up
        if len(title) > 50:
            title = title[:47] + "..."
        
        return title.capitalize() 